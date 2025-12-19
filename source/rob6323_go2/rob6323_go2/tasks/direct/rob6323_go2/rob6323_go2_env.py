# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
import math

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils

from .rob6323_go2_env_cfg import Rob6323Go2EnvCfg


class Rob6323Go2Env(DirectRLEnv):
    """Go2 velocity-tracking locomotion with contact sensing and actuator friction bonus."""

    cfg: Rob6323Go2EnvCfg

    def __init__(self, cfg: Rob6323Go2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # --- Buffers: actions/commands/episode logging ---
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "torque",
                "rew_action_rate",
                "raibert_heuristic",
                "rew_feet_clearance",
                "rew_tracking_contacts_shaped_force",
                "orient",
                "lin_vel_z",
                "dof_vel",
                "ang_vel_xy",
            ]
        }

        # --- PD control parameters ---
        self.Kp = torch.full((self.num_envs, 12), float(cfg.Kp), device=self.device)
        self.Kd = torch.full((self.num_envs, 12), float(cfg.Kd), device=self.device)
        self.torque_limits = float(cfg.torque_limits)
        self._torques = torch.zeros(self.num_envs, 12, device=self.device)

        # --- Bonus Part 1: randomized friction parameters (per env, per episode) ---
        self.friction_mu_v = torch.zeros(self.num_envs, 1, device=self.device)  # viscous
        self.friction_Fs = torch.zeros(self.num_envs, 1, device=self.device)    # stiction

        # --- Action history for smoothness penalties ---
        self.last_actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # --- Sensor/body id bookkeeping ---
        base_ids, _ = self._contact_sensor.find_bodies("base")
        if len(base_ids) == 0:
            raise RuntimeError("Could not find 'base' body in contact sensor.")
        self._base_id = base_ids[0]

        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

        self._feet_ids: list[int] = []
        for name in foot_names:
            body_ids, _ = self.robot.find_bodies(name)
            if len(body_ids) == 0:
                raise RuntimeError(f"Could not find '{name}' body in robot articulation.")
            self._feet_ids.append(body_ids[0])

        feet_ids_sensor: list[int] = []
        for name in foot_names:
            body_ids, _ = self._contact_sensor.find_bodies(name)
            if len(body_ids) == 0:
                raise RuntimeError(f"Could not find '{name}' body in contact sensor.")
            feet_ids_sensor.append(body_ids[0])
        self._feet_ids_sensor = torch.tensor(feet_ids_sensor, device=self.device, dtype=torch.long)

        # --- Gait clock / desired contact state (used by reward shaping) ---
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.foot_indices = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

        self._step_contact_targets()
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        # --- Scene composition: robot + contact sensor + terrain ---
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    @property
    def foot_positions_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self._feet_ids]

    def _step_contact_targets(self):
        # --- Periodic gait phase + smooth desired contact probabilities ---
        frequencies = 3.0
        phases = 0.5
        offsets = 0.0
        bounds = 0.0

        durations = 0.5 * torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        self.gait_indices = torch.remainder(self.gait_indices + self.step_dt * frequencies, 1.0)

        foot_indices = [
            self.gait_indices + phases + offsets + bounds,
            self.gait_indices + offsets,
            self.gait_indices + bounds,
            self.gait_indices + phases,
        ]

        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1.0) < durations
            swing_idxs = torch.remainder(idxs, 1.0) > durations

            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1.0) * (0.5 / durations[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1.0) - durations[swing_idxs]) * (
                0.5 / (1.0 - durations[swing_idxs])
            )

        self.clock_inputs[:, 0] = torch.sin(2.0 * math.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2.0 * math.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2.0 * math.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2.0 * math.pi * foot_indices[3])

        kappa = 0.07
        smoothing_cdf = torch.distributions.normal.Normal(0.0, kappa).cdf

        def _smooth_contact(fi: torch.Tensor) -> torch.Tensor:
            r = torch.remainder(fi, 1.0)
            return smoothing_cdf(r) * (1.0 - smoothing_cdf(r - 0.5)) + smoothing_cdf(r - 1.0) * (
                1.0 - smoothing_cdf(r - 1.5)
            )

        self.desired_contact_states[:, 0] = _smooth_contact(foot_indices[0])
        self.desired_contact_states[:, 1] = _smooth_contact(foot_indices[1])
        self.desired_contact_states[:, 2] = _smooth_contact(foot_indices[2])
        self.desired_contact_states[:, 3] = _smooth_contact(foot_indices[3])

    def _reward_raibert_heuristic(self) -> torch.Tensor:
        # --- Kinematic foot placement heuristic (encourages stable stepping geometry) ---
        cur_footsteps_translated = self.foot_positions_w - self.robot.data.root_pos_w.unsqueeze(1)

        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        root_quat_conj = math_utils.quat_conjugate(self.robot.data.root_quat_w)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = math_utils.quat_apply_yaw(
                root_quat_conj, cur_footsteps_translated[:, i, :]
            )

        desired_stance_width = 0.25
        desired_ys_nom = torch.tensor(
            [desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2],
            device=self.device,
        ).unsqueeze(0)

        desired_stance_length = 0.45
        desired_xs_nom = torch.tensor(
            [desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2],
            device=self.device,
        ).unsqueeze(0)

        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) - 0.5
        frequencies = torch.tensor([3.0], device=self.device)

        x_vel_des = self._commands[:, 0:1]
        yaw_vel_des = self._commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2

        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1.0
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)
        err = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])
        return torch.sum(torch.square(err), dim=(1, 2))

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # --- Actions: normalized policy output -> joint position targets around default pose ---
        self._actions = actions.clone()
        self._step_contact_targets()
        self.desired_joint_pos = self.cfg.action_scale * self._actions + self.robot.data.default_joint_pos

    def _apply_action(self) -> None:
        # --- Low-level controller: PD torque + (stiction + viscous) friction, then clipped ---
        qdot = self.robot.data.joint_vel
        torques = self.Kp * (self.desired_joint_pos - self.robot.data.joint_pos) - self.Kd * qdot

        tau_stiction = self.friction_Fs * torch.tanh(qdot / 0.1)
        tau_viscous = self.friction_mu_v * qdot
        torques = torques - (tau_stiction + tau_viscous)

        torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
        self._torques = torques
        self.robot.set_joint_effort_target(torques)

    def _get_observations(self) -> dict:
        # --- Observations: base state + command + joint state + last action + gait clocks ---
        obs = torch.cat(
            [
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_ang_vel_b,
                self.robot.data.projected_gravity_b,
                self._commands,
                self.robot.data.joint_pos - self.robot.data.default_joint_pos,
                self.robot.data.joint_vel,
                self._actions,
                self.clock_inputs,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # --- Tracking rewards (exp) ---
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1)
        yaw_rate_error = torch.square(self._commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

        # --- Regularization / stability terms ---
        rew_torque = torch.sum(torch.square(self._torques), dim=1)

        rew_action_rate = torch.sum(torch.square(self._actions - self.last_actions[:, :, 0]), dim=1) * (self.cfg.action_scale**2)
        rew_action_rate += torch.sum(
            torch.square(self._actions - 2.0 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1
        ) * (self.cfg.action_scale**2)

        self.last_actions = torch.roll(self.last_actions, shifts=1, dims=2)
        self.last_actions[:, :, 0] = self._actions

        rew_orient = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
        rew_lin_vel_z = torch.square(self.robot.data.root_lin_vel_b[:, 2])
        rew_dof_vel = torch.sum(torch.square(self.robot.data.joint_vel), dim=1)
        rew_ang_vel_xy = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1)

        # --- Task-specific shaping terms (Part 6) ---
        rew_raibert_heuristic = self._reward_raibert_heuristic()

        phases = 1.0 - torch.abs(1.0 - torch.clip((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = self.foot_positions_w[:, :, 2].view(self.num_envs, -1)
        target_height = 0.08 * phases + 0.02
        swing_mask = (1.0 - self.desired_contact_states)
        rew_feet_clearance = torch.sum(torch.square(target_height - foot_height) * swing_mask, dim=1)

        contact_forces_w = self._contact_sensor.data.net_forces_w
        foot_forces = torch.norm(contact_forces_w[:, self._feet_ids_sensor, :], dim=-1)
        sigma = 100.0
        exp_term = torch.exp(-torch.square(foot_forces) / sigma)
        force_on_term = 1.0 - exp_term
        desired = self.desired_contact_states
        per_foot_contact_reward = desired * force_on_term + (1.0 - desired) * exp_term
        rew_tracking_contacts_shaped_force = torch.mean(per_foot_contact_reward, dim=1)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale,
            "torque": rew_torque * self.cfg.torque_reward_scale,
            "rew_action_rate": rew_action_rate * self.cfg.action_rate_reward_scale,
            "raibert_heuristic": rew_raibert_heuristic * self.cfg.raibert_heuristic_reward_scale,
            "rew_feet_clearance": rew_feet_clearance * self.cfg.feet_clearance_reward_scale,
            "rew_tracking_contacts_shaped_force": rew_tracking_contacts_shaped_force * self.cfg.tracking_contacts_shaped_force_reward_scale,
            "orient": rew_orient * self.cfg.orient_reward_scale,
            "lin_vel_z": rew_lin_vel_z * self.cfg.lin_vel_z_reward_scale,
            "dof_vel": rew_dof_vel * self.cfg.dof_vel_reward_scale,
            "ang_vel_xy": rew_ang_vel_xy * self.cfg.ang_vel_xy_reward_scale,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for k, v in rewards.items():
            self._episode_sums[k] += v
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # --- Termination: base contact spike, upside-down, or base too low ---
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        forces_hist = self._contact_sensor.data.net_forces_w_history
        base_force_hist = torch.norm(forces_hist[:, :, self._base_id], dim=-1)
        cstr_termination_contacts = torch.max(base_force_hist, dim=1)[0] > 1.0

        cstr_upsidedown = self.robot.data.projected_gravity_b[:, 2] > 0.0
        base_height = self.robot.data.root_pos_w[:, 2]
        cstr_base_height_min = base_height < self.cfg.base_height_min

        died = cstr_termination_contacts | cstr_upsidedown | cstr_base_height_min
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        # --- Reset: randomize commands + randomize friction + restore default states ---
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if env_ids.numel() == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._torques[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0

        self.gait_indices[env_ids] = 0.0
        self.clock_inputs[env_ids] = 0.0
        self.desired_contact_states[env_ids] = 0.0
        self.foot_indices[env_ids] = 0.0

        n = env_ids.numel()
        mu_lo, mu_hi = self.cfg.friction_mu_v_range
        fs_lo, fs_hi = self.cfg.friction_Fs_range
        self.friction_mu_v[env_ids] = torch.empty((n, 1), device=self.device).uniform_(mu_lo, mu_hi)
        self.friction_Fs[env_ids] = torch.empty((n, 1), device=self.device).uniform_(fs_lo, fs_hi)

        self._commands[env_ids] = 0.0
        self._commands[env_ids, 0] = torch.empty((n,), device=self.device).uniform_(self.cfg.command_x_range[0], self.cfg.command_x_range[1])
        self._commands[env_ids, 1] = torch.empty((n,), device=self.device).uniform_(self.cfg.command_y_range[0], self.cfg.command_y_range[1])
        self._commands[env_ids, 2] = torch.empty((n,), device=self.device).uniform_(self.cfg.command_yaw_range[0], self.cfg.command_yaw_range[1])

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        log_dict: dict[str, float] = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids]).item()
            log_dict["Episode_Reward/" + key] = episodic_sum_avg / float(self.cfg.episode_length_s)
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = self.extras.get("log", {})
        self.extras["log"].update(log_dict)
        self.extras["log"]["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._commands[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat
