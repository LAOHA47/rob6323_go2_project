# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfRandomUniformTerrainCfg
from isaaclab.utils import configclass


@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    """Direct RL env config for Unitree Go2 velocity tracking with mild rough terrain + friction bonus."""

    # --- Core RL timing ---
    decimation = 4
    episode_length_s = 20.0

    # --- Policy I/O ---
    action_scale = 0.25
    action_space = 12
    observation_space = 52
    state_space = 0

    debug_vis = True

    # --- Simulation ---
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # --- Terrain (Bonus Part 2) ---
    # We keep env origins on the default env grid (use_terrain_origins=False) to avoid "robots disappear" issues.
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        collision_group=-1,
        env_spacing=4.0,
        use_terrain_origins=False,
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            curriculum=False,
            size=(280.0, 280.0),
            border_width=10.0,
            border_height=1.0,
            num_rows=1,
            num_cols=1,
            color_scheme="none",
            horizontal_scale=0.20,
            vertical_scale=0.002,
            slope_threshold=None,
            difficulty_range=(1.0, 1.0),
            use_cache=False,
            sub_terrains={
                # Small random height perturbations (±0.05 m) while preserving the original locomotion logic.
                "mild_random_uniform": HfRandomUniformTerrainCfg(
                    proportion=1.0,
                    noise_range=(-0.05, 0.05),
                    noise_step=0.005,
                    downsampled_scale=1.0,
                )
            },
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # --- Robot + actuator model ---
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=0.0,
        damping=0.0,
    )

    # PD gains used in the environment controller (torque = Kp*(q_des-q) - Kd*qdot - friction)
    Kp = 20.0
    Kd = 0.5
    torque_limits = 23.5

    # --- Scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # Contact sensor is used for termination + contact-based reward terms
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # --- Command ranges (target linear xy + yaw rate) ---
    command_x_range = (-1.0, 1.0)
    command_y_range = (-0.5, 0.5)
    command_yaw_range = (-1.0, 1.0)

    # Termination threshold for falling below minimum base height
    base_height_min = 0.05

    # --- Debug markers (desired vs current velocity arrows) ---
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # --- Reward weights (Part 6: reimplemented terms) ---
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5

    action_rate_reward_scale = -0.060
    torque_reward_scale = -1.20e-5
    raibert_heuristic_reward_scale = -28.0

    orient_reward_scale = -3.2
    ang_vel_xy_reward_scale = -9.0e-4
    dof_vel_reward_scale = -8.5e-5
    lin_vel_z_reward_scale = -0.10

    feet_clearance_reward_scale = -330.0
    tracking_contacts_shaped_force_reward_scale = 0.0

    # --- Bonus Part 1: per-episode joint friction randomization ---
    friction_mu_v_range = (0.0, 0.3)   # viscous friction coefficient range
    friction_Fs_range = (0.0, 2.5)     # stiction magnitude range
