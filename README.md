# ROB6323 Go2 Project — Locomotion + Bonus Extensions

This repository is a fork of the course-provided Go2 locomotion project. My contributions keep the original task logic intact (velocity-tracking quadruped locomotion), and extend it with two bonus items:

1. **Bonus 1 — Actuator friction model (stiction + viscous)** with per-episode randomization  
2. **Bonus 2 — Slightly uneven terrain** (procedural heightfield with small random perturbations)

The goal is to preserve the baseline behavior while improving realism (friction) and robustness (walking on mildly uneven ground).

---

## Summary of Major Additions

### (A) Baseline locomotion task (required part)
- Task: command-tracking locomotion for Unitree Go2
- Actions: 12D joint position offsets around the default pose
- Controller: PD torque control
- Observations: base linear/angular velocity, projected gravity, commands, joint pos/vel, previous action, and gait clock inputs
- Rewards: velocity tracking + shaping terms (Raibert heuristic, foot clearance, smooth action regularization, stability penalties, etc.)
- Termination: base contact spikes, upside-down condition, minimum base height threshold

**Rationale:** provides a standard, stable RL locomotion training setup with clear tracking objectives and regularization.

---

### (B) Bonus 1: Actuator friction model (realism)
**What was added**
- A simple joint friction torque model applied inside the PD controller:
  - **Stiction (Coulomb-like)**: `Fs * tanh(qdot / 0.1)`
  - **Viscous**: `mu_v * qdot`
- Both coefficients are randomized **per environment, per episode**:
  - `mu_v ~ Uniform(friction_mu_v_range)`
  - `Fs   ~ Uniform(friction_Fs_range)`

**Where**
- `rob6323_go2_env.py`: friction torques are subtracted from PD torques in `_apply_action()`; coefficients randomized in `_reset_idx()`
- `rob6323_go2_env_cfg.py`: friction coefficient ranges in config

**Rationale:** better matches real actuator behavior and improves robustness to modeling mismatch.

---

### (C) Bonus 2: Slightly uneven terrain (robustness)
**What was added**
- Terrain changed from a flat plane to a **procedurally generated heightfield** with **random uniform noise**.
- Terrain is intentionally mild to keep the baseline reward/obs/action structure unchanged.
- **Robots remain on the default env grid** (`use_terrain_origins=False`) to avoid "robots disappear / teleport far away" issues.

**Where**
- `rob6323_go2_env_cfg.py`: `TerrainImporterCfg` uses `terrain_type="generator"` with `HfRandomUniformTerrainCfg`

**Rationale:** introduces small disturbances so the learned gait generalizes beyond perfect flat ground.

---

## File Organization / What Changed

### `rob6323_go2_env_cfg.py`
- Terrain: switched to a generator-based mild heightfield.
- Added/updated parameters:
  - `terrain_type="generator"`
  - `HfRandomUniformTerrainCfg(noise_range=(-0.05, 0.05), noise_step=0.005)`
  - `use_terrain_origins=False`
- Bonus 1 config ranges:
  - `friction_mu_v_range = (0.0, 0.3)`
  - `friction_Fs_range = (0.0, 2.5)`
- Base termination:
  - `base_height_min = 0.05`

### `rob6323_go2_env.py`
- Added required scene registration:
  - `self.scene.sensors["contact_sensor"] = self._contact_sensor`
- Bonus 1 friction implementation:
  - Apply friction torques in `_apply_action()`
  - Randomize friction in `_reset_idx()`
- Kept original reward / observation / reset structure (only incremental additions).

---

## Reproducible Training Recipe

### Environment / Dependencies
Use the same IsaacLab + IsaacSim environment recommended by the course tutorial. Activate your conda/venv accordingly before running commands.

### Reproducibility (Seeds)
This setup uses fixed seeds in two places:
- **Terrain generator seed**: `seed=0` in `TerrainGeneratorCfg`
- **Training seed**: set explicitly in the training command (below)

> Note: GPU kernels and large-scale RL training can still have minor nondeterminism. The seed ensures consistent initialization and high-level reproducibility.

---

## How to Train

From the repository root:

### (1) Train (headless)
```bash
python -m isaaclab_rl.train \
  --task Rob6323Go2Env \
  --headless \
  --seed 0
