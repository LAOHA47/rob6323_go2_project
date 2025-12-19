# ROB6323 Go2 Project — Locomotion + Bonus Extensions

This repository is a fork of the course-provided Go2 locomotion project. Our contributions keep the original task logic intact (velocity-tracking quadruped locomotion), and extend it with two bonus items:

1. **Bonus 1 — Actuator friction model (stiction + viscous)** with per-episode randomization  
2. **Bonus 2 — Slightly uneven terrain** (procedural heightfield with small random perturbations)

The goal is to preserve the baseline behavior while improving realism (friction) and robustness (walking on mildly uneven ground).

---

## Summary of Major Additions

### (A) Baseline locomotion task 
- Task: command-tracking locomotion for Unitree Go2
- Actions: 12D joint position offsets around the default pose
- Controller: PD torque control
- Observations: base linear/angular velocity, projected gravity, commands, joint pos/vel, previous action, and gait clock inputs
- Rewards: velocity tracking + shaping terms (Raibert heuristic, foot clearance, smooth action regularization, stability penalties, etc.)
- Termination: base contact spikes, upside-down condition, minimum base height threshold

Rationale: provides a standard, stable RL locomotion training setup with clear tracking objectives and regularization.

---

### (B) Bonus 1: Actuator friction model (realism)
What was added:
- A simple joint friction torque model applied inside the PD controller:
  - Stiction (Coulomb-like): `Fs * tanh(qdot / 0.1)`
  - Viscous: `mu_v * qdot`
- Both coefficients are randomized per environment, per episode:
  - `mu_v ~ Uniform(friction_mu_v_range)`
  - `Fs   ~ Uniform(friction_Fs_range)`

Where:
- `rob6323_go2_env.py`: friction torques are subtracted from PD torques in `_apply_action()`; coefficients randomized in `_reset_idx()`
- `rob6323_go2_env_cfg.py`: friction coefficient ranges in config

Rationale: better matches real actuator behavior and improves robustness to modeling mismatch.

---

### (C) Bonus 2: Slightly uneven terrain (robustness)
What was added:
- Terrain changed from a flat plane to a procedurally generated heightfield with random uniform noise.
- Terrain is intentionally mild to keep the baseline reward/obs/action structure unchanged.
- Robots remain on the default env grid (`use_terrain_origins=False`) to avoid "robots disappear / teleport far away" issues.

Where:
- `rob6323_go2_env_cfg.py`: `TerrainImporterCfg` uses `terrain_type="generator"` with `HfRandomUniformTerrainCfg`

Rationale: introduces small disturbances so the learned gait generalizes beyond perfect flat ground.

---

## File Organization / What Changed

### `rob6323_go2_env_cfg.py`
- Terrain: switched to a generator-based mild heightfield.
- Added/updated parameters:
  - `terrain_type="generator"`
  - `HfRandomUniformTerrainCfg(noise_range=(-0.05, 0.05), noise_step=0.005)`
  - `use_terrain_origins=False`
  - Terrain generator seed set in config: `seed=0`
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
- Kept the original reward / observation / reset structure (only incremental additions).

---

## Reproducible Training Recipe (Greene HPC)

### Launch training
From `$HOME/rob6323_go2_project` on Greene, submit a training job via the provided script:

```bash
cd "$HOME"        
git clone <YOUR_FORK_URL> rob6323_go2_project
cd "$HOME/rob6323_go2_project"
./train.sh
ssh burst "squeue -u $USER"
```
Note: Jobs may be preempted/canceled/requeued on shared clusters. This is normal behavior on preemptible partitions.

## Where to find results

- When a job completes, logs are written under logs in your project clone on Greene in the structure logs/[job_id]/rsl_rl/go2_flat_direct/[date_time]/.  
- Inside each run directory you will find a TensorBoard events file (events.out.tfevents...), neural network checkpoints (model_[epoch].pt), YAML files with the exact PPO and environment parameters, and a rollout video under videos/play/ that showcases the trained policy.  

## Download logs to your computer

Use `rsync` to copy results from the cluster to your local machine. It is faster and can resume interrupted transfers. Run this on your machine (NOT on Greene):

```
rsync -avzP -e 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' <netid>@dtn.hpc.nyu.edu:/home/<netid>/rob6323_go2_project/logs ./
```

*Explanation of flags:*
- `-a`: Archive mode (preserves permissions, times, and recursive).
- `-v`: Verbose output.
- `-z`: Compresses data during transfer (faster over network).
- `-P`: Shows progress bar and allows resuming partial transfers.

## Visualize with TensorBoard

You can inspect training metrics (reward curves, loss values, episode lengths) using TensorBoard. This requires installing it on your local machine.

1.  **Install TensorBoard:**
    On your local computer (do NOT run this on Greene), install the package:
    ```
    pip install tensorboard
    ```

2.  **Launch the Server:**
    Navigate to the folder where you downloaded your logs and start the server:
    ```
    # Assuming you are in the directory containing the 'logs' folder
    tensorboard --logdir ./logs
    ```

3.  **View Metrics:**
    Open your browser to the URL shown (usually `http://localhost:6006/`).

## Notes on Seeds / Determinism

Terrain determinism: the terrain generator uses a fixed seed in rob6323_go2_env_cfg.py (seed=0).

Training seed: I did not add custom seed flags to the workflow; training reproducibility follows the original course-provided train.sh and its underlying configs. If train.sh exposes a seed option, set it there; otherwise it uses the default seed behavior from the provided training pipeline. 
    
