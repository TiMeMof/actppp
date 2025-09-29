# Copilot Instructions for ACT++ Mobile ALOHA

This is an imitation learning codebase for Mobile ALOHA robotics, implementing ACT (Action Chunking Transformer), Diffusion Policy, and VINN algorithms for bimanual manipulation tasks.

## Core Architecture

**Multi-Policy Framework**: Three policy classes are supported:
- `ACTPolicy`: Transformer-based with CVAE decoder using DETR backbone 
- `DiffusionPolicy`: Diffusion-based policy with ConditionalUnet1D
- `CNNMLPPolicy`: Simple CNN+MLP baseline

**Dual Control Modes**:
- Joint space control (`sim_env.py`) - 14D action space (6 joints per arm + grippers)
- End-effector space control (`ee_sim_env.py`) - 16D action space (position+quaternion per arm + grippers)
- Use `--use_ee` flag to switch between modes; affects state_dim calculation

**Data Flow**: `utils.EpisodicDataset` → Policy forward pass → Environment step
- Observations: qpos, qvel, ee pose, multi-camera images
- Actions normalized using dataset statistics from `dataset_stats.pkl`

## Essential Development Patterns

**Configuration System**: 
- Task configs in `constants.SIM_TASK_CONFIGS` define dataset paths, episode lengths, camera names
- Policy configs built dynamically in `imitate_episodes.main()` based on policy class

**Training Pipeline**:
```bash
# Standard ACT training
python3 imitate_episodes.py --task_name sim_transfer_cube_scripted \
    --policy_class ACT --chunk_size 100 --kl_weight 10 \
    --hidden_dim 512 --batch_size 8 --num_steps 2000
```

**Evaluation Requirements**:
- Set `export MUJOCO_GL=egl` for headless rendering
- Use `--eval` flag with pre-trained checkpoint
- Success metrics: reward thresholds defined in task classes (max_reward=4)

## Critical Integration Points

**IK/FK Integration**: 
- When `use_ee=True`: actions are end-effector poses, converted to joint angles via IK in `eval_bc()`
- FK verification in `sim_env.get_ee_pose()` for pose accuracy
- IK solver in `show_ws/src/arm_robot_description/scripts/seed_joint_states.RobotController`

**Multi-Camera System**: 
- Camera names: ['top', 'left_wrist', 'right_wrist'] for sim tasks
- Images processed through ResNet18 → SpatialSoftmax → Linear layers
- Normalization: `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

**Gripper Control**:
- Normalized positions: 0=closed, 1=open
- Use `PUPPET_GRIPPER_POSITION_NORMALIZE_FN` for conversion
- Action space includes normalized gripper values, converted in `BimanualViperXTask.before_step()`

## Development Workflows

**Environment Setup**:
```bash
conda create -n actppp python=3.8.10
pip install -r requirement.txt
cd detr && pip install -e .
# Install robomimic with specific branch for Diffusion Policy
```

**Data Generation**:
```bash
python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted \
    --dataset_dir <path> --num_episodes 50
```

**Debugging Patterns**:
- Use `IPython.embed()` (imported as `e`) for breakpoints throughout codebase
- Visualization: `python3 visualize_episodes.py --dataset_dir <path> --episode_idx 0`
- `--onscreen_render` for real-time environment rendering

**Checkpointing Strategy**:
- Best validation checkpoint saved as `policy_best.ckpt`
- Regular saves as `policy_step_{step}_seed_{seed}.ckpt`
- Model serialization via `policy.serialize()/deserialize()` methods

## Project-Specific Conventions

**Naming Patterns**:
- Task names: `sim_transfer_cube_scripted`, `sim_insertion_scripted`
- Mirrored data: `_mirror` suffix for data augmentation
- Camera IDs match physics renderer names exactly

**Error Handling**: 
- Silent failures common in RL loops - check reward signals and success rates
- IK solutions validated with `suc_l, suc_r` boolean flags
- Episode termination based on reward thresholds, not time limits

**Performance Considerations**:
- Temporal aggregation (`--temporal_agg`) uses exponential weighting for action smoothing
- Query frequency controls how often policy is called vs. action replay
- EMA models used in DiffusionPolicy for stable inference

When implementing new features, maintain compatibility with both joint and EE control modes, respect the multi-policy architecture, and follow the established checkpoint/evaluation patterns.