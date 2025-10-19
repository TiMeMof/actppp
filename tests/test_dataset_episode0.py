"""Quick integrity checks for the first EE dataset episode.

This script loads ``episode_0.hdf5`` from the EE dataset folder and
verifies that:

1. Quaternion fields are properly normalised and round-trip through the
   6D representation used by the policies.
2. Gripper commands are normalised to ``[0, 1]`` and match between
   ``action`` (joint commands) and ``action_ee`` (end-effector
   commands).
3. Basic shape consistency checks between the stored datasets hold.

Run with ``python tests/test_dataset_episode0.py`` from the project root.
"""

import glob
import os
import pathlib
import sys
from dataclasses import dataclass

import h5py
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from constants import (
    PUPPET_GRIPPER_POSITION_CLOSE,
    PUPPET_GRIPPER_POSITION_OPEN,
    SIM_TASK_CONFIGS,
)


@dataclass
class EEEpisode:
    path: str
    action: np.ndarray
    action_ee: np.ndarray
    qpos: np.ndarray
    ee: np.ndarray


def locate_first_episode() -> str:
    dataset_dir = SIM_TASK_CONFIGS["sim_transfer_cube_ee"]["dataset_dir"]
    episode_paths = sorted(glob.glob(os.path.join(dataset_dir, "episode_*.hdf5")))
    if not episode_paths:
        raise FileNotFoundError(f"No episodes found under {dataset_dir}")
    return episode_paths[0]


def load_episode(path: str) -> EEEpisode:
    with h5py.File(path, "r") as f:
        action = f["action"][:]
        action_ee = f["action_ee"][:]
        qpos = f["observations/qpos"][:]
        ee = f["observations/ee"][:]
    return EEEpisode(path=path, action=action, action_ee=action_ee, qpos=qpos, ee=ee)


def ensure_shapes(episode: EEEpisode) -> None:
    action_T = episode.action.shape[0]
    assert episode.action_ee.shape[0] == action_T, "action/action_ee length mismatch"
    assert episode.qpos.shape[0] == action_T, "qpos/action length mismatch"
    assert episode.ee.shape[0] == action_T, "ee/action length mismatch"
    assert episode.action.shape[1] == 14, "Joint action should have 14 dims"
    assert episode.action_ee.shape[1] == 16, "EE action should have 16 dims"
    assert episode.ee.shape[1] == 16, "EE observation should have 16 dims"


def quat_to_rot6d(quat: np.ndarray) -> np.ndarray:
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    col1 = np.stack((1 - 2 * (y**2 + z**2), 2 * (x * y + z * w), 2 * (x * z - y * w)), axis=-1)
    col2 = np.stack((2 * (x * y - z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z + x * w)), axis=-1)
    return np.concatenate((col1, col2), axis=-1)


def rot6d_to_quat(rot6d: np.ndarray) -> np.ndarray:
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]

    def normalise(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)

    b1 = normalise(a1)
    b2 = normalise(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1)
    b3 = np.cross(b1, b2)
    R = np.stack((b1, b2, b3), axis=-1)

    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    quat = np.zeros(rot6d.shape[:-1] + (4,), dtype=rot6d.dtype)

    positive_mask = trace > 0
    s = np.sqrt(np.maximum(trace[positive_mask] + 1.0, 1e-8)) * 2
    quat[positive_mask, 0] = 0.25 * s
    quat[positive_mask, 1] = (R[positive_mask, 2, 1] - R[positive_mask, 1, 2]) / s
    quat[positive_mask, 2] = (R[positive_mask, 0, 2] - R[positive_mask, 2, 0]) / s
    quat[positive_mask, 3] = (R[positive_mask, 1, 0] - R[positive_mask, 0, 1]) / s

    negative_mask = ~positive_mask
    if np.any(negative_mask):
        Rm = R[negative_mask]
        quat_m = quat[negative_mask]
        d0 = np.diagonal(Rm, axis1=-2, axis2=-1)

        idx0 = (d0[..., 0] > d0[..., 1]) & (d0[..., 0] > d0[..., 2])
        if np.any(idx0):
            s = np.sqrt(np.maximum(1.0 + d0[idx0, 0] - d0[idx0, 1] - d0[idx0, 2], 1e-8)) * 2
            quat_m[idx0, 0] = (Rm[idx0, 2, 1] - Rm[idx0, 1, 2]) / s
            quat_m[idx0, 1] = 0.25 * s
            quat_m[idx0, 2] = (Rm[idx0, 0, 1] + Rm[idx0, 1, 0]) / s
            quat_m[idx0, 3] = (Rm[idx0, 0, 2] + Rm[idx0, 2, 0]) / s

        idx1 = (~idx0) & (d0[..., 1] > d0[..., 2])
        if np.any(idx1):
            s = np.sqrt(np.maximum(1.0 + d0[idx1, 1] - d0[idx1, 0] - d0[idx1, 2], 1e-8)) * 2
            quat_m[idx1, 0] = (Rm[idx1, 0, 2] - Rm[idx1, 2, 0]) / s
            quat_m[idx1, 1] = (Rm[idx1, 0, 1] + Rm[idx1, 1, 0]) / s
            quat_m[idx1, 2] = 0.25 * s
            quat_m[idx1, 3] = (Rm[idx1, 1, 2] + Rm[idx1, 2, 1]) / s

        idx2 = (~idx0) & (~idx1)
        if np.any(idx2):
            s = np.sqrt(np.maximum(1.0 + d0[idx2, 2] - d0[idx2, 0] - d0[idx2, 1], 1e-8)) * 2
            quat_m[idx2, 0] = (Rm[idx2, 1, 0] - Rm[idx2, 0, 1]) / s
            quat_m[idx2, 1] = (Rm[idx2, 0, 2] + Rm[idx2, 2, 0]) / s
            quat_m[idx2, 2] = (Rm[idx2, 1, 2] + Rm[idx2, 2, 1]) / s
            quat_m[idx2, 3] = 0.25 * s

        quat[negative_mask] = quat_m

    quat /= np.linalg.norm(quat, axis=-1, keepdims=True) + 1e-8
    return quat


def quaternion_checks(label: str, quat: np.ndarray) -> None:
    norms = np.linalg.norm(quat, axis=-1)
    max_dev = np.max(np.abs(norms - 1.0))
    print(f"[{label}] quaternion max |norm - 1|: {max_dev:.3e}")
    assert max_dev < 1e-5, f"{label} quaternion is not unit-length"

    rot6d = quat_to_rot6d(quat)
    quat_roundtrip = rot6d_to_quat(rot6d)

    dot = np.sum(quat * quat_roundtrip, axis=-1, keepdims=True)
    quat_roundtrip *= np.sign(dot)
    roundtrip_err = np.max(np.abs(quat - quat_roundtrip))
    print(f"[{label}] quaternion round-trip max error: {roundtrip_err:.3e}")
    assert roundtrip_err < 1e-5, f"{label} quaternion round-trip mismatch"


def check_grippers(episode: EEEpisode) -> None:
    left_grip = episode.action[:, 6]
    right_grip = episode.action[:, 13]
    assert np.all(left_grip >= -1e-6) and np.all(left_grip <= 1 + 1e-6), "Left gripper out of [0, 1]"
    assert np.all(right_grip >= -1e-6) and np.all(right_grip <= 1 + 1e-6), "Right gripper out of [0, 1]"

    close = PUPPET_GRIPPER_POSITION_CLOSE
    open_ = PUPPET_GRIPPER_POSITION_OPEN
    unnormalised_left = left_grip * (open_ - close) + close
    unnormalised_right = right_grip * (open_ - close) + close
    print(
        f"Gripper physical range left:[{unnormalised_left.min():.5f}, {unnormalised_left.max():.5f}] "
        f"right:[{unnormalised_right.min():.5f}, {unnormalised_right.max():.5f}]"
    )

    left_ee = episode.action_ee[:, 7]
    right_ee = episode.action_ee[:, 15]
    assert np.allclose(left_grip, left_ee), "Left gripper mismatch between action and action_ee"
    assert np.allclose(right_grip, right_ee), "Right gripper mismatch between action and action_ee"


def main() -> None:
    episode_path = locate_first_episode()
    episode = load_episode(episode_path)

    print(f"Loaded episode: {episode_path}")

    ensure_shapes(episode)

    # Quaternion checks for actions and observations
    quaternion_checks("action_ee.left", episode.action_ee[:, 3:7])
    quaternion_checks("action_ee.right", episode.action_ee[:, 11:15])
    quaternion_checks("observations.ee.left", episode.ee[:, 3:7])
    quaternion_checks("observations.ee.right", episode.ee[:, 11:15])

    check_grippers(episode)

    # Optional: report mean absolute difference between commanded and observed EE poses
    left_action_pose = episode.action_ee[:, :7]
    right_action_pose = episode.action_ee[:, 8:15]
    left_obs_pose = episode.ee[:, :7]
    right_obs_pose = episode.ee[:, 8:15]

    left_diff = np.abs(left_action_pose - left_obs_pose)
    right_diff = np.abs(right_action_pose - right_obs_pose)
    print(f"Mean |command - observe| (left): {left_diff.mean():.4f}")
    print(f"Mean |command - observe| (right): {right_diff.mean():.4f}")

    print("All dataset checks passed.")


if __name__ == "__main__":
    main()