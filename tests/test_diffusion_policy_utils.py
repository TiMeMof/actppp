import torch
from torch.nn import functional as F


def quat_to_6d(quat: torch.Tensor) -> torch.Tensor:
    """Replicates DiffusionPolicy.quat_to_6d."""
    quat_norm = F.normalize(quat, dim=-1)
    w, x, y, z = quat_norm[..., 0], quat_norm[..., 1], quat_norm[..., 2], quat_norm[..., 3]

    col1 = torch.stack([
        1 - 2 * (y ** 2 + z ** 2),
        2 * (x * y + z * w),
        2 * (x * z - y * w),
    ], dim=-1)
    col2 = torch.stack([
        2 * (x * y - z * w),
        1 - 2 * (x ** 2 + z ** 2),
        2 * (y * z + x * w),
    ], dim=-1)
    rot6d = torch.cat([col1, col2], dim=-1)
    return rot6d


def rot6d_to_quat(rot6d: torch.Tensor) -> torch.Tensor:
    """Replicates DiffusionPolicy.rot6d_to_quat."""
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]

    def normalize_vector(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return v / (torch.norm(v, dim=-1, keepdim=True) + eps)

    b1 = normalize_vector(a1)
    b2 = normalize_vector(a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1)
    b3 = torch.cross(b1, b2, dim=-1)

    R = torch.stack([b1, b2, b3], dim=-1)
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    quat = torch.zeros((*R.shape[:-2], 4), device=R.device, dtype=R.dtype)
    s = torch.sqrt(trace + 1.0 + 1e-8) * 2
    quat[..., 0] = 0.25 * s
    quat[..., 1] = (R[..., 2, 1] - R[..., 1, 2]) / s
    quat[..., 2] = (R[..., 0, 2] - R[..., 2, 0]) / s
    quat[..., 3] = (R[..., 1, 0] - R[..., 0, 1]) / s

    return F.normalize(quat, dim=-1)


def convert_quat_to_6d_format(actions: torch.Tensor) -> torch.Tensor:
    """Replicates DiffusionPolicy.convert_quat_to_6d_format."""
    actions_no_zeros = actions[..., :-2]
    pos_l = actions_no_zeros[..., :3]
    quat_l = actions_no_zeros[..., 3:7]
    grip_l = actions_no_zeros[..., 7:8]

    pos_r = actions_no_zeros[..., 8:11]
    quat_r = actions_no_zeros[..., 11:15]
    grip_r = actions_no_zeros[..., 15:16]

    rot6d_l = quat_to_6d(quat_l)
    rot6d_r = quat_to_6d(quat_r)

    actions_6d = torch.cat([
        pos_l, rot6d_l, grip_l,
        pos_r, rot6d_r, grip_r,
    ], dim=-1)
    return actions_6d


def convert_6d_to_quat_format(actions_6d: torch.Tensor) -> torch.Tensor:
    """Replicates DiffusionPolicy.convert_6d_to_quat_format."""
    pos_l = actions_6d[..., :3]
    rot6d_l = actions_6d[..., 3:9]
    grip_l = actions_6d[..., 9:10]

    pos_r = actions_6d[..., 10:13]
    rot6d_r = actions_6d[..., 13:19]
    grip_r = actions_6d[..., 19:20]

    quat_l = rot6d_to_quat(rot6d_l)
    quat_r = rot6d_to_quat(rot6d_r)

    zeros = torch.zeros(*actions_6d.shape[:-1], 2, device=actions_6d.device, dtype=actions_6d.dtype)
    actions_quat = torch.cat([
        pos_l, quat_l, grip_l,
        pos_r, quat_r, grip_r,
        zeros,
    ], dim=-1)
    return actions_quat


def compute_structured_noise_loss(noise_pred: torch.Tensor, noise_gt: torch.Tensor, is_pad: torch.Tensor) -> torch.Tensor:
    """Replicates DiffusionPolicy.compute_structured_noise_loss."""
    noise_pos_l_pred = noise_pred[..., :3]
    noise_rot6d_l_pred = noise_pred[..., 3:9]
    noise_grip_l_pred = noise_pred[..., 9:10]

    noise_pos_l_gt = noise_gt[..., :3]
    noise_rot6d_l_gt = noise_gt[..., 3:9]
    noise_grip_l_gt = noise_gt[..., 9:10]

    noise_pos_r_pred = noise_pred[..., 10:13]
    noise_rot6d_r_pred = noise_pred[..., 13:19]
    noise_grip_r_pred = noise_pred[..., 19:20]

    noise_pos_r_gt = noise_gt[..., 10:13]
    noise_rot6d_r_gt = noise_gt[..., 13:19]
    noise_grip_r_gt = noise_gt[..., 19:20]

    pos_loss = F.l1_loss(noise_pos_l_pred, noise_pos_l_gt, reduction='none') + \
               F.l1_loss(noise_pos_r_pred, noise_pos_r_gt, reduction='none')
    rot_loss = F.l1_loss(noise_rot6d_l_pred, noise_rot6d_l_gt, reduction='none') + \
               F.l1_loss(noise_rot6d_r_pred, noise_rot6d_r_gt, reduction='none')
    grip_loss = F.l1_loss(noise_grip_l_pred, noise_grip_l_gt, reduction='none') + \
                F.l1_loss(noise_grip_r_pred, noise_grip_r_gt, reduction='none')

    total_structured_loss = pos_loss.sum(-1) + rot_loss.sum(-1) * 2.0 + grip_loss.sum(-1)
    masked_loss = (total_structured_loss * ~is_pad).mean()
    return masked_loss


def _random_normalized_quat(batch_shape):
    q = torch.randn(*batch_shape, 4)
    return F.normalize(q, dim=-1)


def test_quat_6d_roundtrip(num_samples: int = 1024) -> None:
    quats = _random_normalized_quat((num_samples,))
    rot6d = quat_to_6d(quats)
    recon = rot6d_to_quat(rot6d)

    diff_direct = torch.norm(quats - recon, dim=-1)
    diff_neg = torch.norm(quats + recon, dim=-1)
    min_diff = torch.minimum(diff_direct, diff_neg)
    assert torch.all(min_diff < 2e-3), f"Quaternion round-trip error too large: {min_diff.max()}"


def test_action_format_roundtrip(batch_size: int = 8, horizon: int = 5) -> None:
    actions = torch.zeros(batch_size, horizon, 18)
    actions[..., :3] = torch.randn(batch_size, horizon, 3)
    actions[..., 8:11] = torch.randn(batch_size, horizon, 3)
    q_left = _random_normalized_quat((batch_size, horizon))
    q_right = _random_normalized_quat((batch_size, horizon))
    actions[..., 3:7] = q_left
    actions[..., 11:15] = q_right
    actions[..., 7:8] = torch.rand(batch_size, horizon, 1)
    actions[..., 15:16] = torch.rand(batch_size, horizon, 1)

    actions_6d = convert_quat_to_6d_format(actions)
    recon = convert_6d_to_quat_format(actions_6d)

    # positions, grippers, padding should match exactly
    assert torch.all(torch.abs(actions[..., :3] - recon[..., :3]) < 2e-3)
    assert torch.all(torch.abs(actions[..., 7:8] - recon[..., 7:8]) < 2e-3)
    assert torch.all(torch.abs(actions[..., 8:11] - recon[..., 8:11]) < 2e-3)
    assert torch.all(torch.abs(actions[..., 15:16] - recon[..., 15:16]) < 2e-3)
    assert torch.all(torch.abs(recon[..., -2:]) < 1e-5)

    # quaternions may flip sign; check accordingly
    for start in (3, 11):
        orig = actions[..., start:start + 4]
        rec = recon[..., start:start + 4]
        diff_direct = torch.norm(orig - rec, dim=-1)
        diff_neg = torch.norm(orig + rec, dim=-1)
        min_diff = torch.minimum(diff_direct, diff_neg)
        assert torch.all(min_diff < 1e-4), f"Quaternion round-trip mismatch at slice {start}: {min_diff.max()}"


def test_structured_noise_loss_zero(batch_size: int = 4, horizon: int = 6) -> None:
    noise = torch.randn(batch_size, horizon, 20)
    is_pad = torch.zeros(batch_size, horizon, dtype=torch.bool)
    loss = compute_structured_noise_loss(noise, noise.clone(), is_pad)
    assert torch.allclose(loss, torch.tensor(0.0)), f"Expected zero loss, got {loss}"


def test_structured_noise_loss_positive(batch_size: int = 4, horizon: int = 6) -> None:
    noise_gt = torch.zeros(batch_size, horizon, 20)
    noise_pred = torch.zeros_like(noise_gt)
    noise_pred[..., :3] = 1.0
    is_pad = torch.zeros(batch_size, horizon, dtype=torch.bool)
    loss = compute_structured_noise_loss(noise_pred, noise_gt, is_pad)
    assert loss > 0, "Structured noise loss should be positive when predictions differ"


def run_all_tests() -> None:
    test_quat_6d_roundtrip()
    test_action_format_roundtrip()
    test_structured_noise_loss_zero()
    test_structured_noise_loss_positive()
    print("All diffusion policy utility tests passed.")


if __name__ == "__main__":
    run_all_tests()
