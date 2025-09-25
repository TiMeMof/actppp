import numpy as np
from dm_control import mujoco

from constants import DT, XML_DIR, START_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

def right_arm_fk(right_arm_qpos):    
# ========= 新增：基于 right_arm_qpos 的手写前向运动学（不调用 site API）=========
    # 关节顺序（与 XML 对应）:
    # q0: waist (Z轴旋转)
    # q1: shoulder (Y轴旋转)
    # q2: elbow (Y轴旋转)
    # q3: forearm_roll (X轴旋转)
    # q4: wrist_angle (Y轴旋转)
    # q5: wrist_rotate (X轴旋转)
    q0, q1, q2, q3, q4, q5 = right_arm_qpos

    def _rot(axis, a):
        ax = np.asarray(axis, dtype=float)
        ax = ax / np.linalg.norm(ax)
        x,y,z = ax
        c, s = np.cos(a), np.sin(a)
        C = 1 - c
        R = np.array([
            [c + x*x*C,     x*y*C - z*s, x*z*C + y*s, 0],
            [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s, 0],
            [z*x*C - y*s,   z*y*C + x*s, c + z*z*C,   0],
            [0,0,0,1]
        ])
        return R
    def _trans(x,y,z):
        T = np.eye(4)
        T[:3,3] = [x,y,z]
        return T

    # 各段固定平移（来自 XML 各 body 的 pos，按层级顺序）:
    # shoulder_link:        (0, 0, 0.079)
    # upper_arm_link:       (0, 0, 0.04805)
    # upper_forearm_link:   (0.05955, 0, 0.3)
    # lower_forearm_link:   (0.2, 0, 0)
    # wrist_link:           (0.1, 0, 0)
    # gripper_link:         (0.069744, 0, 0)  (right_ee_site 在该 link 原点)
    T = _trans(0.469, 0.5, 0.0) @ _rot([0,0,1], np.pi)  # base
    # 每级：Trans(body_pos) 再 Rot(axis, q)
    T = T @ _trans(0,0,0.079)      @ _rot([0,0,1], q0)  
    T = T @ _trans(0,0,0.04805)    @ _rot([0,1,0], q1)  
    T = T @ _trans(0.05955,0,0.3)  @ _rot([0,1,0], q2)  
    T = T @ _trans(0.2,0,0)        @ _rot([1,0,0], q3)  
    T = T @ _trans(0.1,0,0)        @ _rot([0,1,0], q4)  
    T = T @ _trans(0.069744,0,0)   @ _rot([1,0,0], q5)
    right_ee_pos_fk = T[:3, 3].copy()
    # site_id = physics.model.name2id("right_ee_site", "site")
    # site_pos = physics.data.site_xpos[site_id].copy()
    # print("FK 计算的末端位置err:", right_ee_pos_fk - site_pos)
    # =========================================================
    return T

def left_arm_fk(left_arm_qpos):
    q0, q1, q2, q3, q4, q5 = left_arm_qpos

    def _rot(axis, a):
        ax = np.asarray(axis, dtype=float)
        ax = ax / np.linalg.norm(ax)
        x,y,z = ax
        c, s = np.cos(a), np.sin(a)
        C = 1 - c
        R = np.array([
            [c + x*x*C,     x*y*C - z*s, x*z*C + y*s, 0],
            [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s, 0],
            [z*x*C - y*s,   z*y*C + x*s, c + z*z*C,   0],
            [0,0,0,1]
        ])
        return R

    def _trans(x,y,z):
        T = np.eye(4)
        T[:3,3] = [x,y,z]
        return T

    T = _trans(-0.469, 0.5, 0.0)
    # 每级：Trans(body_pos) 再 Rot(axis, q)
    T = T @ _trans(0,0,0.079)      @ _rot([0,0,1], q0)  
    T = T @ _trans(0,0,0.04805)    @ _rot([0,1,0], q1)  
    T = T @ _trans(0.05955,0,0.3)  @ _rot([0,1,0], q2)  
    T = T @ _trans(0.2,0,0)        @ _rot([1,0,0], q3)  
    T = T @ _trans(0.1,0,0)        @ _rot([0,1,0], q4)  
    T = T @ _trans(0.069744,0,0)   @ _rot([1,0,0], q5)
    left_ee_pos_fk = T[:3, 3].copy()
    # site_id = physics.model.name2id("left_ee_site", "site")
    # site_pos = physics.data.site_xpos[site_id].copy()
    # print("FK 计算的末端位置err:", left_ee_pos_fk - site_pos)
    # =========================================================
    return T

def matrix_to_euler(mat):
    """
    标准 ZYX 欧拉角（yaw, pitch, roll）提取公式
    输入：mat为3x3旋转矩阵展开成的长度为9的一维数组，要flatten后传入
    输出：返回顺序为 (roll, pitch, yaw)，单位：度
    ## R = Rz * Ry * Rx
    """
    # 检测mat是否为3x3：
    assert len(mat) == 9

    roll = np.arctan2(mat[7], mat[8])
    pitch = np.arctan2(-mat[6], np.sqrt(mat[7]**2 + mat[8]**2))
    yaw = np.arctan2(mat[3], mat[0])

    yaw = np.degrees(yaw)
    pitch = np.degrees(pitch)
    roll = np.degrees(roll)
    return [roll, pitch, yaw]

def quat_to_euler(quat):
    """
    将四元数 
    # (w, x, y, z)
    转换为与 matrix_to_euler 一致的 ZYX 欧拉角 (返回顺序: roll, pitch, yaw) ，单位：度
    与 matrix_to_euler 保持同一分解
    ## R = Rz * Ry * Rx
    """
    quat = np.asarray(quat, dtype=float)
    if quat.shape[-1] != 4:
        raise ValueError("quat 长度必须为4，顺序应为 [w, x, y, z]")
    # 归一化
    norm = np.linalg.norm(quat)
    if norm == 0:
        raise ValueError("零四元数无法转换")
    quat = quat / norm
    w, x, y, z = quat

    # 构造旋转矩阵（row-major）
    R = np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y)]
    ], dtype=float)

    # 按 matrix_to_euler 的同样公式
    roll = np.arctan2(R[2,1], R[2,2])
    pitch = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
    yaw = np.arctan2(R[1,0], R[0,0])

    return [np.degrees(roll), np.degrees(pitch), np.degrees(yaw)]

def get_qpos(physics:mujoco.Physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

def mat2quat(Rm):
    K = np.empty((4,4))
    K[0,0] = (1.0/3.0)*(Rm[0,0]-Rm[1,1]-Rm[2,2])
    K[0,1] = (1.0/3.0)*(Rm[1,0]+Rm[0,1])
    K[0,2] = (1.0/3.0)*(Rm[2,0]+Rm[0,2])
    K[0,3] = (1.0/3.0)*(Rm[1,2]-Rm[2,1])
    K[1,0] = (1.0/3.0)*(Rm[1,0]+Rm[0,1])
    K[1,1] = (1.0/3.0)*(Rm[1,1]-Rm[0,0]-Rm[2,2])
    K[1,2] = (1.0/3.0)*(Rm[2,1]+Rm[1,2])
    K[1,3] = (1.0/3.0)*(Rm[2,0]-Rm[0,2])
    K[2,0] = (1.0/3.0)*(Rm[2,0]+Rm[0,2])
    K[2,1] = (1.0/3.0)*(Rm[2,1]+Rm[1,2])
    K[2,2] = (1.0/3.0)*(Rm[2,2]-Rm[0,0]-Rm[1,1])
    K[2,3] = (1.0/3.0)*(Rm[0,1]-Rm[1,0])
    K[3,0] = (1.0/3.0)*(Rm[1,2]-Rm[2,1])
    K[3,1] = (1.0/3.0)*(Rm[2,0]-Rm[0,2])
    K[3,2] = (1.0/3.0)*(Rm[0,1]-Rm[1,0])
    K[3,3] = (1.0/3.0)*(Rm[0,0]+Rm[1,1]+Rm[2,2])
    # eigenvector of K with max eigenvalue
    w, v = np.linalg.eigh(K)
    q = v[:, np.argmax(w)]
    # reorder to w,x,y,z
    return np.array([q[3], q[0], q[1], q[2]])


def quat_mul(q1, q2):
    """Quaternion multiply (w,x,y,z)"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def orientation_error(R_current, tq):
    """Compute orientation error as an axis-angle (rotvec) between R_current and target quaternion tq.

    Inputs:
        R_current: 3x3 rotation matrix
        tq: target quaternion (w,x,y,z)
    Returns:
        rotvec: length-3 vector (angle * axis)
    """
    qc = mat2quat(R_current)
    qc = qc / np.linalg.norm(qc)
    qc_inv = np.array([qc[0], -qc[1], -qc[2], -qc[3]])
    q_rel = quat_mul(tq, qc_inv)
    if q_rel[0] < 0:
        q_rel = -q_rel
    w = np.clip(q_rel[0], -1.0, 1.0)
    angle = 2 * np.arccos(w)
    s = np.sqrt(max(0.0, 1 - w*w))
    if s < 1e-8:
        axis = q_rel[1:]
    else:
        axis = q_rel[1:] / s
    return angle * axis


def pose_error(q, fk_func, target_pos, tq, position_weight=1.0, orientation_weight=0.3):
    """Compute weighted 6D pose error between fk_func(q) and target (target_pos, tq).

    Returns a length-6 vector: [pos_weight * dp, ori_weight * rotvec]
    计算位置和方向误差，并加权返回
    """
    T = fk_func(q)
    dp = target_pos - T[:3, 3]
    drot = orientation_error(T[:3, :3], tq)
    return np.concatenate([position_weight * dp, orientation_weight * drot])


def numerical_jacobian(q, fk_func, position_weight=1.0, orientation_weight=0.3, eps=1e-5):
    """Numerical Jacobian of 6D pose (pos, ori) w.r.t. joint vector q.

    fk_func should return a 4x4 transformation matrix for given q.
    """
    n = len(q)
    J = np.zeros((6, n))
    e0_T = fk_func(q)
    p0 = e0_T[:3, 3]
    R0 = e0_T[:3, :3]
    for i in range(n):
        dq = np.zeros(n);
        dq[i] = eps
        T1 = fk_func(q + dq)
        p1 = T1[:3, 3]
        R1 = T1[:3, :3]
        J[:3, i] = (p1 - p0) / eps
        Rrel = R0.T @ R1
        tr = np.clip((np.trace(Rrel) - 1) / 2, -1, 1)
        ang = np.arccos(tr)
        if ang < 1e-9:
            rvec = np.zeros(3)
        else:
            rv = np.array([
                Rrel[2, 1] - Rrel[1, 2],
                Rrel[0, 2] - Rrel[2, 0],
                Rrel[1, 0] - Rrel[0, 1]
            ]) / (2 * np.sin(ang))
            rvec = ang * rv
        J[3:, i] = orientation_weight * rvec / eps
    J[:3, :] *= position_weight
    return J

def arm_ik(physics:mujoco.Physics, qpos_init, target_pos, target_quat, arm='right',
           max_iters=150, tol_pos=1e-4, tol_ori=2e-3, verbose=False,
           position_weight=1.0, orientation_weight=0.3, restarts=2):
    """改进版数值IK（阻尼LM + 关节限幅 + 多重启动）

    参数:
        physics: dm_control.mujoco.Physics 实例（用于获取当前关节角初值）
        qpos_init: 当前完整 qpos (或 None). 若提供长度>=14 的 array, 将使用其对应手臂部分作为初值.
        target_pos: (3,) 目标位置
        target_quat: (4,) 目标方向四元数 (w,x,y,z)
        arm: 'left' or 'right'
        max_iters: 单次迭代最大步数
        tol_pos, tol_ori: 位置/方向收敛阈值 (m, rad)
        position_weight, orientation_weight: 误差加权 (提高位置优先级)
        restarts: 失败时随机扰动重启次数（总共尝试 restarts+1 次）

    返回:
        长度6 的目标关节角 numpy.ndarray
    """
    assert len(target_pos) == 3 and len(target_quat) == 4
    target_pos = np.asarray(target_pos, dtype=float)
    tq = np.asarray(target_quat, dtype=float)
    tq = tq / np.linalg.norm(tq)

    # 从 physics 获取当前 qpos (格式: [left6, left_grip, right6, right_grip])
    full_qpos = get_qpos(physics) if qpos_init is None else np.asarray(qpos_init).copy()
    if arm == 'right':
        # 注意原 get_qpos 排布: left(6) + left_grip + right(6) + right_grip
        q_current = full_qpos[7:13].copy()
        # 关节限位 (来自 vx300s xml, 这里直接硬编码; 若与真实差异可放 constants)
        joint_limits = np.array([
            [-np.pi, np.pi],          # waist
            [-1.85005, 1.25664],      # shoulder
            [-1.76278, 1.6057],       # elbow
            [-np.pi, np.pi],          # forearm_roll
            [-1.8675, 2.23402],       # wrist_angle
            [-np.pi, np.pi],          # wrist_rotate
        ])
    else:
        q_current = full_qpos[0:6].copy()
        joint_limits = np.array([
            [-np.pi, np.pi],
            [-1.85005, 1.25664],
            [-1.76278, 1.6057],
            [-np.pi, np.pi],
            [-1.8675, 2.23402],
            [-np.pi, np.pi],
        ])

    def clamp(q):
        """Clamp joint angles to limits."""
        return np.clip(q, joint_limits[:,0], joint_limits[:,1])

    def fk(q):
        """Forward kinematics function for the specified arm."""
        return left_arm_fk(q) if arm == 'left' else right_arm_fk(q)

    best_q = None
    best_err = 1e4

    rng = np.random.default_rng()
    initial_seeds = [q_current]
    for _ in range(restarts):
        noise = rng.normal(0, 0.05, size=6)
        initial_seeds.append(clamp(q_current + noise))

    for seed_id, q0 in enumerate(initial_seeds):
        q = clamp(q0.copy())
        lam = 1e-2   # 初始阻尼
        prev_cost = None
        for it in range(max_iters):
            err = pose_error(q, fk, target_pos, tq, position_weight, orientation_weight)
            pos_err = np.linalg.norm(err[:3]) / max(position_weight, 1e-8)
            ori_err = np.linalg.norm(err[3:]) / max(orientation_weight, 1e-8)
            cost = 0.5 * (err @ err)
            if verbose:
                print(f"[IK {arm} seed{seed_id}] iter {it}: pos={pos_err:.4e}, ori={ori_err:.4e}, lam={lam:.2e}")
            if pos_err < tol_pos and ori_err < tol_ori:
                break
            J = numerical_jacobian(q, fk, position_weight, orientation_weight)
            # LM: (J^T J + lambda I) dq = J^T err
            JTJ = J.T @ J
            g = J.T @ err
            # 为避免数值问题，对角线加微小正则
            H = JTJ + lam * np.eye(JTJ.shape[0])
            try:
                dq = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                dq = np.linalg.pinv(H) @ g
            # 步长限制，避免一次跳太大
            max_step = 0.15  # rad
            step_norm = np.linalg.norm(dq)
            if step_norm > max_step:
                dq = dq * (max_step / (step_norm + 1e-9))
            q_trial = clamp(q + dq)
            err_trial = pose_error(q_trial, fk, target_pos, tq, position_weight, orientation_weight)
            cost_trial = 0.5 * (err_trial @ err_trial)
            # 自适应阻尼: 成功下降则减小 lambda, 否则增大并重试
            if cost_trial < cost:
                q = q_trial
                prev_cost = cost_trial
                lam = max(lam * 0.7, 1e-5)
            else:
                lam = min(lam * 2.5, 1e2)
            # 若阻尼极大仍不下降，可提前终止本 seed
            if lam >= 1e2 and (prev_cost is not None) and cost_trial >= prev_cost:
                break
        # 记录最好解
        final_err = pose_error(q, fk, target_pos, tq, position_weight, orientation_weight)
        final_cost = 0.5 * (final_err @ final_err)
        if final_cost < best_err:
            best_err = final_cost
            best_q = q
        # 若已经很好就不再尝试更多种子
        if (
            np.linalg.norm(final_err[:3]) / max(position_weight,1e-8) < tol_pos and
            np.linalg.norm(final_err[3:]) / max(orientation_weight,1e-8) < tol_ori
        ):
            break

    return best_q if best_q is not None else q_current


if __name__ == "__main__":
    # 测试
    left_arm_qpos = [0, -0.96, 1.16, 0, -0.3, 0]
    right_arm_qpos = [0, -0.96, 1.16, 0, -0.3, 0]
    T_left = left_arm_fk(left_arm_qpos)
    T_right = right_arm_fk(right_arm_qpos)
    print("左臂末端位姿：\n", T_left)
    print("右臂末端位姿：\n", T_right)
    print("左臂末端欧拉角：\n", matrix_to_euler(T_left[:3,:3].flatten()))
    print("左臂末端欧拉角（四元数转）:\n", quat_to_euler([0.99875, 0, -0.04998, 0]))
    print("右臂末端欧拉角：\n", matrix_to_euler(T_right[:3,:3].flatten()))
    print("右臂末端欧拉角（四元数转）:\n", quat_to_euler([0, 0.04998, 0, 0.99875]))
    # 简单 IK 验证：使用 right_arm_fk 得到的末端位姿作为目标，尝试从不同初始值求解
    class DummyPhysics:
        # minimal object to satisfy get_qpos signature
        def __init__(self):
            # use START_ARM_POSE from constants to construct dummy qpos
            from constants import START_ARM_POSE
            self.data = type('d', (), {})()
            self.data.qpos = np.array(START_ARM_POSE)*0.9  # add some extra dims for grippers

    phys = DummyPhysics()
    arm = 'right'  # 'left' or 'right'
    if arm == 'right':
        target_T = T_right
    else:
        target_T = T_left
    target_pos = target_T[:3,3]
    target_quat = mat2quat(target_T[:3,:3])
    print(f'\nRunning IK to recover {arm} arm joint angles for target from FK...')
    sol = arm_ik(phys, target_pos, target_quat, arm=arm)
    print(f'IK solution ({arm} arm): {np.round(sol, 4)}')
    # forward check
    if arm == 'right':
        T_sol = right_arm_fk(sol)
    else:
        T_sol = left_arm_fk(sol)
    print(f'FK of IK solution pos: {T_sol[:3,3]}, target pos: {target_pos}')

    # 新的更严格测试：扰动多个初值并统计误差
    print("\n=== Enhanced IK convergence test ===")
    rng = np.random.default_rng(42)
    for test_arm, target_T in [('left', T_left), ('right', T_right)]:
        target_pos = target_T[:3,3]
        target_quat = mat2quat(target_T[:3,:3])
        base_full_qpos = get_qpos(phys)
        print(f"\nArm: {test_arm}")
        for i in range(5):
            # 构造扰动初值 (仅手臂)
            if test_arm == 'left':
                base_full_qpos[0:6] = np.array(left_arm_qpos) + rng.normal(0, 0.15, size=6)
            else:
                base_full_qpos[7:13] = np.array(right_arm_qpos) + rng.normal(0, 0.15, size=6)
            sol = arm_ik(phys, base_full_qpos, target_pos, target_quat, arm=test_arm, verbose=False)
            T_sol = left_arm_fk(sol) if test_arm=='left' else right_arm_fk(sol)
            perr = np.linalg.norm(T_sol[:3,3]-target_pos)
            # 方向误差角度
            def rot_angle(Ra,Rb):
                Rrel = Ra.T @ Rb
                ang = np.arccos(np.clip((np.trace(Rrel)-1)/2, -1, 1))
                return ang
            oerr = rot_angle(T_sol[:3,:3], target_T[:3,:3])
            print(f" seed {i}: pos_err={perr:.3e} m, ori_err={oerr:.3e} rad, q={np.round(sol,3)}")