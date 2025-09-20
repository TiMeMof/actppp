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

def get_qpos(physics):
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

def arm_ik(physics:mujoco.Physics, qpos_init, target_pos, target_quat, arm='right'):
    """
    逆运动学求解
    输入：physics, 目标末端位置 target_pos (3,), 目标末端姿态 target_quat (4,), arm ('left' or 'right')
    输出：返回求解的左右臂关节角
    """
    assert len(target_pos) == 3 and len(target_quat) == 4
    # 获取初始关节角
    qpos_init = get_qpos(physics)
    # 设置末端目标位置和姿态
    # 说明：为避免依赖外部模型加载（pinocchio 可能不可用或未配置），
    # 这里使用数值雅可比 + 阻尼最小二乘（DLS）对手臂进行迭代求解。
    # qpos_init 格式: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
    # 本函数将默认求解右臂（如果目标在右侧）或左臂，依据 target_pos 的 x 值简单判断；
    # 返回值为长度6的关节角数组（单臂）。

    def fk_arm(arm, q):
        # 返回 4x4 末端变换矩阵
        if arm == 'left':
            return left_arm_fk(q)
        return right_arm_fk(q)

    def pose_error(current_T, target_p, target_q):
        # 位置误差
        p_cur = current_T[:3,3]
        dp = target_p - p_cur
        # 方向误差：使用四元数差转向量（小角近似）
        # 将 current_T 的旋转转为 quaternion
        R = current_T[:3,:3]
        # R to quaternion (w,x,y,z)
        q_cur = mat2quat(R)
        # ensure normalized
        q_cur = q_cur / np.linalg.norm(q_cur)
        tq = np.asarray(target_q, dtype=float)
        tq = tq / np.linalg.norm(tq)
        # quaternion error vector (using small-angle approx):  e =  q_target * q_cur^{-1}
        # q_inv of cur is [w, -x, -y, -z]
        qc_inv = np.array([q_cur[0], -q_cur[1], -q_cur[2], -q_cur[3]])
        # quaternion multiplication: q_err = tq * qc_inv
        w1,x1,y1,z1 = tq
        w2,x2,y2,z2 = qc_inv
        q_err = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
        # small angle approx: rotation vector ~ 2 * imag(q_err)
        dre = 2.0 * q_err[1:]
        # compose 6D error
        err6 = np.concatenate([dp, dre])
        return err6

    def numeric_jacobian(arm, q, eps=1e-6):
        """
        计算出关节空间到任务空间的数值雅可比矩阵
        J: 6 x n"""
        nq = len(q)
        J = np.zeros((6, nq))
        T0 = fk_arm(arm, q)
        for i in range(nq):
            dq = np.zeros_like(q)
            dq[i] = eps
            T1 = fk_arm(arm, q + dq)
            # position difference
            dp = (T1[:3,3] - T0[:3,3]) / eps
            # rotation difference -> using quaternion small-angle
            def Rmat_to_rotvec(Ra, Rb):
                # compute relative rotation R_rel = Ra.T @ Rb
                Rrel = Ra.T @ Rb
                # angle-axis from rotation matrix
                angle = np.arccos(np.clip((np.trace(Rrel)-1)/2, -1, 1))
                if np.isclose(angle, 0):
                    return np.zeros(3)
                rx = (Rrel[2,1] - Rrel[1,2])/(2*np.sin(angle))
                ry = (Rrel[0,2] - Rrel[2,0])/(2*np.sin(angle))
                rz = (Rrel[1,0] - Rrel[0,1])/(2*np.sin(angle))
                return angle * np.array([rx, ry, rz]) / eps
            dre = Rmat_to_rotvec(T0[:3,:3], T1[:3,:3])
            J[:3,i] = dp
            J[3:,i] = dre
        return J

    if arm == 'right':
        q = qpos_init[7:13]  # right_arm in qpos_init ordering
    else:
        q = qpos_init[0:6]

    # q = np.array(q0, dtype=float)
    lam = 1e-2
    max_iters = 100
    tol_pos = 1e-4
    tol_ori = 1e-3
    for it in range(max_iters):
        print(f"IK iter {it}, current q: {q}")
        Tcur = fk_arm(arm, q)
        err6 = pose_error(Tcur, np.asarray(target_pos, dtype=float), np.asarray(target_quat, dtype=float))
        pos_err_norm = np.linalg.norm(err6[:3])
        ori_err_norm = np.linalg.norm(err6[3:])
        if pos_err_norm < tol_pos and ori_err_norm < tol_ori:
            break
        J = numeric_jacobian(arm, q)
        # Damped least squares: dq = J.T * (J J.T + lambda^2 I)^-1 * err
        JJt = J @ J.T
        reg = lam * lam * np.eye(JJt.shape[0])
        try:
            inv = np.linalg.inv(JJt + reg)
            dq = J.T @ (inv @ err6)
        except np.linalg.LinAlgError:
            dq = J.T @ np.linalg.pinv(JJt + reg) @ err6
        # step size control
        alpha = 0.5
        q = q + alpha * dq

    return q

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