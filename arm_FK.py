import numpy as np        
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
    将四元数(w, x, y, z)转换为与 matrix_to_euler 一致的 ZYX 欧拉角 (返回顺序: roll, pitch, yaw) ，单位：度
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
    print("右臂末端欧拉角：\n", matrix_to_euler(T_right[:3,:3].flatten()))
    print("左臂末端欧拉角（四元数转）:\n", quat_to_euler([0.99875, 0, -0.04998, 0]))
    print("右臂末端欧拉角（四元数转）:\n", quat_to_euler([0, 0.04998, 0, 0.99875]))