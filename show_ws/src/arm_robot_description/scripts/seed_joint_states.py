"""
# !/home/ubuntu/miniconda3/envs/py38/bin/python
#!/usr/bin/env python3
"""
import yaml
import os
import numpy as np
# pip install pin==2.6.21
# import pinocchio
try:
    import pinocchio as pin
    from pinocchio.robot_wrapper import RobotWrapper
    PINOCCHIO_AVAILABLE = True
except ImportError:
    # Fallback: 简化 IK 求解，不使用 pinocchio
    PINOCCHIO_AVAILABLE = False
    RobotWrapper = None

# This script publishes a single JointState message with initial positions
# read from the same YAML used for joint_state_publisher parameters.
# It then optionally keeps alive (latching behavior simulated) or exits.

# ---------------- Robot Controller Class -----------------

class RobotController:
    """封装单臂robot相关的所有操作"""
    
    def __init__(self, urdf_path=None, left_frame='left_gripper_link', right_frame='right_gripper_link'):
        self.urdf_path = urdf_path
        self.left_frame = left_frame
        self.right_frame = right_frame
        self.robot = None
        self.model = None
        self.data = None
        self.target_frame_id = None
        self.enabled = False
        self.is_left_arm = True  # 默认为左臂
        
        # 根据URDF路径判断是左臂还是右臂
        if urdf_path:
            self.is_left_arm = 'left' in os.path.basename(urdf_path)
            self.load_robot()
    
    def load_robot(self, urdf_path=None, root_joint=None):
        """加载robot模型（安全版本）"""
        if urdf_path:
            self.urdf_path = urdf_path
            self.is_left_arm = 'left' in os.path.basename(urdf_path)
            
        if not self.urdf_path:
            print("No URDF path provided")
            return False
            
        if not os.path.isfile(self.urdf_path):
            print(f"URDF file not found: {self.urdf_path}")
            return False
            
        if not PINOCCHIO_AVAILABLE:
            print("Pinocchio not available, robot operations disabled")
            return False
            
        try:
            arm_type = "Left" if self.is_left_arm else "Right"
            print(f"Loading {arm_type} arm URDF: {os.path.basename(self.urdf_path)}")
            
            # search paths: use directory containing urdf
            package_dirs = [os.path.dirname(self.urdf_path)]
            
            # 安全加载robot（只加载模型，避免几何体问题）
            self.model = pin.buildModelFromUrdf(self.urdf_path)
            if self.model is None:
                raise RuntimeError("Failed to build robot model from URDF")
                
            self.robot = RobotWrapper(self.model)
                
            self.data = self.model.createData()
            if self.data is None:
                raise RuntimeError("Failed to create model data")
            
            # 根据是左臂还是右臂选择对应的frame
            # 在单臂URDF中，frame名称就是 left_gripper_link 或 right_gripper_link
            target_frame = self.left_frame if arm_type == "Left" else self.right_frame
            
            # 安全获取frame ID
            try:
                self.target_frame_id = self.model.getFrameId(target_frame)
                
                # 验证frame ID
                if self.target_frame_id < 0:
                    raise RuntimeError(f"Invalid frame ID for {target_frame}: {self.target_frame_id}")
                    
            except Exception as frame_error:
                print(f"Frame ID error for {target_frame}: {frame_error}")
                return False
            
            # 验证模型基本信息
            print(f"Single arm model loaded: nq={self.model.nq}, nv={self.model.nv}")
            print(f"Target frame: {target_frame}, ID: {self.target_frame_id}")
            print(f"Arm type: {'Left' if self.is_left_arm else 'Right'}")
            
            self.enabled = True
            print(f"Successfully loaded URDF: {self.urdf_path}")
            return True
            
        except Exception as e:
            print(f"Failed loading URDF: {e}")
            # 清理部分加载的状态
            self.robot = None
            self.model = None
            self.data = None
            self.target_frame_id = None
            self.enabled = False
            return False
    
    def is_enabled(self):
        """检查robot是否可用"""
        return self.enabled and self.robot is not None

    def get_end_effector_positions(self, joint_positions):
        """获取末端执行器位置（单臂版本）"""
        if not self.is_enabled():
            return None
            
        try:
            # 直接使用传入的关节角度，不再进行提取
            # 因为在main函数中已经从YAML中提取了对应臂的关节
            q = np.array(joint_positions, dtype=np.float64)
            
            # 验证输入
            if not np.all(np.isfinite(q)):
                print("Invalid joint positions detected")
                return None
                
            if len(q) != self.model.nq:
                print("Joint length mismatch: got %d, expected %d", len(q), self.model.nq)
                return None
            
            # 安全的前向运动学
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            
            # 安全检查frame访问
            if (self.target_frame_id < 0 or self.target_frame_id >= len(self.data.oMf)):
                print("Frame ID out of bounds")
                return None
            
            # 获取末端执行器位置
            end_effector_pos = np.array(self.data.oMf[self.target_frame_id].translation, dtype=np.float64)
            
            # 验证结果
            if not np.all(np.isfinite(end_effector_pos)):
                print("Invalid position results")
                return None
                
            return end_effector_pos
            
        except Exception as e:
            print("Safe forward kinematics error: %s", e)
            # 不禁用，只是返回None
            return None

    def get_end_effector_pose(self, joint_positions):
        """获取末端执行器位姿（位置+姿态，单臂版本）- 优化内存安全版本"""
        if not self.is_enabled():
            return None, None
            
        try:
            # 直接使用传入的关节角度
            q = np.array(joint_positions, dtype=np.float64)
            
            # 验证输入
            if not np.all(np.isfinite(q)):
                print("Invalid joint positions detected")
                return None, None
                
            if len(q) != self.model.nq:
                print("Joint length mismatch: got %d, expected %d", len(q), self.model.nq)
                return None, None
            
            # 安全的前向运动学
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            
            # 安全检查frame访问
            if (self.target_frame_id < 0 or self.target_frame_id >= len(self.data.oMf)):
                print("Frame ID out of bounds")
                return None, None
            
            # 获取末端执行器的SE3变换
            end_effector_placement = self.data.oMf[self.target_frame_id]
            
            # 提取位置
            position = np.array(end_effector_placement.translation, dtype=np.float64)
            
            # 提取姿态 - 使用更安全的方法，避免重复的四元数对象创建
            rotation_matrix = end_effector_placement.rotation
            
            # 手动计算四元数，避免Pinocchio内部对象创建问题
            # 使用旋转矩阵到四元数的标准转换算法
            R = rotation_matrix
            trace = np.trace(R)
            
            if trace > 0:
                s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
                w = 0.25 * s
                x = (R[2, 1] - R[1, 2]) / s
                y = (R[0, 2] - R[2, 0]) / s  
                z = (R[1, 0] - R[0, 1]) / s
            elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
            
            # 构造四元数 [w, x, y, z]
            orientation = np.array([w, x, y, z], dtype=np.float64)
            
            # 归一化四元数
            norm = np.linalg.norm(orientation)
            if norm > 1e-6:
                orientation = orientation / norm
            else:
                print("Quaternion normalization failed - using identity")
                orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            
            # 验证结果
            if not (np.all(np.isfinite(position)) and np.all(np.isfinite(orientation))):
                print("Invalid pose results")
                return None, None
                
            return position, orientation
            
        except Exception as e:
            print("Safe forward kinematics pose error: %s", e)
            # 不禁用，只是返回None
            return None, None
    
    def disable(self):
        """禁用robot"""
        self.enabled = False
    
    def get_model_info(self):
        """获取模型信息"""
        if not self.is_enabled():
            return None
        return {
            'nq': self.model.nq,
            'nv': self.model.nv,
            'name': self.model.name,
            'target_frame': self.left_frame if self.is_left_arm else self.right_frame,
            'target_frame_id': self.target_frame_id,
            'arm_type': 'left' if self.is_left_arm else 'right'
        }
    
    def validate_joint_positions(self, joint_positions):
        """验证关节位置是否符合模型要求"""
        if not self.is_enabled():
            return False
        # 现在只支持8个关节（单臂），因为已经在main函数中提取了对应臂的关节
        return len(joint_positions) == self.model.nq
    
    def solve_ik(self, target_position, target_orientation=None, initial_joints=None, max_iters=1000, tolerance=1e-4, DT=1e-1):
        """
        单臂逆运动学求解 - 基于Pinocchio的迭代算法
        
        Args:
            target_position: 目标末端执行器位置 [x, y, z]
            target_orientation: 目标末端执行器四元数 [w, x, y, z]，可选
            initial_joints: 初始关节角度，如果None则使用当前配置
            max_iters: 最大迭代次数
            tolerance: 收敛容差
            DT: 积分步长
            
        Returns:
            success: bool, 是否成功求解
            joint_positions: np.array, 求解得到的关节角度
            error: float, 最终误差
            iterations: int, 实际迭代次数
        """
        if not self.is_enabled():
            print("Robot controller not enabled for IK solving")
            return False, None, float('inf'), 0
            
        try:
            # 验证输入
            target_pos = np.array(target_position, dtype=np.float64)
            if len(target_pos) != 3:
                print("Target position must be 3D [x, y, z]")
                return False, None, float('inf'), 0
            
            # 构造目标SE3变换
            if target_orientation is not None:
                # 从四元数转换为旋转矩阵
                target_quat = np.array(target_orientation, dtype=np.float64)  # [w, x, y, z]
                if len(target_quat) != 4:
                    print("Target orientation must be quaternion [w, x, y, z]")
                    return False, None, float('inf'), 0
                
                # 归一化四元数
                target_quat = target_quat / np.linalg.norm(target_quat)
                
                # 四元数转旋转矩阵 [w, x, y, z] -> R
                w, x, y, z = target_quat
                R_target = np.array([
                    [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                    [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                    [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
                ])
            else:
                # 只考虑位置，使用单位旋转矩阵
                R_target = np.eye(3)
            
            # 构造目标SE3变换
            oMdes = pin.SE3(R_target, target_pos)
            
            # 初始关节配置
            if initial_joints is not None:
                q = np.array(initial_joints, dtype=np.float64)
                if len(q) != self.model.nq:
                    print("Initial joints length mismatch: got {}, expected {}".format(len(q), self.model.nq))
                    return False, None, float('inf'), 0
            else:
                # 使用neutral配置作为初始值
                q = pin.neutral(self.model)
            
            # IK求解参数
            eps = tolerance
            damp = 1e-12  # 阻尼因子，防止雅可比矩阵奇异
            
            i = 0
            success = False
            err_norm = float('inf')
            
            print(f"Starting IK solve: target_pos=[{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            
            while i < max_iters:
                # 前向运动学
                pin.forwardKinematics(self.model, self.data, q)
                pin.updateFramePlacements(self.model, self.data)
                
                # 计算当前末端执行器到目标的变换误差
                current_transform = self.data.oMf[self.target_frame_id]
                iMd = current_transform.actInv(oMdes)
                err = pin.log(iMd).vector  # 6D误差向量 (旋转+平移)
                
                err_norm = np.linalg.norm(err)
                
                # 检查收敛
                if err_norm < eps:
                    success = True
                    print(f"IK converged in {i} iterations, final error: {err_norm:.6f}")
                    break
                
                # 计算雅可比矩阵（在当前frame坐标系下）
                J = pin.computeFrameJacobian(self.model, self.data, q, self.target_frame_id, pin.LOCAL)
                
                # 应用对数映射的雅可比矩阵修正
                J = -np.dot(pin.Jlog6(iMd.inverse()), J)
                
                # 阻尼最小二乘法求解关节速度
                # v = -J^T * (J*J^T + λI)^(-1) * err
                JJT_damped = J.dot(J.T) + damp * np.eye(6)
                try:
                    v = -J.T.dot(np.linalg.solve(JJT_damped, err))
                except np.linalg.LinAlgError:
                    print(f"Singular jacobian at iteration {i}, increasing damping")
                    damp *= 10
                    if damp > 1e-6:
                        print("Damping too large, IK failed")
                        break
                    continue
                
                # 积分更新关节角度
                q = pin.integrate(self.model, q, v * DT)
                
                # 调试输出（每10次迭代）
                if not i % 10:
                    print(f"IK iteration {i}: error = {err_norm:.6f}")
                
                i += 1
            
            # 验证最终结果
            if success:
                # 验证关节限制（可选）
                if hasattr(self.model, 'lowerPositionLimit') and hasattr(self.model, 'upperPositionLimit'):
                    for j in range(len(q)):
                        if (q[j] < self.model.lowerPositionLimit[j]):
                            q[j] = self.model.lowerPositionLimit[j]
                            print(f"Joint {j} exceeds limits: {q[j]:.3f} not in [{self.model.lowerPositionLimit[j]:.3f}, {self.model.upperPositionLimit[j]:.3f}]")
                        elif q[j] > self.model.upperPositionLimit[j]:
                            q[j] = self.model.upperPositionLimit[j]
                            print(f"Joint {j} exceeds limits: {q[j]:.3f} not in [{self.model.lowerPositionLimit[j]:.3f}, {self.model.upperPositionLimit[j]:.3f}]")

                
                # 验证最终位置误差
                pin.forwardKinematics(self.model, self.data, q)
                pin.updateFramePlacements(self.model, self.data)
                final_pos = self.data.oMf[self.target_frame_id].translation
                pos_error = np.linalg.norm(final_pos - target_pos)
                print(f"IK final position error: {pos_error:.6f} m")
                
            else:
                print(f"IK failed to converge after {max_iters} iterations, final error: {err_norm:.6f}")
            
            return success, q, err_norm, i
            
        except Exception as e:
            print(f"IK solving error: {e}")
            return False, None, float('inf'), 0



# ---------------- Utility Functions -----------------

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# --------------- Main ---------------
ee = [-0.3171888114641615, 0.5, 0.29525083628615956, 0.9987502603949664, 0.0, -0.04997916927067836, 0.0]
joints = [0,-0.96, 1.16, 0, -0.3, 0]
def main():
    import rospy
    from sensor_msgs.msg import JointState
    rospy.init_node('seed_joint_states')
    yaml_path = rospy.get_param('~yaml_path', '')
    urdf_path = rospy.get_param('~urdf_path', '')
    # yaml_path = '/home/ubuntu/actppp/assets/show_ws/src/arm_robot_description/config/initial_pose.yaml'
    # urdf_path = '/home/ubuntu/actppp/assets/show_ws/src/arm_robot_description/urdf/vx300s_bimanual copy.urdf'

    if not yaml_path:
        print('No ~yaml_path param provided to seed_joint_states (e.g. yaml_path:=...)')
        return
    if not os.path.isfile(yaml_path):
        print('YAML file not found: %s', yaml_path)
        return
    data = load_yaml(yaml_path) or {}

    # Preserve YAML order (PyYAML >=5 keeps insertion order); optionally sort if param set
    sort_joints = rospy.get_param('~sort_joints', False)
    items = list(data.items())
    if sort_joints:
        items.sort(key=lambda kv: kv[0])

    # 根据URDF路径判断是左臂还是右臂
    is_left_arm = 'left' in os.path.basename(urdf_path) if urdf_path else True
    arm_prefix = 'vx300s_left_' if is_left_arm else 'vx300s_right_'
    
    joint_names = []
    positions = []
    for k, v in items:
        if isinstance(v, (int, float)):
            # 只选择对应臂的关节
            if k.startswith(arm_prefix):
                joint_names.append(k)
                positions.append(float(v))

    print(f"Selected {len(joint_names)} joints for {'left' if is_left_arm else 'right'} arm")
    print(f"Joint names: {joint_names}")
    if not joint_names:
        print('No joint numeric entries found in YAML: %s', yaml_path)
    pub = rospy.Publisher('/joint_states', JointState, queue_size=1, latch=True)
    rospy.sleep(0.2)  # small delay to allow publisher registration before publish

    # Load robot model for IK if URDF provided (安全初始化)
    robot_controller = RobotController(urdf_path)

    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.name = joint_names
    msg.position = positions
    msg.velocity = []
    msg.effort = []

    # Publish initial
    print('Publishing initial JointState with %d joints from %s', len(joint_names), yaml_path)
    pub.publish(msg)
    
    # 显示测试信息
    if robot_controller and robot_controller.is_enabled():
        arm_type = "Left" if robot_controller.is_left_arm else "Right"
        print(f"Single arm robot controller loaded successfully ({arm_type} arm)")
        print("Will test:")
        print("  - Forward kinematics every 20 iterations")
        print("  - IK testing skipped (not implemented yet)")
        model_info = robot_controller.get_model_info()
        if model_info:
            print(f"Model info: nq={model_info['nq']}, target_frame={model_info['target_frame']}")
    else:
        print("Robot controller not available - only publishing joint states")

    hz = 20
    rate = rospy.Rate(hz)  # 20 Hz

    for i in range(1000):
        msg.header.stamp = rospy.Time.now()       

        if robot_controller is not None and robot_controller.is_enabled():
            if i % 2 == 0:  # 每20次迭代测试一次
                # 首先获取当前末端位姿
                end_effector_pose = robot_controller.get_end_effector_pose(msg.position)
                
                if end_effector_pose[0] is not None and end_effector_pose[1] is not None:
                    current_position, current_orientation = end_effector_pose
                    arm_type = "Left" if robot_controller.is_left_arm else "Right"
                    
                    print("Iteration %d Current FK: %s arm EE pos [%.3f %.3f %.3f] quat [%.3f %.3f %.3f %.3f]", 
                                i, arm_type, current_position[0], current_position[1], current_position[2],
                                current_orientation[0], current_orientation[1], current_orientation[2], current_orientation[3])
                    
                    # 设置目标位置
                    target_position = current_position.copy()
                    target_position[1] += 0.001
                    current_orientation[3] += 0.001
                    
                    print("Target position: [%.3f %.3f %.3f]", 
                                target_position[0], target_position[1], target_position[2])
                    
                    # 使用IK求解到达目标位置所需的关节角度
                    success, new_joints, error, iterations = robot_controller.solve_ik(
                        target_position=target_position,
                        target_orientation=current_orientation,  # 保持相同姿态
                        initial_joints=msg.position,
                        max_iters=100,  # 减少迭代次数以加快测试
                        tolerance=1e-3,  # 稍微放宽容差
                        DT=0.05  # 较小的步长
                    )
                    
                    if success:
                        print("IK Success! Updating joint positions. Error: %.6f, Iterations: %d", error, iterations)
                        # 更新关节位置
                        msg.position = list(new_joints)
                        
                        # 验证新的FK结果
                        new_pose = robot_controller.get_end_effector_pose(new_joints)
                        if new_pose[0] is not None:
                            new_pos = new_pose[0]
                            achieved_error = np.linalg.norm(new_pos - target_position)
                            print("Verification: achieved pos [%.3f %.3f %.3f], error: %.6f", 
                                        new_pos[0], new_pos[1], new_pos[2], achieved_error)
                    else:
                        print("IK Failed! Error: %.6f, Iterations: %d", error, iterations)
                              


        pub.publish(msg)
        rate.sleep()
        if rospy.is_shutdown():
            return


    hold_time = rospy.get_param('~hold_seconds', 1.0)
    print('Holding node alive for %.2f seconds...', hold_time)
    rospy.sleep(hold_time)
    print('Seed joint_states node exiting.')

if __name__ == '__main__':
    main()
