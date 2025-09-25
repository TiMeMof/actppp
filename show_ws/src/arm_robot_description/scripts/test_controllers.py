#!/home/ubuntu/miniconda3/envs/py38/bin/python

# 简单测试脚本
import sys
import os
sys.path.append('/home/ubuntu/actppp/assets/show_ws/src/arm_robot_description/scripts')

import rospy
from seed_joint_states import RobotController, load_yaml

def test_robot_controller():
    # 模拟ROS节点初始化
    rospy.init_node('test_robot_controller', anonymous=True)
    
    # 测试左臂
    print("=== Testing Left Arm ===")
    urdf_path_left = '/home/ubuntu/actppp/assets/show_ws/src/arm_robot_description/urdf/vx300s_left.urdf'
    controller_left = RobotController(urdf_path_left)
    
    if controller_left.is_enabled():
        info = controller_left.get_model_info()
        print(f"Left arm model info: {info}")
        
        # 测试前向运动学
        test_joints = [0, -0.96, 1.16, 0, -0.3, 0, 0.024, -0.024]  # 8个关节
        pos = controller_left.get_end_effector_positions(test_joints)
        print(f"Left arm FK result: {pos}")
    else:
        print("Left arm controller failed to load")
    
    print("\n=== Testing Right Arm ===")
    urdf_path_right = '/home/ubuntu/actppp/assets/show_ws/src/arm_robot_description/urdf/vx300s_right.urdf'
    controller_right = RobotController(urdf_path_right)
    
    if controller_right.is_enabled():
        info = controller_right.get_model_info()
        print(f"Right arm model info: {info}")
        
        # 测试前向运动学
        test_joints = [0, -0.96, 1.16, 0, -0.3, 0, 0.024, -0.024]  # 8个关节
        pos = controller_right.get_end_effector_positions(test_joints)
        print(f"Right arm FK result: {pos}")
    else:
        print("Right arm controller failed to load")

if __name__ == '__main__':
    test_robot_controller()