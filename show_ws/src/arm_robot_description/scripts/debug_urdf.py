#!/home/ubuntu/miniconda3/envs/py38/bin/python

import os
import sys
try:
    import pinocchio as pin
    from pinocchio.robot_wrapper import RobotWrapper
    PINOCCHIO_AVAILABLE = True
except ImportError:
    PINOCCHIO_AVAILABLE = False
    print("Pinocchio not available!")
    sys.exit(1)

def debug_urdf(urdf_path):
    print(f"\n=== Debugging URDF: {urdf_path} ===")
    
    if not os.path.isfile(urdf_path):
        print(f"File not found: {urdf_path}")
        return
    
    try:
        package_dirs = [os.path.dirname(urdf_path)]
        # 只加载模型，不加载几何体
        model = pin.buildModelFromUrdf(urdf_path)
        robot = RobotWrapper(model)
        
        if robot is None:
            print("Failed to build robot")
            return
            
        model = robot.model
        print(f"Model name: {model.name}")
        print(f"nq (position DOF): {model.nq}")
        print(f"nv (velocity DOF): {model.nv}")
        print(f"Total joints: {len(model.names)}")
        
        print("\nJoint names and types:")
        for i in range(len(model.names)):
            if i > 0:  # skip universe joint
                joint_name = model.names[i]
                joint_type = model.joints[i].shortname()
                print(f"  {i}: {joint_name} ({joint_type})")
        
        print("\nFrame names:")
        for i in range(model.nframes):
            frame_name = model.frames[i].name
            print(f"  {i}: {frame_name}")
            
        # 查找gripper frame
        try:
            left_id = model.getFrameId('left_gripper_link')
            print(f"\nleft_gripper_link frame ID: {left_id}")
        except:
            print(f"\nleft_gripper_link frame not found")
            
        try:
            right_id = model.getFrameId('right_gripper_link')  
            print(f"right_gripper_link frame ID: {right_id}")
        except:
            print(f"right_gripper_link frame not found")
            
    except Exception as e:
        print(f"Error loading URDF: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    urdf_dir = '/home/ubuntu/actppp/assets/show_ws/src/arm_robot_description/urdf'
    
    debug_urdf(os.path.join(urdf_dir, 'vx300s_left.urdf'))
    debug_urdf(os.path.join(urdf_dir, 'vx300s_right.urdf'))
    
    # 也测试双臂URDF
    debug_urdf(os.path.join(urdf_dir, 'vx300s_bimanual.urdf'))