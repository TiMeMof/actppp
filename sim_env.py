import numpy as np
import os
import collections
import matplotlib.pyplot as plt
import mujoco as mj
from arm_FK import matrix_to_euler, quat_to_euler
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from constants import DT, XML_DIR, START_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

import IPython
e = IPython.embed

BOX_POSE = [None] # to be changed from outside

def make_sim_env(task_name):
    """
    Environment for simulated robot bi-manual manipulation, with joint position control
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    if 'sim_transfer_cube' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_insertion' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_insertion.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env


def ik_solve(physics:mujoco.Physics, site_name, target_pos, old_pos, qpos_init, max_iter=10, tol=1e-4, step_size=0.1):
    site_id = physics.model.name2id(site_name, "site")
    q = qpos_init.copy()
    for i in range(max_iter):
        # # 更新 physics data 使之反映当前 q
        # physics.data.qpos[8:14] = q   # 假设 right arm 在 qpos 的 8:14
        # mujoco.mj_forward(physics.model.ptr, physics.data.ptr)

        # current_pos = physics.data.site_xpos[site_id].copy()
        err = target_pos - old_pos
        print("itttttt",err)
        
        if np.linalg.norm(err) < tol:
            break
        
        # 雅可比矩阵
        jacp = np.zeros((3, physics.model.nv))
        mujoco.mj_jacSite(physics.model.ptr, physics.data.ptr, jacp, None, site_id)
        # jacp.shape=(3,22)

        if site_name == "right_ee_site":
            J = jacp[:, 8:14]
        elif site_name == "left_ee_site":
            J = jacp[:, :6]
        
        # 伪逆
        dq = step_size * np.linalg.pinv(J) @ err
        q = q + dq
    return q

class BimanualViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)
        # self.right_ee_pos_old=[0.]*3
        # self.right_ee_pos=[0.]*3
        # self.left_arm_qpos=[0.]*6
        # self.left_arm_qpos_old=[0.]*6
        # self.right_arm_qpos=[0.]*6
        # self.right_arm_qpos_old=[0.]*6
        # self.start = True

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        right_arm_action = action[7:7+6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7+6]

        left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_left_gripper_action)
        right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_right_gripper_action)

        full_left_gripper_action = [left_gripper_action, -left_gripper_action]
        full_right_gripper_action = [right_gripper_action, -right_gripper_action]

        env_action = np.concatenate([left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action])
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        # 对应6个机械连杆名称[waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        # 归一化夹爪位置
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        # from arm_FK import right_arm_fk, left_arm_fk
        # # 计算正向运动学，验证 qpos 的正确性
        # right_ee_pos_fk = right_arm_fk(right_arm_qpos)
        # left_ee_pos_fk = left_arm_fk(left_arm_qpos)
        # site_id = physics.model.name2id("right_ee_site", "site")
        # site_pos = physics.data.site_xpos[site_id].copy()
        # print("right fk err:", right_ee_pos_fk[:3,3] - site_pos)
        # site_id = physics.model.name2id("left_ee_site", "site")
        # site_pos = physics.data.site_xpos[site_id].copy()
        # print("left fk err:", left_ee_pos_fk[:3,3] - site_pos)
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])


    
    def get_ee_pose(self, physics):
        


        site_id = physics.model.name2id("left_ee_site", "site")
        left_ee_pos = physics.data.site_xpos[site_id].copy()  # 有效
        left_ee_mat = physics.data.site_xmat[site_id]
        left_euler = matrix_to_euler(left_ee_mat)
        left_quat=np.zeros(4)
        # w x y z
        mj.mju_mat2Quat(left_quat, left_ee_mat)

        # left_ee_mat2 = np.zeros(9)
        # np.reshape(left_ee_mat2, (3, 3))
        # mj.mju_quat2Mat(left_ee_mat2,left_quat)
        # left_euler2 = matrix_to_euler(left_ee_mat2)
        # print("errrrrrrrrrr:",np.asarray(left_euler2) - np.asarray(left_euler))


        # 得到现在的right ee位姿，以及旧的
        site_id = physics.model.name2id("right_ee_site", "site")
        right_ee_pos = physics.data.site_xpos[site_id].copy()           
        right_ee_mat = physics.data.site_xmat[site_id]
        right_euler = matrix_to_euler(right_ee_mat)
        right_quat=np.zeros(4)
        mj.mju_mat2Quat(right_quat, right_ee_mat)

        # # 得到现在的关节角度，以及旧的
        # qpos_raw = physics.data.qpos.copy()
        # left_qpos_raw = qpos_raw[:8]
        # right_qpos_raw = qpos_raw[8:16]
        # self.left_arm_qpos_old = self.left_arm_qpos.copy()
        # self.right_arm_qpos_old = self.right_arm_qpos.copy()
        # if self.start:
        #     self.left_arm_qpos_old = left_qpos_raw[:6]
        #     self.right_arm_qpos_old = right_qpos_raw[:6]
        #     self.start = False
        # self.left_arm_qpos = left_qpos_raw[:6]
        # self.right_arm_qpos = right_qpos_raw[:6]

        # print("迭代之前误差：",self.right_ee_pos - self.right_ee_pos_old)
        # print("迭代之前误差2：",self.right_arm_qpos - self.right_arm_qpos_old)
        # qpos_ik = ik_solve(physics, "right_ee_site", self.right_ee_pos, self.right_ee_pos_old, self.right_arm_qpos_old)
        # print("之后误差：",self.right_arm_qpos - qpos_ik)
        
        return left_ee_pos, left_euler, left_quat, right_ee_pos, right_euler, right_quat

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        left_ee_pos, left_ee_euler,left_ee_quat,right_ee_pos,right_ee_euler,right_ee_quat = self.get_ee_pose(physics)
        # 3 + 4 + 1 + 3 + 4 + 1 = 16
        # print(obs['qpos'][6], obs['qpos'][13])
        obs['ee'] = np.concatenate([left_ee_pos,left_ee_quat,[obs['qpos'][6]],right_ee_pos,right_ee_quat, [obs['qpos'][13]]])
        print(f"ee: {[f'{x:.4f}' for x in obs['ee']]}")
        # print(f"====={right_ee_euler}=====")
        # right_rot = quat_to_euler(right_ee_quat)
        # print("rot:", right_rot)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['left_wrist'] = physics.render(height=480, width=640, camera_id='left_wrist')
        obs['images']['right_wrist'] = physics.render(height=480, width=640, camera_id='right_wrist')
        # obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        # obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class TransferCubeTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_left_gripper: # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table: # successful transfer
            reward = 4
        return reward


class InsertionTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7*2:] = BOX_POSE[0] # two objects
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper = ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = ("socket-1", "table") in all_contact_pairs or \
                             ("socket-2", "table") in all_contact_pairs or \
                             ("socket-3", "table") in all_contact_pairs or \
                             ("socket-4", "table") in all_contact_pairs
        peg_touch_socket = ("red_peg", "socket-1") in all_contact_pairs or \
                           ("red_peg", "socket-2") in all_contact_pairs or \
                           ("red_peg", "socket-3") in all_contact_pairs or \
                           ("red_peg", "socket-4") in all_contact_pairs
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table): # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table): # peg and socket touching
            reward = 3
        if pin_touched: # successful insertion
            reward = 4
        return reward


def get_action(master_bot_left, master_bot_right):
    action = np.zeros(14)
    # arm action
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7:7+6] = master_bot_right.dxl.joint_states.position[:6]
    # gripper action
    left_gripper_pos = master_bot_left.dxl.joint_states.position[7]
    right_gripper_pos = master_bot_right.dxl.joint_states.position[7]
    normalized_left_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(left_gripper_pos)
    normalized_right_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(right_gripper_pos)
    action[6] = normalized_left_pos
    action[7+6] = normalized_right_pos
    return action

def test_sim_teleop():
    """ Testing teleoperation in sim with ALOHA. Requires hardware and ALOHA repo to work. """
    from interbotix_xs_modules.arm import InterbotixManipulatorXS

    BOX_POSE[0] = [0.2, 0.5, 0.05, 1, 0, 0, 0]

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_right', init_node=False)

    # setup the environment
    env = make_sim_env('sim_transfer_cube')
    ts = env.reset()
    episode = [ts]
    # setup plotting
    ax = plt.subplot()
    plt_img = ax.imshow(ts.observation['images']['angle'])
    plt.ion()

    for t in range(1000):
        action = get_action(master_bot_left, master_bot_right)
        ts = env.step(action)
        episode.append(ts)

        plt_img.set_data(ts.observation['images']['angle'])
        plt.pause(0.02)


if __name__ == '__main__':
    test_sim_teleop()

