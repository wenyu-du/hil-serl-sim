import os
import jax
import jax.numpy as jnp
import numpy as np
import franka_env.envs.touch_expert as touch_expert
import gymnasium as gym
from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from examples.experiments.config import DefaultTrainingConfig
from examples.experiments.ram_insertion.wrapper import RAMEnv
import gym
import matplotlib.pyplot as plt

class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "052622071155",
            "dim": (1280, 720),
            "exposure": 40000,
        },
        # "wrist_2": {
        #     "serial_number": "127122270350",
        #     "dim": (1280, 720),
        #     "exposure": 40000,
        # },
    }
    IMAGE_CROP = {
        "wrist_1": lambda img: img[150:450, 350:1100],
        "wrist_2": lambda img: img[100:500, 400:900],
    }
    TARGET_POSE = np.array([0.381241235410154,0.3578590131997776,0.57843494179085326, np.pi, 0, 0])
    GRASP_POSE = np.array([0.5857508505445138,-0.22036261105675414,0.2731021902359492, np.pi, 0, 0])
    RESET_POSE = TARGET_POSE + np.array([0, 0, 0.05, 0, 0.05, 0])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.2, 0.2, 0.2, 0.01, 0.1, 0.4])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.2, 0.2, 0.2, 0.01, 0.1, 0.4])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    # ACTION_SCALE = (0.01, 0.06, 1)
    ACTION_SCALE = (2, 1, 1)
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 100
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.0075,
        "translational_clip_y": 0.0016,
        "translational_clip_z": 0.0055,
        "translational_clip_neg_x": 0.002,
        "translational_clip_neg_y": 0.0016,
        "translational_clip_neg_z": 0.005,
        "rotational_clip_x": 0.01,
        "rotational_clip_y": 0.025,
        "rotational_clip_z": 0.005,
        "rotational_clip_neg_x": 0.01,
        "rotational_clip_neg_y": 0.025,
        "rotational_clip_neg_z": 0.005,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 250,
        "rotational_damping": 9,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.1,
        "translational_clip_y": 0.1,
        "translational_clip_z": 0.1,
        "translational_clip_neg_x": 0.1,
        "translational_clip_neg_y": 0.1,
        "translational_clip_neg_z": 0.1,
        "rotational_clip_x": 0.5,
        "rotational_clip_y": 0.5,
        "rotational_clip_z": 0.5,
        "rotational_clip_neg_x": 0.5,
        "rotational_clip_neg_y": 0.5,
        "rotational_clip_neg_z": 0.5,
        "rotational_Ki": 0.0,
    }


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist_1", "wrist_2"]
    classifier_keys = ["wrist_1", "wrist_2"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = RAMEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        # env = GripperCloseEnv(env)
        # if not fake_env:
        #     env = SpacemouseIntervention(env)
        env = TouchIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                # added check for z position to further robustify classifier, but should work without as well
                return int(sigmoid(classifier(obs)) > 0.85 and obs['state'][0, 6] > 0.04)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env



class TouchIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None, position_scale=50, rotation_scale=10):
        super().__init__(env)
        
        # 判断action维度来决定是否enable夹爪
        # self.gripper_enabled = True
        # if self.action_space.shape == (6,):
        #     self.gripper_enabled = False
            
        self.action_indices = action_indices
        self.intervened = False
        self.env.intervened = self.intervened
        
        # 初始化Geomagic Touch控制器
        self.touch_expert = touch_expert.GeomagicExpert(
            position_scale=position_scale,
            rotation_scale=rotation_scale
        )
        self.gripper_state = 'open'
        
    def visualize_expert(self, expert_a):
        plt.clf()  # 清除当前图形
        plt.bar(range(len(expert_a)), expert_a)  # 绘制条形图
        plt.ylim(-1.5, 1.5)  # 设置y轴范围
        plt.xlabel('Expert Action Index')
        plt.ylabel('Action Value')
        plt.title('Expert Actions Visualization')
        plt.pause(0.1)  # 暂停以更新图形

    def action(self, action: np.ndarray) -> np.ndarray:
        # 获取Touch设备的动作和按钮状态
        expert_a, buttons = self.touch_expert.get_action()
        
        
        # 处理夹爪控制
        # if self.gripper_enabled:
        #     # 使用灰色按钮(buttons[0])控制夹爪
        if buttons[0] == 1:
            if self.gripper_state == 'open':
                self.gripper_state = 'close'
            else:
                self.gripper_state = 'open'
                
        gripper_action = np.random.uniform(0.9, 1, size=(1,)) if self.gripper_state == 'close' else np.random.uniform(-1, -0.9, size=(1,))

        expert_a_inversed = np.concatenate((expert_a, gripper_action), axis=0)
        self.visualize_expert(expert_a_inversed)

        # 使用白色按钮(buttons[1])切换干预状态
        if buttons[1] == 1:
            # self.intervened = not self.intervened
            # self.env.intervened = self.intervened
            # print(f"Intervention toggled: {self.intervened}")
            print("Intervention toggled: True")

            # return expert_a, True
            
            print(expert_a_inversed, 'touch 介入')
            return expert_a_inversed, True
            
        # if self.action_indices is not None:
        #     filtered_expert_a = np.zeros_like(expert_a)
        #     filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
        #     expert_a = filtered_expert_a
            
        # if self.intervened:
        #     print(expert_a)
        #     return expert_a, True
            
        else:
            # print("Intervention toggled: False")
            return np.zeros(7), False
        
            
    def step(self, action):
        new_action, replaced = self.action(action)
        # print(new_action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        return obs, rew, done, truncated, info
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.gripper_state = 'open'
        return obs, info
        
    def close(self):
        self.touch_expert.close()
        return super().close()