# import debugpy
# debugpy.listen(10010)
# print('wait debugger')
# debugpy.wait_for_client()
# print("Debugger Attached")


from franka_sim.envs.panda_pick_gym_env import PandaPickCubeGymEnv
from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv
)
from pynput import keyboard
from examples.experiments.pick_cube_sim.config import TrainConfig
import os
import cv2
from datetime import datetime
import numpy as np

# env = PandaPickCubeGymEnv(render_mode="human")
# env = PandaPickCubeGymEnv(render_mode="human", image_obs=True, config=EnvConfig())
env = TrainConfig().get_environment()

# 创建保存图片的目录
save_dir = f"images_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(save_dir, exist_ok=True)

obs, _ = env.reset()
print("Observation space keys:", obs.keys())  # 打印观察空间的键
step = 0

while True:
    actions = env.action_space.sample()
    # actions = np.zeros(env.action_space.sample().shape) 
    next_obs, reward, done, truncated, info = env.step(actions)

    # 保存图片
    for cam_name in ['front', 'wrist', 'wrist_1', 'wrist_2']:
        if cam_name in next_obs:
            img = next_obs[cam_name]
            print(f"Image {cam_name} shape before: {img.shape}, dtype: {img.dtype}")
            
            # 调整维度顺序 (1, height, width, channels) -> (height, width, channels)
            if len(img.shape) == 4 and img.shape[0] == 1:
                img = img.squeeze(0)
            
            print(f"Image {cam_name} shape after: {img.shape}")
            
            try:
                img_path = os.path.join(save_dir, f"step_{step:04d}_{cam_name}.png")
                cv2.imwrite(img_path, img)
                print(f"Successfully saved {img_path}")
            except Exception as e:
                print(f"Error saving {cam_name} image: {str(e)}")

    step += 1

    if done:
        obs, info = env.reset()
