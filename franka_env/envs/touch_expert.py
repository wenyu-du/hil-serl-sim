#!/usr/bin/env python3
import multiprocessing
import numpy as np
import rclpy
from rclpy.node import Node
from omni_msgs.msg import OmniButtonEvent
from geometry_msgs.msg import PoseStamped
from typing import Tuple
from scipy.spatial.transform import Rotation

class GeomagicExpert:
    """
    This class provides an interface to the Geomagic Touch device.
    It continuously reads the device state from ROS2 topic and converts
    the absolute pose to incremental pose changes.
    """

    def __init__(self, position_scale=1, rotation_scale=1):

        # Manager to handle shared state between processes
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6  # [dx, dy, dz, droll, dpitch, dyaw]
        self.latest_data["buttons"] = [0, 0]    # Geomagic Touch的两个按钮状态
        
        # 缩放因子
        self.position_scale = position_scale  # 位置增量缩放
        self.rotation_scale = rotation_scale  # 旋转增量缩放

        # 存储上一时刻的位姿，用于计算增量
        self.last_position = self.manager.list([0.0] * 3)
        self.last_quaternion = self.manager.list([0.0] * 4)
        self.is_first_read = self.manager.Value('b', True)
        self.last_timestamp = self.manager.Value('d', 0.0)  # 添加时间戳存储
        self.target_dt = 0.05  # 目标时间差50ms

        # Start a process to continuously read the ROS2 messages
        self.process = multiprocessing.Process(target=self._read_geomagic_state)
        print(f"Process: {self.process}")
        self.process.daemon = True
        self.process.start()

    def _read_geomagic_state(self):
        """ROS2 subscriber process"""
        rclpy.init()
        node = GeomagicSubscriber(
            self.latest_data, 
            self.last_position, 
            self.last_quaternion,
            self.is_first_read,
            self.position_scale,
            self.rotation_scale
        )

        rclpy.spin(node)


    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest incremental action and button state."""
        action = self.latest_data["action"]
        buttons = self.latest_data["buttons"]
        return np.array(action), buttons
    
    def close(self):
        """Cleanup resources"""
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()  # 等待进程完全终止
            self.manager.shutdown()  # 关闭共享内存管理器


class GeomagicSubscriber(Node):
    """ROS2 Node for subscribing to Geomagic Touch state"""
    
    def __init__(self, shared_data, last_position, last_quaternion, 
                 is_first_read, position_scale, rotation_scale):
        super().__init__('geomagic_subscriber')
        self.shared_data = shared_data
        self.last_position = last_position
        self.last_quaternion = last_quaternion
        self.is_first_read = is_first_read
        self.position_scale = position_scale
        self.rotation_scale = rotation_scale
        self.last_timestamp = 0.0  # 添加时间戳变量
        
        # 创建位姿订阅者
        self.pose_subscription = self.create_subscription(
            PoseStamped,
            '/phantom/pose',
            self.pose_callback,
            10)
        
        # 创建按钮订阅者
        self.button_subscription = self.create_subscription(
            OmniButtonEvent,
            '/phantom/button',
            self.button_callback,
            10)
        
        self.get_logger().info('Geomagic Touch subscriber initialized')

    def compute_pose_delta(self, 
                         current_position: np.ndarray, 
                         current_quaternion: np.ndarray,
                         current_timestamp: float) -> np.ndarray:
        """
        计算位姿增量，基于时间差
        current_position: [x, y, z]
        current_quaternion: [x, y, z, w]
        current_timestamp: 当前时间戳
        returns: [dx, dy, dz, droll, dpitch, dyaw]
        """
        # 如果是第一次读取，初始化last值并返回零增量
        if self.is_first_read.value:
            self.last_position[:] = current_position
            self.last_quaternion[:] = current_quaternion
            self.last_timestamp = current_timestamp
            self.is_first_read.value = False
            return np.zeros(6)

        # 计算时间差
        dt = current_timestamp - self.last_timestamp
        if dt <= 0:
            return np.zeros(6)

        # 计算位置增量，考虑时间差
        position_delta = (current_position - np.array(list(self.last_position))) * self.position_scale * (0.05 / dt)

        # 计算姿态增量，考虑时间差
        last_rot = Rotation.from_quat(self.last_quaternion)
        current_rot = Rotation.from_quat(current_quaternion)
        relative_rot = current_rot * last_rot.inv()
        euler_delta = relative_rot.as_euler('xyz') * self.rotation_scale * (0.05 / dt)

        # 更新last值
        self.last_position[:] = current_position
        self.last_quaternion[:] = current_quaternion
        self.last_timestamp = current_timestamp

        return np.concatenate([position_delta, euler_delta])

    def button_callback(self, msg: OmniButtonEvent):
        """处理接收到的按钮事件消息"""
        # 更新按钮状态
        # self.get_logger().info(f'Button state updated: {msg.grey_button}, {msg.white_button}')
        self.shared_data["buttons"] = [msg.grey_button, msg.white_button]

    def pose_callback(self, msg: PoseStamped):
        """处理接收到的Geomagic Touch位姿消息"""
        current_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        
        current_quaternion = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])

        # 获取当前时间戳
        current_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # 计算位姿增量
        pose_delta = self.compute_pose_delta(current_position, current_quaternion, current_timestamp)

        # 更新共享数据中的位姿增量
        self.shared_data["action"] = pose_delta.tolist()


def test_geomagic_expert():
    """Test function for GeomagicExpert"""
    expert = GeomagicExpert()
    try:
        while True:
            action, buttons = expert.get_action()
            print(f"Delta Pose: {action}")
            print(f"Buttons: {buttons}")
    except KeyboardInterrupt:
        expert.close()


# if __name__ == "__main__":
#     test_geomagic_expert()