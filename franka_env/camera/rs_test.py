import numpy as np
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API
import matplotlib.pyplot as plt
import time

# 获取设备信息
devices = rs.context().devices
serial_numbers = [d.get_info(rs.camera_info.serial_number) for d in devices]
print("检测到的相机序列号:", serial_numbers)

# 创建并启动管道
pipeline = rs.pipeline()
config = rs.config()

# 配置管道
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 启动管道
pipeline.start(config)

# 创建图形窗口
plt.ion()  # 打开交互模式
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

# 初始化图像显示
color_plot = ax1.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
depth_plot = ax2.imshow(np.zeros((480, 640), dtype=np.uint16), cmap='jet')
plt.colorbar(depth_plot, ax=ax2)

ax1.set_title('Color Image')
ax2.set_title('Depth Image')
ax1.axis('off')
ax2.axis('off')

try:
    while True:
        # 等待新帧
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            print("无法获取图像帧")
            continue

        # 转换为numpy数组
        img = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        # 更新图像显示
        color_plot.set_data(img)
        depth_plot.set_data(depth)
        
        # 更新颜色映射范围
        depth_plot.set_clim(vmin=depth.min(), vmax=depth.max())
        
        # 刷新显示
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        # 添加短暂延迟，避免占用过多CPU
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n程序被用户中断")
finally:
    # 停止管道
    pipeline.stop()
    plt.ioff()  # 关闭交互模式
    plt.close()  # 关闭图形窗口

