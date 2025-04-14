
### 仿真环境touch操作流程

#### 打开touch的ros驱动
source /home/ae/dwy/Geomagic_Touch_ROS2/install/setup.bash
ros2 launch omni_common omni_state.launch.py

#### 进入项目运行test程序
source /home/ae/dwy/Geomagic_Touch_ROS2/install/setup.bash
conda activate hilserl_2
cd ~/dwy/hil-serl-sim/examples/experiments/pick_cube_sim
python ../../../env_test.py


#### 配置选择操控方式
/home/ae/dwy/hil-serl-sim/examples/experiments/pick_cube_sim/config.py
'''


### 下一步工作计划

#### 双臂操作
#### 调试优化touch操控
#### 实机测试