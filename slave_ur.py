"""
Slave UR Script using ZeroMQ to follow Master Sigma's pose and gripper state.
Date: 2025-12-30
"""
import zmq
import numpy as np
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

import utilize.robotiq_gripper as robotiq_gripper
from utilize.const import ROBOT_IP, PORT

def main():
    # 1. 初始化 ZeroMQ 订阅者
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5555")
    # 设置订阅过滤（空字符串表示接收所有消息）
    socket.setsockopt_string(zmq.SUBSCRIBE, "") 
    # 配置丢弃旧消息，仅保留最新位姿，减少堆积延迟
    socket.setsockopt(zmq.CONFLATE, 1)

    gripper = robotiq_gripper.RobotiqGripper()
    gripper.connect(ROBOT_IP, PORT)
    gripper.activate()

    rtde_r = RTDEReceiveInterface(ROBOT_IP)
    rtde_c = RTDEControlInterface(ROBOT_IP)
    
    print("从端已启动，正在跟随主端...")

    try:
        while True:
            # 非阻塞或阻塞接收最新位姿
            data = socket.recv_pyobj()
            pos = data.get("pos", np.zeros(3))
            orn = data.get("orientation", np.zeros((3, 3)))
            gripper = data.get("gripper", 0.0)
            timestamp = data.get("timestamp", 0.0)

            
            # 3. 坐标映射与转换 (关键步骤)
            # 你需要根据实际安装位置，将 Sigma 的坐标系映射到 UR 的坐标系
            # target_pose = transform(master_pose)
            
            # 4. 驱动机器人运动
            # servoL 适合高频流式控制。参数：位姿, 速度, 加速度, 时间, 采样平滑
            # rtde_c.servoL(master_pose, 0.5, 0.5, 0.002, 0.1, 300)
            gripper.move_and_wait_for_pos(, 255, 255)

    except KeyboardInterrupt:
        rtde_c.servoStop()
        rtde_c.stopScript()

    finally:
        print("断开连接...")
        gripper.disconnect()
        rtde_r.disconnect()
        rtde_c.disconnect()

if __name__ == "__main__":
    main()