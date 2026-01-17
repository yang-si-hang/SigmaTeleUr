"""
测试 UR10e 机器人的 LeRobot 数据集录制流程

Date: 2026-01-16
"""
import shutil
from pathlib import Path
import time
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.utils.constants import OBS_STR, ACTION

from lerobot_robot_ur10e.ur10e import ACTION_NAMES, ACTION_NAMES, Ur10eRobot
from lerobot_robot_ur10e.config_ur10e import Ur10eRobotConfig

from const import ROBOT_IP, PORT, GEMINI_336_NUMBER
from lerobot_camera_orbbec import OrbbecCameraConfig, OrbbecCamera


output_dir = Path("outputs/test_ur10e_dataset")
if output_dir.exists():
    shutil.rmtree(output_dir) # 每次运行清空旧数据

def test_recording_flow():
    # 1. 初始化配置 (请确保 IP 和端口正确)
    config = Ur10eRobotConfig(
        robot_ip=ROBOT_IP, 
        gripper_port=PORT,
        cameras={
            "main": OrbbecCameraConfig(
                serial_number_or_name=GEMINI_336_NUMBER,
                color_mode="rgb",
                use_depth=False,
                width=640,
                height=480,
                fps=30
            )
        }
    )
    
    robot = Ur10eRobot(config)
    
    try:
        # 2. 连接测试
        print("--- 步骤 1: 连接机器人 ---")
        robot.connect()

        print("\n--- 步骤 1: 初始化 LeRobotDataset ---")
        dataset = LeRobotDataset.create(
            repo_id="test_user/ur10e_test",
            fps=30,
            root=output_dir,
            robot_type=robot.name,
            features=robot.observation_features | robot.action_features, # 合并观察和动作特征
            use_videos=True, # 启用视频编码（录制结束后会转成MP4）
        )

        print("--- 步骤 2: 启动图像后台写入线程 ---")
        dataset.start_image_writer(num_threads=4)
        
        # 3. 验证特征定义 (这决定了数据集的结构)
        print("\n--- 步骤 2: 验证特征定义 ---")
        obs_features = robot.observation_features
        act_features = robot.action_features
        print(f"观察特征键值: {list(obs_features.keys())}")
        print(f"动作特征键值: {list(act_features.keys())}")

        # 4. 模拟采集循环
        num_frames = 60
        print("\n--- 步骤 3: 模拟采集循环 (60帧) ---")
        for i in range(num_frames):
            start_t = time.perf_counter()
            
            # 读取观察值 (包含图像转换逻辑)
            obs = robot.get_observation()

            # 验证图像维度是否为 (C, H, W)
            for k, v in obs.items():
                if "images" in k:
                    print(f"帧 {i} | {k} 维度: {v.shape} (预期首位为 3 或 1)")
                    if v.shape[0] not in [1, 3]:
                        print(f"⚠️ 警告: {k} 可能不是 (C, H, W) 格式!")

            # 模拟生成一个动作 (这里直接用当前状态作为目标)
            current_state = obs["observation.state"]
            action_dict = {name: current_state[idx] for idx, name in enumerate(robot.action_features["action"]["names"])}
            
            # 模拟 LeRobot 内部的帧构建逻辑 (验证字段对齐)
            # try:
            #     # 这一步报错通常意味着 get_observation 返回的 key 与 observation_features 不符
            #     frame = build_dataset_frame(obs_features, obs, prefix=OBS_STR)
            #     print(f"帧 {i} | 成功构建数据集 Observation 帧")
            # except Exception as e:
            #     print(f"❌ 构建数据帧失败: {e}")

            # 执行动作测试
            robot.send_action(action_dict)
            
            # 这里的动作可以从手动输入、Teleop 或者直接读取当前状态模拟
            state = obs["observation.state"]
            # 构造 action 字典
            action_data = {name: state[idx] for idx, name in enumerate(ACTION_NAMES)}
            
            # 组合成一个完整的 Dataset Frame (LeRobot 格式要求)
            # 注意：LeRobotDataset.add_frame 接受扁平化的字典
            frame = {
                **obs,
                "action": state, # 或者是 action_data 的数组形式
                "task": "Test UR10e recording",
            }
            
            # 写入缓冲区
            dataset.add_frame(frame)
            
            if i % 10 == 0:
                print(f"进度: {i}/{num_frames}")

            elapsed = time.perf_counter() - start_t
            time.sleep(max(0, 1/30 - elapsed)) # 模拟 30FPS

        print("\n--- 步骤 4: 停止录制并持久化数据 ---")
        dataset.save_episode() # 将缓冲区数据写入 Parquet 和 PNG

        dataset.finalize() # 生成视频编码和统计信息

        dataset.stop_image_writer()

        print(f"✅ 数据集已成功保存至: {output_dir}")
        print(f"包含文件: {[f.name for f in (output_dir / 'data').iterdir()]}")
        print("\n✅ 录制流验证完成！")

    except Exception as e:
        print(f"\n❌ 运行中出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        robot.disconnect()
        print("--- 机器人已断开连接 ---")

if __name__ == "__main__":
    test_recording_flow()