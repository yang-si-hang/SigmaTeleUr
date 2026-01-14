"""
将本地的数据处理成lerobot需要的格式
Date: 2026-1-5
"""
import pandas as pd
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import shutil
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from const import DEMO_PATH, DATA_PATH, SAMPLE_PATH

# lerobot要求必须指定task参数
TASK_DESCRIPTION = "grasp soda can with gripper"

def convert_episodes_to_lerobot(raw_root_dir, output_repo, ACTION_AS_DELTA=True):
    raw_root = Path(raw_root_dir)
    
    # 定义状态列名（匹配您的 CSV）
    # X, Y, Z, Rx, Ry, Rz, GripperPos
    state_columns = ['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz', 'GripperPos']
    Image_names = ["channel", "height", "width"],

    if isinstance(output_repo, Path):
        output_repo = str(output_repo)

    # 1. 初始化数据集
    dataset = LeRobotDataset.create(
        repo_id=output_repo,
        fps=30,
        robot_type="UR10e", 
        features={
            # 状态向量 (6+1)
            "observation.state": {
                "dtype": "float32", 
                "shape": (len(state_columns),),
                "names": state_columns
            },
            # 图像观测 (LeRobot 会自动调用 ffmpeg 编码)
            "observation.images.main": {
                "dtype": "video", 
                "shape": (3, 480, 640),
                "names": Image_names
            },
            # 动作向量
            "action": {
                "dtype": "float32", 
                "shape": (len(state_columns),),
                "names": state_columns
            },
        },
        video_backend="torchcodec"
    )

    # 2. 识别并排序所有轨迹文件夹 (000, 001, ...)
    # 这里通过 d.name.isdigit() 过滤掉非轨迹文件夹
    episode_dirs = sorted([
        d for d in raw_root.iterdir() 
        if d.is_dir() and d.name.isdigit()
    ])

    if not episode_dirs:
        print(f"❌ 在 {raw_root_dir} 下未找到任何数字命名的轨迹文件夹")
        return

    print(f"🚀 找到 {len(episode_dirs)} 条轨迹，开始转换...")

    for ep_dir in tqdm(episode_dirs, desc="Processing Episodes"):
        csv_path = ep_dir / "all_data.csv"
        # 注意：根据您的示例，ImagePath 包含 "images/" 前缀，
        # 所以我们需要确保相对于 ep_dir 的路径正确
        
        if not csv_path.exists():
            print(f"⚠️ 跳过 {ep_dir.name}: 未找到 all_data.csv")
            continue

        df = pd.read_csv(csv_path)
        num_frames = len(df)

        # 遍历当前 Episode 的每一帧
        for i in range(len(df)-1):
            row = df.iloc[i]
            next_row = df.iloc[i + 1]

            # A. 处理图像
            # row['ImagePath'] 已经是 "images/xxx.png"
            full_img_path = ep_dir / row['ImagePath']
            img = Image.open(full_img_path).convert("RGB")
            # 转换为 (C, H, W) uint8 tensor
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)

            # B. 提取状态 (State)
            state_array = row[state_columns].values.astype(np.float32)

            # C. 提取动作 (Action)
            next_state_array = next_row[state_columns].values.astype(np.float32)

            if ACTION_AS_DELTA:
                # delta state
                action_array = next_state_array - state_array
            else:
                # absolute state
                action_array = next_state_array

            # D. 压入数据
            payload = {
                "observation.state": torch.from_numpy(state_array),
                "observation.images.main": img_tensor,
                "action": torch.from_numpy(action_array),
                "task": TASK_DESCRIPTION,
            }
            
            dataset.add_frame(payload)

        dataset.save_episode()

    # 3. 固化数据 (生成视频编码、计算全局统计信息)
    dataset.finalize()
    print(f"\n✅ 转换完成！数据集位于: {output_repo}")

# 执行转换
if __name__ == "__main__":
    # --- 配置区域 ---
    # 假设你的数据集结构是：
    # dataset_root/
    #   ├── 000/
    #   │    ├── all_data.csv
    #   │    └── images/
    #   ├── 001/
    #   ...

    epsiode_path = DEMO_PATH / "grasp_soda" / "202601061638_strict"
    
    lerobot_path = DEMO_PATH / "lerobot" / "grasp_soda" / "202601061638-full-absolute"
    if lerobot_path.exists():
        shutil.rmtree(lerobot_path)
    # 请确保该目录下有 episode_xxx 文件夹，每个文件夹内有 data.csv
    convert_episodes_to_lerobot(epsiode_path, lerobot_path, ACTION_AS_DELTA=False)
