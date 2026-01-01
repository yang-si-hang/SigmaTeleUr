"""
Trasnfer images and timestamps from HDF5 file to PNG and CSV format.
Date: 2026-1-2
"""
import h5py
import cv2
import os
import csv
import numpy as np
from pathlib import Path

from const import DEMO_PATH

def export_h5_data(h5_path, output_dir, start_time, end_time):
    """
    导出指定时间段内的图片（无损）和时间戳（CSV）
    """
    # 初始化路径
    output_path = Path(output_dir)
    img_dir = output_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file_path = output_path / "timestamps.csv"

    print(f"正在读取文件: {h5_path}")
    
    with h5py.File(h5_path, 'r') as h5f:
        # 获取数据集
        color_ds = h5f['color']
        sys_ts = h5f['system_time'][:]  # 读取到内存以便快速检索
        
        # 1. 筛选时间范围内的索引
        # 找到满足 start_time <= ts <= end_time 的所有索引
        indices = np.where((sys_ts >= start_time) & (sys_ts <= end_time))[0]
        
        if len(indices) == 0:
            print("错误: 在指定的时间范围内未找到任何数据。")
            return

        print(f"匹配到 {len(indices)} 帧。正在导出...")

        # 准备 CSV 数据
        selected_timestamps = []

        for idx in indices:
            ts = sys_ts[idx]
            img = color_ds[idx]
            
            # 2. 保存图片 (无损 PNG)
            # 文件名使用原始时间戳（保留小数以防重名）
            img_name = f"{ts:.6f}.png"
            img_rel_path = os.path.join("images", img_name)
            img_full_path = str(img_dir / img_name)
            
            # cv2.imwrite 对于 .png 默认是无损的
            # 可以通过 [cv2.IMWRITE_PNG_COMPRESSION, 0] 设置压缩等级，0 为不压缩
            cv2.imwrite(img_full_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
            # 存入列表准备写 CSV
            selected_timestamps.append([ts, img_rel_path])

        # 3. 保存 CSV 文件
        with open(csv_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'image_path']) # 表头
            writer.writerows(selected_timestamps)

    print(f"--- 导出完成 ---")
    print(f"图片保存目录: {img_dir}")
    print(f"时间戳文件: {csv_file_path}")

if __name__ == "__main__":
    # --- 配置参数 ---
    H5_FILE = DEMO_PATH / "capture_20260102_160402.h5"  # 替换为你的实际文件名
    OUTPUT_FOLDER = DEMO_PATH / "grasp_soda" / "000"
    
    # 填入你想要截取的时间戳范围 (float)
    START = 1767341069.2          # 示例：起始时间
    END = 1767341085.0   # 示例：结束时间
    # ----------------

    # 如果不知道具体时间戳，可以先运行你之前的脚本查看打印出的起始时间
    export_h5_data(H5_FILE, OUTPUT_FOLDER, START, END)