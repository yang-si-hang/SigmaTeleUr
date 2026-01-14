"""
Unified Robotics Data Processor (Strict Intersection Mode)
Date: 2026-01-10
Function: 
1. 基于机器人和图像的时间戳计算严格交集。
2. 支持通过外部汇总表进一步裁剪 Start/End。
3. 仅导出交集范围内的数据，不进行任何外推。
"""
import h5py
import cv2
import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from const import DEMO_PATH, SAMPLE_PATH


def load_summary_metadata(csv_path):
    """读取汇总表，返回 {episode_id: (start, end)} 字典"""
    if not isinstance(csv_path, Path):
        csv_path = Path(csv_path)
    
    if not csv_path.exists():
        return {}
    
    try:
        df = pd.read_csv(csv_path, dtype={'episode_id': str})
        # 确保列名存在，这里假设列名为 episode_id, start_time, end_time
        meta_dict = {}
        for _, row in df.iterrows():
            s = float(row['start_time']) if pd.notnull(row.get('start_time')) else None
            e = float(row['end_time']) if pd.notnull(row.get('end_time')) else None
            meta_dict[row['episode_id']] = (s, e)
        print(f"ℹ️ 已加载汇总表，包含 {len(meta_dict)} 条记录。")
        return meta_dict
    except Exception as e:
        print(f"⚠️ 汇总表读取失败: {e}，将仅使用自动对齐。")
        return {}

def process_episode_strictly(ep_dir, output_root, manual_bounds=None):
    """
    Args:
        ep_dir: 轨迹目录
        output_root: 输出根目录
        manual_bounds: tuple (start, end) or None, 来自汇总表的手动裁剪点
    """
    ep_id = ep_dir.name
    current_output_folder = output_root / ep_id
    
    # 预先清理
    if current_output_folder.exists():
        shutil.rmtree(current_output_folder)
    
    try:
        # 1. 资源定位
        img_h5_files = list(ep_dir.glob("capture_*.h5"))
        robot_h5_files = list(ep_dir.glob("robot_trajectory_*.h5"))

        if not img_h5_files or not robot_h5_files:
            return False

        latest_img_h5 = max(img_h5_files, key=os.path.getctime)
        latest_robot_h5 = max(robot_h5_files, key=os.path.getctime)

        # ---------------------------------------------------------------------
        # 核心逻辑修改：先读取时间戳，计算 Intersection (交集)
        # ---------------------------------------------------------------------
        
        # A. 读取机器人时间范围
        with h5py.File(latest_robot_h5, 'r') as f:
            robot_ts = f['timestamp'][:]
            # 读取所有数据备用
            robot_poses = f['actual_tcp_pose'][:]
            robot_gripper = f['actual_gripper_pos'][:]

        if len(robot_ts) < 2: 
            return False
        
        r_start, r_end = robot_ts[0], robot_ts[-1]

        # B. 读取图像时间范围
        with h5py.File(latest_img_h5, 'r') as f:
            img_ts_all = f['system_time'][:]
            # 此时先不读由图像内容，只读元数据以节省内存
            color_ds = f['color']
            # 获取维度信息用于列名
            # img_shape_str = f"[{','.join(map(str, color_ds.shape[1:]))}]"
            
        if len(img_ts_all) == 0:
            return False
            
        i_start, i_end = img_ts_all[0], img_ts_all[-1]

        # C. 计算有效交集 (Intersection Logic)
        # 基础交集：图像 与 机器人
        valid_start = max(r_start, i_start)
        valid_end = min(r_end, i_end)

        # 叠加汇总表限制 (如果有)
        if manual_bounds:
            m_start, m_end = manual_bounds
            if m_start is not None:
                valid_start = max(valid_start, m_start) # 取较晚的开始时间
            if m_end is not None:
                valid_end = min(valid_end, m_end)       # 取较早的结束时间

        # D. 最终有效性检查
        # 如果起点 >= 终点，说明没有交集或裁剪过度
        if valid_start >= valid_end:
            # print(f"⚠️ {ep_id} 无有效时间交集，跳过。")
            return False

        # ---------------------------------------------------------------------
        # 执行数据提取与对齐
        # ---------------------------------------------------------------------
        
        img_dir = current_output_folder / "images"
        img_dir.mkdir(parents=True)
        img_col_name = f"ImagePath"

        # E. 筛选图像：只处理 valid_start 到 valid_end 之间的帧
        # 使用 mask 进行筛选，不修改原始 img_ts_all 的数值（保留原始时间戳）
        mask = (img_ts_all >= valid_start) & (img_ts_all <= valid_end)
        valid_indices = np.where(mask)[0]

        if len(valid_indices) == 0:
            shutil.rmtree(current_output_folder)
            return False

        selected_data = []
        
        # 重新打开图像H5进行读取 (或者保持打开状态)
        with h5py.File(latest_img_h5, 'r') as f:
            color_ds = f['color']
            
            for idx in valid_indices:
                ts = img_ts_all[idx]
                img = color_ds[idx]
                
                # 导出图片
                img_name = f"{ts:.6f}.png"
                img_rel_path = f"images/{img_name}"
                img_full_path = str(img_dir / img_name)
                
                cv2.imwrite(img_full_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                selected_data.append({'timestamp': ts, img_col_name: img_rel_path})

        # F. 对齐机器人数据 (插值)
        # 这里的 target_ts 必然都在 robot_ts 的范围内，因此不需要 extrapolate
        target_ts = np.array([d['timestamp'] for d in selected_data])

        # 线性插值
        # 注意：不再使用 fill_value="extrapolate"，如果越界理应报错（但前面已过滤）
        f_pos = interp1d(robot_ts, robot_poses[:, :3], axis=0, kind='linear')
        f_gripper = interp1d(robot_ts, robot_gripper, axis=0, kind='linear')
        
        interp_xyz = f_pos(target_ts)
        interp_gripper = f_gripper(target_ts)

        # 旋转插值 (Slerp)
        rotations = R.from_rotvec(robot_poses[:, 3:6])
        slerp = Slerp(robot_ts, rotations)
        interp_rotvec = slerp(target_ts).as_rotvec()

        # G. 保存 CSV
        df = pd.DataFrame(selected_data)
        df['X'], df['Y'], df['Z'] = interp_xyz[:, 0], interp_xyz[:, 1], interp_xyz[:, 2]
        df['Rx'], df['Ry'], df['Rz'] = interp_rotvec[:, 0], interp_rotvec[:, 1], interp_rotvec[:, 2]
        df['GripperPos'] = interp_gripper

        cols_order = ['timestamp', 'X', 'Y', 'Z', 'Rx', 'Ry', 'Rz', 'GripperPos', img_col_name]
        df[cols_order].to_csv(current_output_folder / "all_data.csv", index=False)
        
        return True

    except Exception as e:
        if current_output_folder.exists():
            shutil.rmtree(current_output_folder)
        print(f"❌ 处理出错 {ep_id}: {e}")
        return False

def run_conversion(input_root, output_root, time_horizon_file):
    """
    Args:
        input_root: 输入数据根目录
        output_root: 输出数据根目录
        time_horizon_file: 时间区间汇总表路径 (可选)
    """
    input_root = Path(input_root)
    output_root = Path(output_root)

    # 0. 加载汇总表 (如果有)
    metadata = load_summary_metadata(time_horizon_file)

    episodes = sorted([d for d in input_root.iterdir() if d.is_dir() and d.name.isdigit()])
    
    print("="*50)
    print(f"🚀 开始转换 (严格交集模式)")
    print(f"📂 输入: {input_root}")
    print(f"📂 输出: {output_root}")
    if metadata:
        print(f"📋 启用汇总表裁剪，覆盖 {len(metadata)} 条记录")
    print("="*50)

    success_count = 0
    fail_count = 0

    for ep in tqdm(episodes, desc="Processing"):
        # 获取该轨迹的手动裁剪点，如果没有则为 None
        bounds = metadata.get(ep.name, None)
        
        if process_episode_strictly(ep, output_root, manual_bounds=bounds):
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "="*50)
    print(f"✅ 完成! 成功: {success_count} | 放弃: {fail_count}")
    print("="*50)


if __name__ == "__main__":
    SUMMARY_METADATA_PATH = SAMPLE_PATH / "202601061638" / "metadata.csv" 
    INPUT_DIR = SAMPLE_PATH / "202601061638"
    OUTPUT_DIR = DEMO_PATH / "grasp_soda" / "202601061638_strict"

    run_conversion(INPUT_DIR, OUTPUT_DIR, SUMMARY_METADATA_PATH)