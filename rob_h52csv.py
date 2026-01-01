import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from const import DEMO_PATH

def align_robot_data(h5_path, target_csv_path, output_csv_path):
    """
    根据目标 CSV 的时间戳，对 H5 中的机器人轨迹进行插值对齐。
    """
    # 1. 加载目标时间戳
    target_df = pd.read_csv(target_csv_path)
    # 假设第一列是时间戳
    target_ts = target_df.iloc[:, 0].values
    correspond_img_path = target_df.iloc[:, 1].values
    
    # 2. 从 H5 读取原始机器人数据
    with h5py.File(h5_path, 'r') as f:
        source_ts = f['timestamp'][:]
        poses = f['actual_tcp_pose'][:]      # (N, 6) -> X, Y, Z, Rx, Ry, Rz
        gripper = f['actual_gripper_pos'][:] # (N,)
    
    print(f"原始数据帧数: {len(source_ts)}")
    print(f"目标对齐帧数: {len(target_ts)}")

    # 3. 线性插值：位置 (X, Y, Z) 和 夹爪 (Gripper)
    # interp1d 会根据 source_ts 生成一个映射函数
    f_pos = interp1d(source_ts, poses[:, :3], axis=0, fill_value="extrapolate")
    f_gripper = interp1d(source_ts, gripper, axis=0, fill_value="extrapolate")
    
    interp_xyz = f_pos(target_ts)
    interp_gripper = f_gripper(target_ts)

    # 4. 球面线性插值 (SLERP)：旋转 (Rx, Ry, Rz)
    # 旋转向量转为四元数
    rotations = R.from_rotvec(poses[:, 3:6])
    slerp = Slerp(source_ts, rotations)
    # 在目标时间戳进行插值
    interp_rots = slerp(target_ts)
    interp_rotvec = interp_rots.as_rotvec() # 转回旋转向量

    # 5. 合并数据
    # 格式：Timestamp, X, Y, Z, Rx, Ry, Rz, GripperPos
    result_df = pd.DataFrame({
        'Timestamp': target_ts,
        'X': interp_xyz[:, 0],
        'Y': interp_xyz[:, 1],
        'Z': interp_xyz[:, 2],
        'Rx': interp_rotvec[:, 0],
        'Ry': interp_rotvec[:, 1],
        'Rz': interp_rotvec[:, 2],
        'GripperPos': interp_gripper,
        'ImagePath': correspond_img_path  # 字符串路径可以正常存放
    })

    # 保存 CSV，不保存索引列
    result_df.to_csv(output_csv_path, index=False)

    print(f"对齐数据已保存至: {output_csv_path}")

if __name__ == "__main__":
    # 配置路径
    H5_FILE = DEMO_PATH / "robot_trajectory_20260102_160419.h5"
    TARGET_TIMESTAMPS = DEMO_PATH / "grasp_soda" / "000" / "timestamps.csv" # 上一步保存的图片时间戳文件
    OUTPUT_FILE = DEMO_PATH / "grasp_soda" / "000" / "all_data.csv"
    
    align_robot_data(H5_FILE, TARGET_TIMESTAMPS, OUTPUT_FILE)