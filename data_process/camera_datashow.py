"""
Camera Data Viewer and Frame Rate Analyzer
Date: 2026-1-2
"""
import h5py
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

from const import DEMO_PATH, VIS_PATH, SAMPLE_PATH

def main(h5_path):
    ts_data_for_plot = None

    # 以只读模式打开文件
    with h5py.File(h5_path, 'r') as h5f:
        # 1. 读取元数据和数据集引用
        color_ds = h5f['color']
        sys_ts = h5f['system_time']
        dev_ts = h5f['device_time']

        ts_data_for_plot = sys_ts[:]
        
        # 读取内参（作为演示，这里只打印，你可以用于点云重建）
        meta = h5f['metadata'].attrs
        print(f"--- 载入数据信息 ---")
        print(f"总帧数: {len(color_ds)}")
        print(f"相机内参: fx={meta['fx']:.2f}, fy={meta['fy']:.2f}")
        print(f"录制起始时间: {time.ctime(sys_ts[0])}")
        
        # 2. 创建 OpenCV 窗口和滑动条
        cv2.namedWindow("HDF5 Player", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Frame", "HDF5 Player", 0, len(color_ds) - 1, lambda x: None)

        print("\n操作指南:")
        print("- 拖动滑动条预览不同帧")
        print("- 按 'q' 键退出播放器")

        while True:
            # 获取滑动条当前位置（即帧索引）
            idx = cv2.getTrackbarPos("Frame", "HDF5 Player")

            # 读取该帧数据
            color_img = color_ds[idx]
            s_time = sys_ts[idx]
            d_time = dev_ts[idx]

            # 在彩色图上标注时间戳信息
            info_text = f"Frame: {idx} | SysTS: {s_time:.4f} | DevTS: {d_time}"
            cv2.putText(color_img, info_text, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 拼接显示：左彩色，右深度
            combined_view = color_img
            
            cv2.imshow("HDF5 Player", combined_view)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

    # 分析帧率并保存绘图
    if ts_data_for_plot is not None and len(ts_data_for_plot) > 1:
        print("正在生成帧率分析图表...")
        
        # 1. 计算时间差 (Delta Time)
        # np.diff 计算相邻元素的差值: t[i+1] - t[i]
        deltas = np.diff(ts_data_for_plot)
        
        # 2. 处理异常值 (防止除以0)
        # 将极小的时间间隔替换为 1e-6，防止计算 FPS 出现无穷大
        deltas = np.maximum(deltas, 1e-6)
        
        # 3. 计算瞬时帧率 (FPS = 1 / dt)
        fps = 1.0 / deltas
        
        # 4. 绘制折线图
        plt.figure(figsize=(12, 6))
        
        # 绘制数据
        plt.plot(fps, color='#1f77b4', linewidth=1, label='Instant FPS')
        
        # 添加一条平均帧率的虚线
        mean_fps = np.mean(fps)
        plt.axhline(mean_fps, color='red', linestyle='--', label=f'Mean FPS: {mean_fps:.2f}')
        
        # 图表装饰
        plt.title('Frame Rate Analysis over Time Steps', fontsize=14)
        plt.xlabel('Frame Index', fontsize=12)
        plt.ylabel('FPS (Hz)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 5. 保存图片
        save_name = VIS_PATH / 'camera_data_frequency.svg'
        plt.savefig(save_name, dpi=150)
        print(f"图表已保存为: {save_name}")
        
        # 可选：如果你想直接弹窗显示图表，取消下面这行的注释
        # plt.show()
    else:
        print("数据不足，无法生成帧率图表。")


if __name__ == "__main__":
    # 请确保文件名与你保存的文件名一致
    import glob
    dir_path = SAMPLE_PATH / "202601061638" / "027"
    h5_files = glob.glob(str(dir_path / "capture_*.h5"))
    if h5_files:
        latest_file = max(h5_files) # 默认读取最新的文件
        main(latest_file)
    else:
        print(f"路径{dir_path}未找到 .h5 文件，请先运行采样脚本。")