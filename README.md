### FroceDimension Python Package
The Python package for control the Sigma device: [https://github.com/EmDash00/forcedimension-python](https://github.com/EmDash00/forcedimension-python)

You need set the `FDSDK` environment varable:

```bash
export FDSDK="$HOME/ysh/Program/sdk-3.17.6"
```

### Coordinate Frame Transformation


### Data Collection

相关的代码：

- `master_sigma.py`: publish the pose data and connect signal
- `slave_ur.py`: subscribe the sigma pose and connect signal, then save the ur data.
- `camera_datasample.py`: subscribe the connect signal and save the image data

运行逻辑：

- 三个代码文件开启先后顺序：`master_sigma.py` -> `camera_datasample.py` -> `slave_ur.py`
- `slave_ur.py`一直使能姿态和夹爪开合映射
- 在`master_sigma.py`中键入`t`，使能位置映射，同时开始记录图像和机器人位姿数据，开始遥操作动作

数据转换：

- 使用`camera_h52csv.py`，将hdf5文件中的图片单独存到一个文件夹，然后用时间戳命名对应的图像文件，png格式。

    在转换之前，通过开始和结束时间戳，截取图片数据集中有效的区间。

- 使用`rob_h52csv.py`，将机器人轨迹数据与图片数据对齐，并保存为`all_data.csv`文件：包含时间戳，对应的机器人轨迹，图片路径

- 使用`data_assemble.py`，直接对齐处理图像和机器人轨迹数据，转换为images文件夹和csv格式的机器人轨迹

### LeRobot Description

- "image.name": info.json的信息，需要在转换代码中手动给定

- "chunk_size": 数据分块的个数(每个块最多1000 frame ?)

- "q01, q10...": 数据从小到大的对应百分位位置的数值