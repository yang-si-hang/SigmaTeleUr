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