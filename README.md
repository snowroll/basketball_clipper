# 篮球投篮自动剪辑工具

自动检测篮球视频中投篮穿过篮筐的瞬间，剪辑并输出视频片段。

## 依赖

- Python 3.11+
- OpenCV (`opencv-python-headless` 或 `opencv-python`)
- NumPy

```bash
pip install opencv-python-headless numpy
```

## 使用方法

### 基本用法

```bash
python basketball_clipper.py input_video.mp4
```

首次运行会弹出 OpenCV 窗口，用鼠标拖拽框选篮筐位置，按 Enter 确认。ROI 会自动保存到 `视频名_roi.txt`，下次可复用。

### 指定篮筐位置

```bash
# 直接传入 ROI (x, y, width, height)
python basketball_clipper.py input_video.mp4 --roi 300,100,80,60

# 从文件读取 ROI
python basketball_clipper.py input_video.mp4 --roi-file video_roi.txt
```

### 调整检测参数

```bash
# 提高灵敏度（适合光线较暗或球速较快的视频）
python basketball_clipper.py input_video.mp4 --threshold 20 --min-area 150

# 降低灵敏度（减少误检）
python basketball_clipper.py input_video.mp4 --threshold 35 --min-area 300
```

### 自定义剪辑时长

```bash
# 剪辑事件前 8 秒 + 后 2 秒
python basketball_clipper.py input_video.mp4 --clip-before 8 --clip-after 2
```

### 自定义输出目录

```bash
python basketball_clipper.py input_video.mp4 --output-dir my_clips
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input` | (必填) | 输入视频路径 |
| `--roi` | 无 | 篮筐 ROI: x,y,w,h |
| `--roi-file` | 无 | 从文件读取 ROI |
| `--threshold` | 25 | 帧差二值化阈值，越小越敏感 |
| `--min-area` | 200 | 最小轮廓面积，过滤噪声 |
| `--clip-before` | 5.0 | 剪辑事件前秒数 |
| `--clip-after` | 1.0 | 剪辑事件后秒数 |
| `--output-dir` | output | 输出目录 |

## ROI 标注操作

- **鼠标拖拽**: 绘制矩形框
- **Enter / Space**: 确认选区
- **R**: 重新绘制
- **Q**: 退出

## 输出

- 输出格式：MP4 (H.264)
- 文件命名：`shot_001_12.5s.mp4`（序号 + 事件时间）
- 默认保存到 `output/` 目录

## 检测原理

1. 每秒提取 1 帧
2. 对篮筐区域相邻帧做差分，检测新出现的运动物体
3. 验证运动轨迹是否从上到下穿过方框（排除水平运动、反弹等）
4. 确认投篮后剪辑前后视频片段
