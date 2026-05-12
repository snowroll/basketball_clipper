# Basketball Goal Detector

自动检测篮球视频中的进球事件，输出进球前3s+后1s的完整视频片段。

主程序：`backboard_ball_detector.py`

## 检测流程

```
Phase 1: 篮板ROI滚动差分扫描（每0.1s采样）
    ↓ 球候选帧
Phase 2: 聚类（gap≤2s合并为事件）
    ↓ 事件列表
Phase 3: 轨迹分析 + 进球判定 + 漂移修正
    ↓ 进球事件
输出: 完整画面视频片段（进球前3s + 后1s）
```

## 快速开始

```bash
# 运行检测（首次运行若没有ROI文件，会交互式选择参考帧并框选ROI）
python3 backboard_ball_detector.py video.mp4

# 指定输出目录
python3 backboard_ball_detector.py video.mp4 --output-dir my_output

# 保留事件截图和轨迹调试图
python3 backboard_ball_detector.py video.mp4 --keep-debug-images

# 输出耗时统计
python3 backboard_ball_detector.py video.mp4 --profile
```

ROI文件命名规则：`video_backboard_roi.txt`、`video_hoop_roi.txt`（与视频同名前缀）

## ROI 标定说明

- **篮板ROI**：框住整个篮板矩形，包含篮筐在内，适当留出余量
- **篮筐ROI**：精确框住篮筐内圆口（篮筐开口，不含篮板）
- 坐标格式：`x,y,w,h`（绝对像素坐标）
- 首次运行主程序时，如果 ROI 文件不存在，会先进入参考帧选择窗口：拖动画面底部进度条选帧，`A/D` 或方向键可单帧微调，`Enter` 确认。
- 参考帧确认后，按提示依次框选篮板 ROI 和篮筐 ROI，`Enter` 确认，`R` 重绘，`Q` 退出。

```
test-22_backboard_roi.txt: 655,320,83,48
test-22_hoop_roi.txt:      671,358,26,25
```

## 进球判定逻辑（check_basket）

1. **排除静止伪影**：所有轨迹点聚在3px范围内 → 拒绝
2. **轨迹去噪**：去除占40%以上的静止聚类点，暴露真实球运动
3. **Rule 1**：球必须有至少一个点高于篮筐口（y < hoop_y_rel）
4. **Rule 2a**：球从篮筐上方穿过（上→下），且穿越X坐标在篮筐范围±4px内，穿越后不反弹回篮筐上方 → 进球
5. **Rule 2b**：球在event_time+0.3s内消失，最后X在篮筐范围±4px内，且向下运动 → 进球（穿网）

## 镜头漂移修正

长时间拍摄时镜头会缓慢偏移，导致篮筐在ROI内的位置变化。程序在每次轨迹分析前，
通过模板匹配将当前帧与视频第一帧比较，自动计算偏移量（dx, dy），动态修正篮筐坐标。

test-22.mp4 实测漂移：视频开始约+2px，11分钟时峰值+13px，之后略有回落（均为水平方向）。

## 长事件处理

事件跨度 > 3s 时，以 best_frame（信号最强帧）为中心，同时将轨迹分析窗口向前延伸至
事件起始帧（`extra_pre = event_time - event_start`），确保进球时刻落在分析范围内。

## 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--interval` | 0.1s | Phase 1 采样间隔 |
| `--rolling-window` | 1.5s | 滚动差分参考窗口 |
| `--threshold` | 25 | 差分二值化阈值 |
| `--min-coverage` | 0.55 | 球检测差分覆盖率下限 |
| `--cluster-gap` | 2.0s | 事件聚类间隔阈值 |
| `--traj-window` | 1.0s | 轨迹分析窗口（事件前后各N秒） |
| `--traj-step` | 0.05s | 轨迹采样步长 |
| `--keep-debug-images` | 关闭 | 保留事件截图和轨迹 PNG |
| `--profile` | 关闭 | 输出扫描和事件处理耗时 |

## 输出结构

默认只保留进球视频，事件截图和轨迹图不会落盘。需要调试检测过程时，加 `--keep-debug-images`。

```
output_dir/
└── baskets/
    ├── basket01_t17.9s.mp4    # 进球完整画面（前3s + 后1s）
    ├── basket02_t67.5s.mp4
    └── ...
```

加 `--keep-debug-images` 后会额外输出：

```
output_dir/
├── event01_17.9s.png          # Phase 1/2 检测帧（篮板ROI + 差分热力图）
├── event01_17.9s_traj.png     # Phase 3 轨迹图（黄线=篮筐口，蓝→红=时间）
└── ...
```

