#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
篮板碰球检测器

1. 每隔 0.3s 采一帧，滚动基础帧差分
2. 多重过滤确认球体：面积 / 圆形度 / 边缘排除 / 差分覆盖率
3. 事件聚类（gap ≤ 1s）
4. 对每个事件做 ±0.5s 轨迹分析，判断是否进球
5. 进球事件剪辑篮板 ROI 视频
"""

import argparse
import collections
import os
import shutil
import subprocess
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


_FFMPEG_BIN = None
_MORPH_KERNEL_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

def find_ffmpeg():
    """查找 ffmpeg 可执行文件路径，找不到返回 None。"""
    global _FFMPEG_BIN
    if _FFMPEG_BIN is not None:
        return _FFMPEG_BIN or None
    path = shutil.which("ffmpeg")
    if not path:
        for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/opt/local/bin/ffmpeg"):
            if os.path.exists(p):
                path = p
                break
    _FFMPEG_BIN = path or ""
    return path or None

# ── PIL 文字渲染 ─────────────────────────────────────────────────────────────

_CJK_FONT = None

def _find_cjk_font():
    global _CJK_FONT
    if _CJK_FONT is not None:
        return _CJK_FONT
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            _CJK_FONT = p
            return p
    _CJK_FONT = ""
    return ""


_FONT_CACHE = {}

def _get_pil_font(size):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    path = _find_cjk_font()
    font = ImageFont.load_default()
    if path:
        try:
            font = ImageFont.truetype(path, size)
        except Exception:
            pass
    _FONT_CACHE[size] = font
    return font


def pil_put_text(frame_bgr, text, org, font_size=16, color=(255, 255, 255),
                 bg_color=None, anchor="lt"):
    """在 OpenCV BGR 帧上用 PIL 绘制文字，支持中文。

    Args:
        frame_bgr: numpy BGR 图像 (就地修改)
        text:      文字内容
        org:       (x, y) 左上角坐标
        font_size: 字体大小
        color:     BGR 颜色
        bg_color:  背景色 (BGR)，None 表示无背景
        anchor:    PIL 锚点 (lt=左上)
    Returns: 修改后的帧
    """
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _get_pil_font(font_size)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = org

    if bg_color is not None:
        pad = 4
        bg_rgb = (bg_color[2], bg_color[1], bg_color[0])
        draw.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad], fill=bg_rgb)

    txt_rgb = (color[2], color[1], color[0])
    draw.text((x, y), text, font=font, fill=txt_rgb)

    cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR, dst=frame_bgr)
    return frame_bgr


def load_roi(path):
    with open(path) as f:
        return tuple(int(p) for p in f.read().strip().split(','))


def crop_roi(frame, roi):
    x, y, w, h = roi
    fh, fw = frame.shape[:2]
    return frame[max(0, y):min(fh, y + h), max(0, x):min(fw, x + w)]


def select_frame(video_path):
    """用进度条手动选取一帧，返回 numpy 帧或 None。"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0:
        cap.release()
        return None

    cur_fn = [0]
    frame_cache = [None]
    dragging = [False]

    def load(f):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, img = cap.read()
        return img if ret else None

    frame_cache[0] = load(0)

    win = "选择参考帧  Enter确认 | Q退出"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    hint = "拖动底部进度条选帧 | Enter确认 | Q退出"

    def seek_from_x(x, image_width):
        bar_x1 = 28
        bar_x2 = max(bar_x1 + 1, image_width - 28)
        ratio = (min(max(x, bar_x1), bar_x2) - bar_x1) / (bar_x2 - bar_x1)
        fn = int(round(ratio * (total - 1)))
        if fn != cur_fn[0]:
            cur_fn[0] = fn
            new_frame = load(fn)
            if new_frame is not None:
                frame_cache[0] = new_frame

    def mouse_cb(event, x, y, flags, _):
        if frame_cache[0] is None:
            return
        h, w = frame_cache[0].shape[:2]
        bar_y = h - 38
        in_bar = bar_y - 16 <= y <= bar_y + 16
        if event == cv2.EVENT_LBUTTONDOWN and in_bar:
            dragging[0] = True
            seek_from_x(x, w)
        elif event == cv2.EVENT_MOUSEMOVE and dragging[0]:
            seek_from_x(x, w)
        elif event == cv2.EVENT_LBUTTONUP:
            if dragging[0]:
                seek_from_x(x, w)
            dragging[0] = False

    cv2.setMouseCallback(win, mouse_cb)

    while True:
        if frame_cache[0] is not None:
            display = frame_cache[0].copy()
            fn = cur_fn[0]
            sec = fn / fps
            pil_put_text(display, f"帧 {fn}/{total}  {sec:.2f}s", (8, 8),
                         font_size=30, color=(0, 255, 255), bg_color=(0, 0, 0))
            pil_put_text(display, hint, (8, 56), font_size=28, color=(200, 200, 200),
                         bg_color=(0, 0, 0))
            h, w = display.shape[:2]
            bar_x1, bar_x2 = 28, w - 28
            bar_y = h - 38
            ratio = fn / max(total - 1, 1)
            knob_x = int(round(bar_x1 + ratio * (bar_x2 - bar_x1)))
            cv2.line(display, (bar_x1, bar_y), (bar_x2, bar_y), (80, 80, 80), 8, cv2.LINE_AA)
            cv2.line(display, (bar_x1, bar_y), (knob_x, bar_y), (0, 220, 255), 8, cv2.LINE_AA)
            cv2.circle(display, (knob_x, bar_y), 13, (0, 220, 255), -1, cv2.LINE_AA)
            cv2.circle(display, (knob_x, bar_y), 13, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow(win, display)

        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32) and frame_cache[0] is not None:
            cv2.destroyWindow(win)
            cap.release()
            return frame_cache[0]
        elif key == ord('q'):
            cv2.destroyWindow(win)
            cap.release()
            return None
        elif key in (81, 2424832, ord('a')):
            cur_fn[0] = max(0, cur_fn[0] - 1)
            frame_cache[0] = load(cur_fn[0])
        elif key in (83, 2555904, ord('d')):
            cur_fn[0] = min(total - 1, cur_fn[0] + 1)
            frame_cache[0] = load(cur_fn[0])


def select_roi(frame, label="ROI"):
    """在已有的第一帧上交互式框选 ROI，返回 (x, y, w, h) 或 None。
    窗口在确认后立刻关闭。
    """
    roi = None
    drawing = False
    x1 = y1 = x2 = y2 = 0

    hint = f"框选{label}  拖拽绘制 | Enter确认 | R重绘 | Q退出"
    # 预渲染提示文字到 base 上，鼠标回调里不再做文字渲染
    base = frame.copy()
    pil_put_text(base, hint, (8, 8), font_size=28, color=(200, 200, 200),
                 bg_color=(0, 0, 0))

    def mouse_cb(event, x, y, flags, _):
        nonlocal drawing, x1, y1, x2, y2, roi
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            x1, y1 = x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            x2, y2 = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x2, y2 = x, y
            xn, yn = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            if w > 5 and h > 5:
                roi = (xn, yn, w, h)

    win = f"框选{label}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    cv2.setMouseCallback(win, mouse_cb)

    while True:
        # 每帧从 base 复制，只做 cv2.rectangle（纯 C 实现，极快）
        display = base.copy()
        if drawing:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        elif roi:
            xn, yn, w, h = roi
            cv2.rectangle(display, (xn, yn), (xn + w, yn + h), (0, 255, 0), 2)
            pil_put_text(display, f"{label}: x={xn} y={yn} w={w} h={h}",
                         (8, 8), font_size=28, color=(0, 255, 0), bg_color=(0, 0, 0))
        cv2.imshow(win, display)
        key = cv2.waitKey(15) & 0xFF
        if key in (13, 32) and roi:
            cv2.destroyWindow(win)
            return roi
        elif key == ord('r'):
            roi = None
        elif key == ord('q'):
            cv2.destroyWindow(win)
            return None


def save_roi(roi, path):
    with open(path, 'w') as f:
        f.write(f"{roi[0]},{roi[1]},{roi[2]},{roi[3]}\n")


def detect_ball(diff_raw, ball_diameter, noise_thresh, min_coverage, border_margin, roi_w, roi_h):
    """差分图中检测球体：面积 / 圆形度 / 边缘距离 / 差分覆盖率四重过滤

    Returns: list of (cx, cy, radius, coverage)
    """
    _, thresh = cv2.threshold(diff_raw, noise_thresh, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, _MORPH_KERNEL_3)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    r = ball_diameter / 2
    min_area = np.pi * r ** 2 * 0.25
    max_area = np.pi * r ** 2 * 3.5

    hits = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        radius = max(1, int(np.sqrt(area / np.pi)))

        if cx < border_margin or cy < border_margin:
            continue
        if cx > roi_w - border_margin or cy > roi_h - border_margin:
            continue

        perim = cv2.arcLength(cnt, True)
        if perim > 0 and (4 * np.pi * area / perim ** 2) < 0.35:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted = cnt.copy()
        shifted[:, 0, 0] -= x
        shifted[:, 0, 1] -= y
        cv2.drawContours(mask, [shifted], -1, 255, cv2.FILLED)
        inside_total = cv2.countNonZero(mask)
        if inside_total == 0:
            continue
        roi_diff = diff_raw[y:y + h, x:x + w]
        active = int(np.count_nonzero(roi_diff[mask > 0] >= noise_thresh))
        coverage = active / inside_total
        if coverage < min_coverage:
            continue

        hits.append((cx, cy, radius, coverage))
    return hits


def make_event_image(crop_bgr, diff_filtered, detections, scale):
    """左：原图圈球  右：热力差分叠加圈球"""
    bh, bw = crop_bgr.shape[:2]
    norm = cv2.normalize(diff_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    heatmap[diff_filtered == 0] = crop_bgr[diff_filtered == 0]
    right = cv2.addWeighted(crop_bgr, 0.35, heatmap, 0.65, 0)
    left = crop_bgr.copy()
    for cx, cy, r, _ in detections:
        cv2.circle(left, (cx, cy), r, (0, 255, 0), 1)
        cv2.circle(right, (cx, cy), r, (0, 255, 0), 1)
    s = scale
    return np.hstack([
        cv2.resize(left, (bw * s, bh * s), interpolation=cv2.INTER_NEAREST),
        cv2.resize(right, (bw * s, bh * s), interpolation=cv2.INTER_NEAREST),
    ])


def cluster_detections(raw_list, max_gap):
    if not raw_list:
        return []
    events, cur = [], [raw_list[0]]
    for det in raw_list[1:]:
        if det['time'] - cur[-1]['time'] <= max_gap:
            cur.append(det)
        else:
            events.append(cur)
            cur = [det]
    events.append(cur)
    return events


def best_frame(event):
    def score(d):
        return max(h[3] for h in d['hits']) * d['change_px']
    return max(event, key=score)


# ── 轨迹分析 ────────────────────────────────────────────────────────────────

def analyze_trajectory(video_path, event_time, bb_roi, ball_diameter, fps,
                       noise_thresh, min_coverage, border_margin,
                       window=0.5, step=0.05, extra_pre=0.0):
    """
    在事件时间前后 window 秒内，每 step 秒采一帧，与事件前 2s 的背景帧差分，
    追踪球的位置，返回轨迹点和原始帧数据。

    extra_pre: 在常规窗口之前额外向前延伸的秒数（用于覆盖 event_start）

    Returns:
        trajectory: [(t, cx, cy, cov), ...]  按时间排序
        frames:     [{'t', 'crop', 'diff_filt', 'hits'}, ...]
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    roi_w, roi_h = bb_roi[2], bb_roi[3]

    # 背景帧：分析起点前 2s，用于稳定差分
    bg_fn = max(0, int(round((event_time - 2.0 - extra_pre) * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, bg_fn)
    ret, bg_frame = cap.read()
    if not ret:
        cap.release()
        return [], []
    bg_crop = crop_roi(bg_frame, bb_roi)
    if bg_crop.shape[0] != roi_h or bg_crop.shape[1] != roi_w:
        cap.release()
        return [], []
    bg_gray = cv2.cvtColor(bg_crop, cv2.COLOR_BGR2GRAY)

    step_frames = max(1, int(round(fps * step)))
    start_fn = max(0, int(round((event_time - window - extra_pre) * fps)))
    end_fn = min(total - 1, int(round((event_time + window) * fps)))

    trajectory, frames = [], []
    fn = start_fn
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_fn)
    while fn <= end_fn:
        ret, frame = cap.read()
        if not ret:
            break
        t = fn / fps
        crop = crop_roi(frame, bb_roi)
        if crop.shape[0] != roi_h or crop.shape[1] != roi_w:
            fn += step_frames
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(bg_gray, gray)
        diff_filt = diff.astype(np.float32)
        diff_filt[diff_filt < noise_thresh] = 0

        hits = detect_ball(diff, ball_diameter, noise_thresh, min_coverage,
                           border_margin, roi_w, roi_h)
        if hits:
            best = max(hits, key=lambda h: h[3])
            trajectory.append((t, best[0], best[1], best[3]))

        frames.append({'t': t, 'crop': crop.copy(),
                       'diff_filt': diff_filt.astype(np.uint8), 'hits': hits})
        fn += step_frames
        if fn <= end_fn and step_frames > 1:
            if not _skip_frames(cap, step_frames - 1):
                break

    cap.release()
    return trajectory, frames


def is_static_trajectory(trajectory, static_radius=3):
    """Return True if all trajectory points cluster within static_radius px — likely a static artifact."""
    if len(trajectory) < 3:
        return False
    xs = [p[1] for p in trajectory]
    ys = [p[2] for p in trajectory]
    return (max(xs) - min(xs)) <= static_radius and (max(ys) - min(ys)) <= static_radius


def denoise_trajectory(trajectory, mode_thresh=0.4, cluster_r=2):
    """
    Remove the dominant static cluster from a mixed trajectory.

    If one position accounts for >= mode_thresh of all points (within cluster_r px),
    strip those points out so the real ball motion is exposed.
    Applied once; caller may loop if multiple artifacts exist.

    Returns: (filtered_trajectory, n_removed)
    """
    n = len(trajectory)
    if n < 5:
        return trajectory, 0

    xs = [p[1] for p in trajectory]
    ys = [p[2] for p in trajectory]

    best_count, best_i = 0, 0
    for i in range(n):
        cnt = sum(1 for j in range(n)
                  if abs(xs[j] - xs[i]) <= cluster_r and abs(ys[j] - ys[i]) <= cluster_r)
        if cnt > best_count:
            best_count, best_i = cnt, i

    if best_count / n < mode_thresh:
        return trajectory, 0

    cx0, cy0 = xs[best_i], ys[best_i]
    filtered = [(t, cx, cy, cov) for t, cx, cy, cov in trajectory
                if not (abs(cx - cx0) <= cluster_r and abs(cy - cy0) <= cluster_r)]
    return filtered, n - len(filtered)


def compute_hoop_drift(video_path, bb_roi, fps, ref_fn, query_fn, search_range=15):
    """
    模板匹配估算镜头漂移量。
    以 ref_fn 帧的篮板 ROI 为模板，在 query_fn 帧的扩展区域内搜索，
    返回 (dx, dy)：正值表示篮板向右/下偏移。
    """
    x, y, w, h = bb_roi
    cap = cv2.VideoCapture(video_path)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh_v = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sx1 = max(0, x - search_range)
    sy1 = max(0, y - search_range)
    sx2 = min(fw, x + w + search_range)
    sy2 = min(fh_v, y + h + search_range)

    def get_gray(fn):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        return cv2.cvtColor(frame[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2GRAY) if ret else None

    ref_crop = get_gray(ref_fn)
    qry_crop = get_gray(query_fn)
    cap.release()

    if ref_crop is None or qry_crop is None:
        return 0, 0

    tx, ty = x - sx1, y - sy1
    template = ref_crop[ty:ty + h, tx:tx + w]
    if template.shape != (h, w):
        return 0, 0

    result = cv2.matchTemplate(qry_crop, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < 0.5:
        return 0, 0

    return int(max_loc[0] - tx), int(max_loc[1] - ty)


def check_basket(trajectory, bb_roi, hoop_roi, event_time, x_tol=4):
    """
    Determine if trajectory is a basket.

    Criteria:
      0. Reject static artifacts (all points within 3px radius)
      1. Ball detected above hoop (y < hoop_y_rel)
      2a. Ball crosses hoop level (above→below) AND crossing x is within full hoop
          width ± x_tol pixels, AND ball does not bounce back above hoop afterward
      2b. OR ball disappears within +0.3s of event AND last x is within hoop x-range

    Returns: (is_basket: bool, reason: str)
    """
    if len(trajectory) < 2:
        return False, "too few points (<2)"

    if is_static_trajectory(trajectory):
        return False, "static artifact (all points within 3px)"

    hoop_y_rel = hoop_roi[1] - bb_roi[1]
    hoop_x_l   = hoop_roi[0] - bb_roi[0]
    hoop_x_r   = hoop_x_l + hoop_roi[2]
    check_x_l  = hoop_x_l - x_tol      # full hoop width + tolerance on each side
    check_x_r  = hoop_x_r + x_tol

    ys = [p[2] for p in trajectory]
    xs = [p[1] for p in trajectory]
    ts = [p[0] for p in trajectory]

    above_idx = [i for i, y in enumerate(ys) if y < hoop_y_rel]
    if not above_idx:
        return False, f"no point above hoop (y<{hoop_y_rel})"

    last_above = max(above_idx)
    cross_x    = xs[last_above]   # ball x when last above hoop

    # 2a: crossed hoop level
    cross_indices = [j for j in range(last_above + 1, len(ys)) if ys[j] >= hoop_y_rel]
    if cross_indices:
        if not (check_x_l <= cross_x <= check_x_r):
            return False, f"side hit: x={cross_x} outside hoop+tol [{check_x_l},{check_x_r}]"
        # Bounce-back check: a rim deflection sends the ball back above the hoop
        first_cross = cross_indices[0]
        if any(ys[j] < hoop_y_rel for j in range(first_cross + 1, len(ys))):
            return False, f"bounce back above hoop after crossing (x={cross_x})"
        return True, f"basket (crossed hoop, x={cross_x})"

    # 2b: gone through net
    gone = ts[-1] < event_time + 0.3
    if gone:
        last_x = xs[-1]
        if not (check_x_l <= last_x <= check_x_r):
            return False, f"gone but x={last_x} outside hoop+tol [{check_x_l},{check_x_r}]"
        # Ball must be moving downward when it disappears (going through net, not bouncing up)
        if len(ys) >= 2 and ys[-1] <= ys[-2]:
            return False, f"gone but moving up at disappearance (y {ys[-2]}→{ys[-1]})"
        return True, f"basket (gone through net, x={last_x})"

    return False, "no crossing or disappearance near hoop"


def make_trajectory_image(bg_crop, trajectory, bb_roi, hoop_roi, label, scale=4):
    """在背景帧上叠加轨迹点（时间蓝→红）+ 篮筐线，返回可视化图"""
    vis = bg_crop.copy()
    hoop_y_rel = hoop_roi[1] - bb_roi[1]
    hoop_x1   = hoop_roi[0] - bb_roi[0]
    hoop_x2   = hoop_x1 + hoop_roi[2]

    # 篮筐口水平线（黄色）
    cv2.line(vis, (hoop_x1, hoop_y_rel), (hoop_x2, hoop_y_rel), (0, 220, 220), 1)

    if trajectory:
        t_min = trajectory[0][0]
        t_max = trajectory[-1][0]
        t_rng = max(t_max - t_min, 1e-6)
        prev = None
        for t, cx, cy, _ in trajectory:
            ratio = (t - t_min) / t_rng
            color = (int(255 * (1 - ratio)), 50, int(255 * ratio))   # 蓝→红
            cv2.circle(vis, (cx, cy), 4, color, -1)
            if prev:
                cv2.line(vis, prev, (cx, cy), color, 1)
            prev = (cx, cy)

    s = scale
    bh, bw = vis.shape[:2]
    big = cv2.resize(vis, (bw * s, bh * s), interpolation=cv2.INTER_NEAREST)

    is_basket = "进球" in label or "basket" in label
    color = (50, 220, 50) if is_basket else (50, 50, 220)
    pil_put_text(big, label, (6, 4), font_size=16, color=color, bg_color=(0, 0, 0))
    return big


def _get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total / fps if fps > 0 else 0.0


def _ffmpeg_clip(video_path, start_time, duration, output_path, vf=None):
    """调用 ffmpeg 剪辑指定时间段（含音频）。vf 可选视频滤镜，如裁剪。"""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("未找到 ffmpeg，请先 `brew install ffmpeg`（音频需要 ffmpeg）")

    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-ss", f"{start_time:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
    ]
    if vf:
        cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    cmd += ["-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", output_path]

    subprocess.run(cmd, check=True)


def clip_roi_video(video_path, bb_roi, event_time, fps, output_path, window=0.5):
    """将事件 ±window 秒的篮板 ROI 帧写成视频（含原音频）"""
    duration_total = _get_video_duration(video_path)
    start_time = max(0.0, event_time - window)
    end_time   = min(duration_total, event_time + window)
    if end_time <= start_time:
        return
    x, y, w, h = bb_roi
    _ffmpeg_clip(video_path, start_time, end_time - start_time, output_path,
                 vf=f"crop={w}:{h}:{x}:{y}")


def clip_full_video(video_path, event_time, output_path, pre=3.0, post=1.0):
    """将进球事件前 pre 秒、后 post 秒的完整画面写成视频（含原音频）"""
    duration_total = _get_video_duration(video_path)
    start_time = max(0.0, event_time - pre)
    end_time   = min(duration_total, event_time + post)
    if end_time <= start_time:
        return
    _ffmpeg_clip(video_path, start_time, end_time - start_time, output_path)


def _skip_frames(cap, count):
    for _ in range(count):
        if not cap.grab():
            return False
    return True


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="篮板碰球检测 + 进球识别")
    ap.add_argument("video", nargs="?", default="4-29-test.mp4")
    ap.add_argument("--backboard-roi")
    ap.add_argument("--hoop-roi")
    ap.add_argument("--interval",       type=float, default=0.1)
    ap.add_argument("--rolling-window", type=float, default=1.5)
    ap.add_argument("--threshold",      type=int,   default=25)
    ap.add_argument("--min-change",     type=int,   default=20)
    ap.add_argument("--min-coverage",   type=float, default=0.55)
    ap.add_argument("--border-margin",  type=int,   default=4)
    ap.add_argument("--cluster-gap",    type=float, default=2.0,
                    help="事件聚类间隔阈值（秒，默认 2.0）")
    ap.add_argument("--traj-window",    type=float, default=1.0,
                    help="轨迹分析窗口（事件前后各 N 秒，默认 1.0）")
    ap.add_argument("--traj-step",      type=float, default=0.05,
                    help="轨迹采样间隔（秒，默认 0.05）")
    ap.add_argument("--scale",          type=int,   default=4)
    ap.add_argument("--output-dir",     default="backboard_detections")
    ap.add_argument("--profile",        action="store_true",
                    help="输出各阶段耗时统计")
    ap.add_argument("--keep-debug-images", action="store_true",
                    help="保留事件截图和轨迹 PNG（默认只输出进球视频）")
    args = ap.parse_args()

    stem = os.path.splitext(args.video)[0]
    bb_roi_path   = args.backboard_roi or f"{stem}_backboard_roi.txt"
    hoop_roi_path = args.hoop_roi      or f"{stem}_hoop_roi.txt"

    if not os.path.exists(args.video):
        print(f"错误: 视频不存在: {args.video}"); sys.exit(1)

    if not find_ffmpeg():
        print("错误: 未找到 ffmpeg，进球剪辑需要它来保留音频。"); print("       请运行: brew install ffmpeg"); sys.exit(1)

    need_select = not os.path.exists(bb_roi_path) or not os.path.exists(hoop_roi_path)
    ref_frame = None
    if need_select:
        print("请选择一帧作为参考（拖动进度条选帧）...")
        ref_frame = select_frame(args.video)
        if ref_frame is None:
            print("已取消"); sys.exit(1)

    # ROI 文件不存在时，交互式框选
    if not os.path.exists(bb_roi_path):
        print("框选篮板范围...")
        bb_roi = select_roi(ref_frame, label="标记篮板范围")
        if bb_roi is None:
            print("已取消"); sys.exit(1)
        save_roi(bb_roi, bb_roi_path)
        print(f"篮板 ROI 已保存: {bb_roi}")
    else:
        bb_roi = load_roi(bb_roi_path)

    if not os.path.exists(hoop_roi_path):
        print("框选篮筐位置...")
        hoop_roi = select_roi(ref_frame, label="标记篮筐位置")
        if hoop_roi is None:
            print("已取消"); sys.exit(1)
        save_roi(hoop_roi, hoop_roi_path)
        print(f"篮筐 ROI 已保存: {hoop_roi}")
    else:
        hoop_roi = load_roi(hoop_roi_path)

    ref_frame = None  # 释放帧数据
    ball_diameter = hoop_roi[2] / 2
    roi_w, roi_h  = bb_roi[2], bb_roi[3]
    hoop_y_rel    = hoop_roi[1] - bb_roi[1]

    print(f"篮板 ROI      : {bb_roi}")
    print(f"篮筐 ROI      : {hoop_roi}  (篮筐口在篮板ROI内 y={hoop_y_rel})")
    print(f"球径估算      : {ball_diameter:.1f}px")
    print(f"采样间隔      : {args.interval}s  滚动窗口: {args.rolling_window}s")
    print(f"差分阈值      : {args.threshold}  覆盖率下限: {args.min_coverage:.0%}")
    print(f"聚类 gap      : {args.cluster_gap}s")
    print(f"轨迹窗口      : ±{args.traj_window}s  步长: {args.traj_step}s\n")
    print(f"调试图片      : {'保留' if args.keep_debug_images else '不保留'}\n")

    # ── 流式扫描 + 即时处理 ───────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    interval_frames = max(1, int(round(fps * args.interval)))
    buf_size = max(2, round(args.rolling_window / args.interval) + 1)

    print(f"视频: {total}帧 @ {fps:.0f}fps | 步长: {interval_frames}帧 | 缓冲: {buf_size}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    basket_dir = os.path.join(args.output_dir, "baskets")
    os.makedirs(basket_dir, exist_ok=True)
    stage_times = collections.Counter()

    def process_event(cluster, event_idx):
        """处理一个已完结的事件：事件图 → 轨迹分析 → 进球则剪辑"""
        t_stage = time.perf_counter()
        t0  = cluster[0]['time']
        t1  = cluster[-1]['time']
        rep = best_frame(cluster)
        best_cov = max(h[3] for h in rep['hits'])
        print(f"  事件 {event_idx:02d}  {t0:.1f}~{t1:.1f}s  ({len(cluster)}帧)  "
              f"最佳: {rep['time']:.1f}s  覆盖={best_cov:.0%}  Δ={rep['change_px']}px")

        if args.keep_debug_images:
            img = make_event_image(rep['crop'], rep['diff_filt'], rep['hits'], args.scale)
            pil_put_text(img,
                         f"事件{event_idx:02d} t={rep['time']:.1f}s 区间={t0:.1f}~{t1:.1f}s n={len(cluster)} 覆盖={best_cov:.0%}",
                         (6, 4), font_size=16, color=(255, 255, 255), bg_color=(0, 0, 0))
            cv2.imwrite(os.path.join(args.output_dir, f"event{event_idx:02d}_{rep['time']:.1f}s.png"), img)

        # 轨迹分析
        span_dur = t1 - t0
        event_time = rep['time'] if span_dur > 3.0 else (t0 + t1) / 2
        extra_pre = max(0.0, event_time - t0) if span_dur > 3.0 else 0.0

        traj, _ = analyze_trajectory(
            args.video, event_time, bb_roi, ball_diameter, fps,
            args.threshold, args.min_coverage, args.border_margin,
            args.traj_window, args.traj_step, extra_pre,
        )
        traj, n_removed = denoise_trajectory(traj)

        bg_fn = max(0, int(round((event_time - 2.0 - extra_pre) * fps)))
        dx, dy = compute_hoop_drift(args.video, bb_roi, fps, ref_fn=0, query_fn=bg_fn)
        cur_hoop_roi = (hoop_roi[0] + dx, hoop_roi[1] + dy, hoop_roi[2], hoop_roi[3])

        is_basket, reason = check_basket(traj, bb_roi, cur_hoop_roi, event_time)

        traj_summary = " ".join(
            f"t{p[0]:.2f}→({p[1]},{p[2]})" for p in traj
        ) or "(无轨迹点)"
        mark = "进球" if is_basket else "未进"
        span = f"{t0:.1f}~{t1:.1f}s"
        denoise_note = f"  [-{n_removed}噪声]" if n_removed else ""
        center_note  = f"(最佳帧)" if span_dur > 3.0 else "(中点)"
        drift_note   = f"  drift=({dx:+d},{dy:+d})" if (dx or dy) else ""
        print(f"\n  事件 {event_idx:02d} 区间={span} 中心={event_time:.1f}s{center_note}  轨迹={len(traj)}点{denoise_note}{drift_note}  [{mark}] {reason}")
        print(f"    {traj_summary}")

        # 轨迹可视化
        if args.keep_debug_images:
            cap_tmp = cv2.VideoCapture(args.video)
            cap_tmp.set(cv2.CAP_PROP_POS_FRAMES, bg_fn)
            ret_bg, bg_frame = cap_tmp.read()
            cap_tmp.release()
            if ret_bg:
                bg_crop = crop_roi(bg_frame, bb_roi)
                dn_note = f" 去噪-{n_removed}" if n_removed else ""
                label   = f"事件{event_idx:02d} t={event_time:.1f}s{dn_note} | {mark}: {reason}"
                traj_img = make_trajectory_image(bg_crop, traj, bb_roi, cur_hoop_roi,
                                                 label, args.scale)
                cv2.imwrite(
                    os.path.join(args.output_dir, f"event{event_idx:02d}_{event_time:.1f}s_traj.png"),
                    traj_img,
                )

        # 进球则立即剪辑
        if is_basket:
            out_path = os.path.join(basket_dir, f"basket{event_idx:02d}_t{event_time:.1f}s.mp4")
            clip_full_video(args.video, event_time, out_path, pre=3.0, post=1.0)
            print(f"  → 进球剪辑: {out_path}")
        stage_times["event"] += time.perf_counter() - t_stage

    buf = collections.deque(maxlen=buf_size)
    n_sampled = n_change = 0
    n_events = 0
    cur_cluster = []
    fn = 0
    t_scan = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fn % interval_frames == 0:
            crop = crop_roi(frame, bb_roi)
            if crop.shape[0] == roi_h and crop.shape[1] == roi_w:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                buf.append((fn, gray, crop.copy()))
                n_sampled += 1
                if len(buf) == buf_size:
                    ref_gray = buf[0][1]
                    diff = cv2.absdiff(ref_gray, gray)
                    diff_filt = diff.astype(np.float32)
                    diff_filt[diff_filt < args.threshold] = 0
                    change_px = int(np.count_nonzero(diff_filt))
                    if change_px >= args.min_change:
                        n_change += 1
                        hits = detect_ball(diff, ball_diameter, args.threshold,
                                           args.min_coverage, args.border_margin,
                                           roi_w, roi_h)
                        if hits:
                            det = {
                                'fn': fn, 'time': fn / fps,
                                'crop': crop.copy(),
                                'diff_filt': diff_filt.astype(np.uint8),
                                'hits': hits, 'change_px': change_px,
                            }
                            # 新事件：当前事件与上一个候选间隔超过阈值 → 结算上一个事件
                            if cur_cluster and (det['time'] - cur_cluster[-1]['time'] > args.cluster_gap):
                                n_events += 1
                                process_event(cur_cluster, n_events)
                                cur_cluster = []
                            cur_cluster.append(det)
        if interval_frames > 1 and fn + interval_frames < total:
            if not _skip_frames(cap, interval_frames - 1):
                break
            fn += interval_frames - 1
        fn += 1

    cap.release()
    stage_times["scan"] = time.perf_counter() - t_scan

    # 处理最后一个事件
    if cur_cluster:
        n_events += 1
        process_event(cur_cluster, n_events)

    print(f"\n扫描完成: 采样 {n_sampled} | 变化 {n_change} | 球候选 {n_change} | 事件 {n_events}")
    if args.profile:
        print(f"耗时统计: 扫描={stage_times['scan']:.2f}s  事件处理={stage_times['event']:.2f}s")


if __name__ == "__main__":
    main()
