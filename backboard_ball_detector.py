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
import sys
import cv2
import numpy as np


def load_roi(path):
    with open(path) as f:
        return tuple(int(p) for p in f.read().strip().split(','))


def crop_roi(frame, roi):
    x, y, w, h = roi
    fh, fw = frame.shape[:2]
    return frame[max(0, y):min(fh, y + h), max(0, x):min(fw, x + w)]


def detect_ball(diff_raw, ball_diameter, noise_thresh, min_coverage, border_margin, roi_w, roi_h):
    """差分图中检测球体：面积 / 圆形度 / 边缘距离 / 差分覆盖率四重过滤

    Returns: list of (cx, cy, radius, coverage)
    """
    _, thresh = cv2.threshold(diff_raw, noise_thresh, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
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

        mask = np.zeros(diff_raw.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)
        inside_total = cv2.countNonZero(mask)
        if inside_total == 0:
            continue
        active = int(np.sum(diff_raw[mask > 0] >= noise_thresh))
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
    while fn <= end_fn:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
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

    is_basket = "basket" in label
    color = (50, 220, 50) if is_basket else (50, 50, 220)
    cv2.putText(big, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA)
    return big


def clip_roi_video(video_path, bb_roi, event_time, fps, output_path, window=0.5):
    """将事件 ±window 秒的篮板 ROI 帧写成视频"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    roi_w, roi_h = bb_roi[2], bb_roi[3]

    start_fn = max(0, int(round((event_time - window) * src_fps)))
    end_fn   = min(total - 1, int(round((event_time + window) * src_fps)))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, src_fps, (roi_w, roi_h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_fn)
    for _ in range(end_fn - start_fn + 1):
        ret, frame = cap.read()
        if not ret:
            break
        crop = crop_roi(frame, bb_roi)
        if crop.shape[:2] == (roi_h, roi_w):
            writer.write(crop)

    writer.release()
    cap.release()


def clip_full_video(video_path, event_time, output_path, pre=3.0, post=1.0):
    """将进球事件前 pre 秒、后 post 秒的完整画面写成视频"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_fn = max(0, int(round((event_time - pre) * src_fps)))
    end_fn   = min(total - 1, int(round((event_time + post) * src_fps)))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, src_fps, (fw, fh))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_fn)
    for _ in range(end_fn - start_fn + 1):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    writer.release()
    cap.release()


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
    args = ap.parse_args()

    stem = os.path.splitext(args.video)[0]
    bb_roi_path   = args.backboard_roi or f"{stem}_backboard_roi.txt"
    hoop_roi_path = args.hoop_roi      or f"{stem}_hoop_roi.txt"

    for p in [args.video, bb_roi_path, hoop_roi_path]:
        if not os.path.exists(p):
            print(f"错误: 不存在: {p}"); sys.exit(1)

    bb_roi   = load_roi(bb_roi_path)
    hoop_roi = load_roi(hoop_roi_path)
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

    # ── Phase 1: 扫描检测 ────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    interval_frames = max(1, int(round(fps * args.interval)))
    buf_size = max(2, round(args.rolling_window / args.interval) + 1)

    print(f"视频: {total}帧 @ {fps:.0f}fps | 步长: {interval_frames}帧 | 缓冲: {buf_size}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    buf = collections.deque(maxlen=buf_size)
    n_sampled = n_change = 0
    all_detections = []
    fn = 0

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
                    ref_fn, ref_gray, _ = buf[0]
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
                            all_detections.append({
                                'fn': fn, 'time': fn / fps,
                                'crop': crop.copy(),
                                'diff_filt': diff_filt.astype(np.uint8),
                                'hits': hits, 'change_px': change_px,
                            })
        fn += 1

    cap.release()
    print(f"扫描完成: 采样 {n_sampled} | 变化 {n_change} | 球候选 {len(all_detections)}")

    # ── Phase 2: 聚类 ────────────────────────────────────────────
    events = cluster_detections(all_detections, args.cluster_gap)
    print(f"聚类（gap≤{args.cluster_gap}s）→ {len(events)} 个事件\n")

    for i, event in enumerate(events, 1):
        t0  = event[0]['time']
        t1  = event[-1]['time']
        rep = best_frame(event)
        best_cov = max(h[3] for h in rep['hits'])
        print(f"  事件 {i:02d}  {t0:.1f}~{t1:.1f}s  ({len(event)}帧)  "
              f"最佳: {rep['time']:.1f}s  cov={best_cov:.0%}  Δ={rep['change_px']}px")

        img = make_event_image(rep['crop'], rep['diff_filt'], rep['hits'], args.scale)
        cv2.putText(img,
                    f"ev{i:02d} t={rep['time']:.1f}s span={t0:.1f}~{t1:.1f}s n={len(event)} cov={best_cov:.0%}",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(args.output_dir, f"event{i:02d}_{rep['time']:.1f}s.png"), img)

    # ── Phase 3: 轨迹分析 + 进球识别 ────────────────────────────
    print(f"\n── 轨迹分析（±{args.traj_window}s @ {args.traj_step}s步）────────────────")
    basket_dir = os.path.join(args.output_dir, "baskets")
    os.makedirs(basket_dir, exist_ok=True)

    basket_candidates = []

    for i, event in enumerate(events, 1):
        rep = best_frame(event)
        span_dur = event[-1]['time'] - event[0]['time']
        event_time = rep['time'] if span_dur > 3.0 else (event[0]['time'] + event[-1]['time']) / 2

        extra_pre = max(0.0, event_time - event[0]['time']) if span_dur > 3.0 else 0.0
        traj, _ = analyze_trajectory(
            args.video, event_time, bb_roi, ball_diameter, fps,
            args.threshold, args.min_coverage, args.border_margin,
            args.traj_window, args.traj_step, extra_pre,
        )

        traj, n_removed = denoise_trajectory(traj)

        # 镜头漂移修正：用背景帧与视频第一帧做模板匹配，动态调整篮筐坐标
        bg_fn = max(0, int(round((event_time - 2.0 - extra_pre) * fps)))
        dx, dy = compute_hoop_drift(args.video, bb_roi, fps, ref_fn=0, query_fn=bg_fn)
        cur_hoop_roi = (hoop_roi[0] + dx, hoop_roi[1] + dy, hoop_roi[2], hoop_roi[3])

        is_basket, reason = check_basket(traj, bb_roi, cur_hoop_roi, event_time)

        traj_summary = " ".join(
            f"t{p[0]:.2f}→({p[1]},{p[2]})" for p in traj
        ) or "(no points)"
        mark = "BASKET" if is_basket else "miss"
        span = f"{event[0]['time']:.1f}~{event[-1]['time']:.1f}s"
        denoise_note = f"  [-{n_removed}noise]" if n_removed else ""
        center_note  = f"(best)" if span_dur > 3.0 else "(mid)"
        drift_note   = f"  drift=({dx:+d},{dy:+d})" if (dx or dy) else ""
        print(f"\n  event {i:02d} span={span} center={event_time:.1f}s{center_note}  traj={len(traj)}pts{denoise_note}{drift_note}  [{mark}] {reason}")
        print(f"    {traj_summary}")

        # trajectory overlay image（黄线显示漂移修正后的篮筐位置）
        cap_tmp = cv2.VideoCapture(args.video)
        cap_tmp.set(cv2.CAP_PROP_POS_FRAMES, bg_fn)
        ret_bg, bg_frame = cap_tmp.read()
        cap_tmp.release()
        if ret_bg:
            bg_crop = crop_roi(bg_frame, bb_roi)
            dn_note = f" dn-{n_removed}" if n_removed else ""
            label   = f"ev{i:02d} t={event_time:.1f}s{dn_note} | {mark}: {reason}"
            traj_img = make_trajectory_image(bg_crop, traj, bb_roi, cur_hoop_roi,
                                             label, args.scale)
            cv2.imwrite(
                os.path.join(args.output_dir, f"event{i:02d}_{event_time:.1f}s_traj.png"),
                traj_img,
            )

        if is_basket:
            ys_t = [p[2] for p in traj]
            y_span = (max(ys_t) - min(ys_t)) if ys_t else 0
            basket_candidates.append({
                'event_idx': i,
                'event_time': event_time,
                'conf': len(traj) * y_span,   # more pts × larger arc = higher confidence
                'reason': reason,
            })

    basket_candidates.sort(key=lambda x: x['event_time'])
    for j, b in enumerate(basket_candidates, 1):
        out_path = os.path.join(basket_dir, f"basket{j:02d}_t{b['event_time']:.1f}s.mp4")
        clip_full_video(args.video, b['event_time'], out_path, pre=3.0, post=1.0)
        print(f"  basket{j:02d}  ev{b['event_idx']:02d}  t={b['event_time']:.1f}s  → {out_path}")

    print(f"\nDetected {len(basket_candidates)}/{len(events)} baskets  |  clips: {basket_dir}/")


if __name__ == "__main__":
    main()
