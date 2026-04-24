#!/usr/bin/env python3
"""
篮球投篮自动剪辑工具

功能：
1. 每秒抽取一帧
2. 用户指定篮筐位置（矩形框）
3. 帧差法检测框内新物体
4. 验证物体从上到下穿过方框
5. 剪辑投篮瞬间前后视频片段并输出
"""

import argparse
import os
import sys
import cv2
import numpy as np


class FrameExtractor:
    """从视频中每秒提取一帧"""

    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_seconds = self.total_frames / self.fps if self.fps > 0 else 0

    def extract(self):
        """提取每秒一帧，返回 [(frame, frame_number, timestamp_sec), ...]"""
        frames = []
        for sec in range(int(self.total_seconds) + 1):
            target_frame = int(sec * self.fps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append((frame, target_frame, sec))
        return frames

    def read_frame_at(self, frame_number):
        """读取指定帧号的帧"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()


class ROISelector:
    """交互式选择篮筐区域"""

    def __init__(self, frame):
        self.frame = frame.copy()
        self.display = frame.copy()
        self.drawing = False
        self.x1 = self.y1 = self.x2 = self.y2 = 0
        self.roi = None

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.x1, self.y1 = x, y
            self.x2, self.y2 = x, y
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.x2, self.y2 = x, y
            self.display = self.frame.copy()
            cv2.rectangle(self.display, (self.x1, self.y1), (self.x2, self.y2),
                          (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.x2, self.y2 = x, y
            # 规范化坐标
            x_min = min(self.x1, self.x2)
            y_min = min(self.y1, self.y2)
            w = abs(self.x2 - self.x1)
            h = abs(self.y2 - self.y1)
            if w > 5 and h > 5:
                self.roi = (x_min, y_min, w, h)
            self.display = self.frame.copy()
            cv2.rectangle(self.display, (x_min, y_min), (x_min + w, y_min + h),
                          (0, 255, 0), 2)

    def select(self):
        """显示窗口让用户选择 ROI，返回 (x, y, w, h)"""
        win = "Draw ROI on basket - Enter to confirm, R to reset, Q to quit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
        cv2.setMouseCallback(win, self._mouse_callback)

        print("\n=== 篮筐区域选择 ===")
        print("1. 用鼠标拖拽绘制矩形框，框住篮筐")
        print("2. 按 Enter/Space 确认")
        print("3. 按 R 重新绘制")
        print("4. 按 Q 退出")

        while True:
            cv2.imshow(win, self.display)
            key = cv2.waitKey(1) & 0xFF

            if key in (13, 32):  # Enter or Space
                if self.roi is not None:
                    cv2.destroyWindow(win)
                    return self.roi
                print("请先用鼠标绘制矩形框！")
            elif key == ord('r'):
                self.roi = None
                self.display = self.frame.copy()
                print("已重置")
            elif key == ord('q'):
                cv2.destroyAllWindows()
                sys.exit(0)

        cv2.destroyAllWindows()


class ShotDetector:
    """帧差法检测投篮 + 轨迹验证"""

    def __init__(self, roi, threshold=25, min_area=200):
        self.roi = roi  # (x, y, w, h)
        self.threshold = threshold
        self.min_area = min_area
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def _compute_motion_mask(self, gray1, gray2):
        """计算两帧之间的运动掩码"""
        x, y, w, h = self.roi
        roi1 = gray1[y:y + h, x:x + w]
        roi2 = gray2[y:y + h, x:x + w]
        diff = cv2.absdiff(roi1, roi2)
        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.morph_kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.morph_kernel)
        return thresh

    def detect_candidates(self, frames):
        """检测有新物体出现的帧，返回候选帧索引列表"""
        candidates = []
        grays = [cv2.cvtColor(f[0], cv2.COLOR_BGR2GRAY) for f in frames]
        cooldown = 0

        for i in range(1, len(frames)):
            if cooldown > 0:
                cooldown -= 1
                continue

            mask = self._compute_motion_mask(grays[i - 1], grays[i])
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            has_significant_motion = any(
                cv2.contourArea(c) >= self.min_area for c in contours
            )

            if has_significant_motion:
                candidates.append(i)
                cooldown = 2  # 2秒冷却避免重复检测
                print(f"  [候选] 第 {i} 帧 (t={frames[i][2]}s) - 检测到运动")

        return candidates

    def validate_trajectory(self, candidate_idx, frames):
        """验证候选帧中物体是否从上到下穿过方框

        检查事件前后各2帧的运动轨迹：
        - 前期运动在上部（质心 y 坐标较小）
        - 后期运动在下部（质心 y 坐标较大）
        - 确认从上到下穿过
        """
        x, y, w, h = self.roi
        roi_h = h
        grays = [cv2.cvtColor(f[0], cv2.COLOR_BGR2GRAY) for f in frames]

        start = max(1, candidate_idx - 2)
        end = min(len(frames) - 1, candidate_idx + 2)

        centroids_before = []
        centroids_after = []

        for i in range(start, end + 1):
            mask = self._compute_motion_mask(grays[i - 1], grays[i])
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            # 找最大轮廓的质心
            max_area = 0
            centroid = None
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > max_area and area >= self.min_area:
                    max_area = area
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        centroid = (cx, cy)

            if centroid is not None:
                # cy 是 ROI 内的相对 y 坐标 (0=顶部, roi_h=底部)
                if i < candidate_idx:
                    centroids_before.append(centroid[1])
                elif i > candidate_idx:
                    centroids_after.append(centroid[1])

        # 没有足够的运动数据则拒绝
        if not centroids_before and not centroids_after:
            return False

        # 判断运动方向：前期在上部，后期在下部
        all_centroids = centroids_before + centroids_after
        if len(all_centroids) < 1:
            return False

        # 如果有前期数据，检查是否在上半部分
        if centroids_before:
            avg_before = np.mean(centroids_before)
            if avg_before > roi_h * 0.6:
                return False

        # 如果有后期数据，检查是否在下半部分
        if centroids_after:
            avg_after = np.mean(centroids_after)
            if avg_after < roi_h * 0.4:
                return False

        # 检查是否有从上到下的趋势
        if centroids_before and centroids_after:
            if np.mean(centroids_after) <= np.mean(centroids_before):
                return False

        print(f"  [确认] 第 {candidate_idx} 帧 - 从上到下穿过篮筐")
        return True


class VideoClipper:
    """剪辑并保存视频片段"""

    def __init__(self, video_path):
        self.video_path = video_path

    def clip(self, event_frame_number, fps, before_sec=5, after_sec=1,
             output_dir="output", index=0):
        """剪辑指定帧前后的视频片段"""
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_frame = max(0, event_frame_number - int(before_sec * src_fps))
        end_frame = min(total - 1, event_frame_number + int(after_sec * src_fps))

        event_sec = event_frame_number / src_fps
        start_sec = start_frame / src_fps
        end_sec = end_frame / src_fps

        print(f"\n  剪辑片段 {index + 1}:")
        print(f"    事件时间: {event_sec:.1f}s")
        print(f"    范围: {start_sec:.1f}s - {end_sec:.1f}s"
              f" ({end_sec - start_sec:.1f}s)")
        print(f"    帧: {start_frame} - {end_frame}")

        output_path = os.path.join(
            output_dir,
            f"shot_{index + 1:03d}_{event_sec:.1f}s.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, src_fps, (src_w, src_h))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for fn in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)

        writer.release()
        cap.release()
        print(f"    输出: {output_path}")
        return output_path


def save_roi(roi, path):
    """保存 ROI 到文件"""
    with open(path, 'w') as f:
        f.write(f"{roi[0]},{roi[1]},{roi[2]},{roi[3]}\n")
    print(f"ROI 已保存到: {path}")


def load_roi(path):
    """从文件加载 ROI"""
    with open(path, 'r') as f:
        parts = f.read().strip().split(',')
        roi = tuple(int(p) for p in parts)
    print(f"从 {path} 加载 ROI: {roi}")
    return roi


def parse_roi(roi_str):
    """解析 ROI 字符串 'x,y,w,h'"""
    parts = roi_str.split(',')
    return tuple(int(p) for p in parts)


def main():
    parser = argparse.ArgumentParser(description="篮球投篮自动剪辑工具")
    parser.add_argument("input", help="输入视频路径")
    parser.add_argument("--roi", help="篮筐 ROI: x,y,w,h")
    parser.add_argument("--roi-file", help="从文件读取 ROI")
    parser.add_argument("--threshold", type=int, default=25,
                        help="帧差二值化阈值 (默认 25)")
    parser.add_argument("--min-area", type=int, default=200,
                        help="最小轮廓面积 (默认 200)")
    parser.add_argument("--clip-before", type=float, default=5.0,
                        help="剪辑事件前秒数 (默认 5)")
    parser.add_argument("--clip-after", type=float, default=1.0,
                        help="剪辑事件后秒数 (默认 1)")
    parser.add_argument("--output-dir", default="output",
                        help="输出目录 (默认 output)")
    args = parser.parse_args()

    # 加载视频
    print(f"加载视频: {args.input}")
    extractor = FrameExtractor(args.input)
    print(f"  帧率: {extractor.fps:.1f} FPS")
    print(f"  总帧数: {extractor.total_frames}")
    print(f"  时长: {extractor.total_seconds:.1f}s")

    # 提取帧
    print(f"\n每秒提取一帧...")
    frames = extractor.extract()
    print(f"  提取了 {len(frames)} 帧")

    if not frames:
        print("错误：无法从视频中提取帧")
        sys.exit(1)

    # 获取 ROI
    if args.roi:
        roi = parse_roi(args.roi)
    elif args.roi_file:
        if not os.path.exists(args.roi_file):
            print(f"错误：ROI 文件不存在: {args.roi_file}")
            sys.exit(1)
        roi = load_roi(args.roi_file)
    else:
        selector = ROISelector(frames[0][0])
        roi = selector.select()
        # 自动保存 ROI 供下次使用
        roi_path = os.path.splitext(args.input)[0] + "_roi.txt"
        save_roi(roi, roi_path)

    print(f"\n篮筐 ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")

    # 检测
    print(f"\n开始检测投篮... (阈值={args.threshold}, 最小面积={args.min_area})")
    detector = ShotDetector(roi, args.threshold, args.min_area)
    candidates = detector.detect_candidates(frames)
    print(f"\n发现 {len(candidates)} 个候选事件")

    # 验证轨迹
    valid_events = []
    for idx in candidates:
        if detector.validate_trajectory(idx, frames):
            valid_events.append(frames[idx])

    print(f"\n确认 {len(valid_events)} 个投篮事件")

    if not valid_events:
        print("未检测到投篮事件，尝试降低 --threshold 或 --min-area 参数")
        extractor.release()
        sys.exit(0)

    # 剪辑输出
    print(f"\n开始剪辑 (前{args.clip_before}s + 后{args.clip_after}s)...")
    clipper = VideoClipper(args.input)
    for i, (frame, frame_num, ts) in enumerate(valid_events):
        clipper.clip(frame_num, extractor.fps, args.clip_before, args.clip_after,
                     args.output_dir, i)

    print(f"\n完成！共输出 {len(valid_events)} 个片段到 {args.output_dir}/")
    extractor.release()


if __name__ == "__main__":
    main()
