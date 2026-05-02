#!/usr/bin/env python3
"""交互式选择ROI"""
import cv2, sys

def select(video_path, label="ROI"):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("无法读取视频")
        return None

    roi = None
    drawing = False
    x1 = y1 = x2 = y2 = 0
    display = frame.copy()

    def mouse_cb(event, x, y, flags, _):
        nonlocal drawing, x1, y1, x2, y2, display, roi
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            x1, y1 = x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            x2, y2 = x, y
            display = frame.copy()
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x2, y2 = x, y
            xn, yn = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            if w > 5 and h > 5:
                roi = (xn, yn, w, h)
            display = frame.copy()
            cv2.rectangle(display, (xn, yn), (xn+w, yn+h), (0, 255, 0), 2)

    win = f"框选{label} - Enter确认 R重绘 Q退出"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    cv2.setMouseCallback(win, mouse_cb)

    while True:
        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (13, 32) and roi:
            cv2.destroyAllWindows()
            return roi
        elif key == ord('r'):
            roi = None
            display = frame.copy()
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None

if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "test-22.mp4"
    label = sys.argv[2] if len(sys.argv) > 2 else "篮板ROI"
    roi = select(video, label)
    if roi:
        print(f"ROI: {roi[0]},{roi[1]},{roi[2]},{roi[3]}")
