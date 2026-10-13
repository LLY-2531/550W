#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOSS 视觉大脑  ·  流浪地球 550W 复刻项目 · 阶段 7
============================================================
它做的事：
    1. 打开电脑摄像头，逐帧读取画面
    2. 用 OpenCV 判断「画面里有没有人脸」和「画面有没有在动」
    3. 把结果整理成一行 JSON 输出到终端（方便以后接大模型/存日志）
    4. （可选）通过串口把 MODE / TEXT 命令推给 ESP32，让 MOSS 转头看你

四种画面来源（--source）：
    0              默认摄像头（也可以写 1、2 ... 换第 N 个）
    demo           合成画面（没有摄像头也能跑，用来验证整条链路）
    xxx.mp4        视频文件
    xxx.jpg        图片文件（循环显示，适合用来测人脸识别）

典型用法：
    python moss_vision.py                          # 摄像头 + 预览窗
    python moss_vision.py --source demo            # 没摄像头时先试这个
    python moss_vision.py --source face.jpg        # 拿一张有人脸的照片测识别
    python moss_vision.py --port COM3              # 同时把命令发给真机
    python moss_vision.py --port rfc2217://localhost:4000   # 发给 Wokwi 仿真
    python moss_vision.py --no-window --frames 120 # 不开窗、跑 120 帧就退出

输出示例：
    {"t": 12.34, "faces": 1, "motion": 0.073, "mode": "VISION", "text": "face x1", "sent": true}
"""

import argparse
import json
import os
import sys
import time

# ---------------------------------------------------------------- 依赖检查
# 新手最常见的报错就是"没装库"。这里主动检查，并直接告诉你怎么装，
# 比让 Python 抛一堆红字友好得多。
try:
    import cv2
except ImportError:
    sys.exit("[缺少依赖] 没装 OpenCV。请在终端执行：pip install opencv-python")

try:
    import numpy as np
except ImportError:
    sys.exit("[缺少依赖] 没装 NumPy。请在终端执行：pip install numpy")

try:
    import serial  # 来自 pyserial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


# ================================================================
#  1. 命令行参数
# ================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="MOSS 视觉大脑 —— 读摄像头，判断人脸/运动，输出 JSON，可选推送串口"
    )
    p.add_argument("--source", default="0",
                   help="画面来源：0=摄像头 / demo=合成画面 / 视频或图片文件路径")
    p.add_argument("--port", default=None,
                   help="串口。COM3=真机；rfc2217://localhost:4000=Wokwi 仿真；不给就不发")
    p.add_argument("--baud", type=int, default=115200, help="串口波特率，默认 115200")
    p.add_argument("--no-window", action="store_true", help="不弹预览窗口（纯输出 JSON）")
    p.add_argument("--frames", type=int, default=0, help="跑多少帧后退出，0=一直跑")
    p.add_argument("--width", type=int, default=640, help="采集宽度")
    p.add_argument("--height", type=int, default=480, help="采集高度")
    p.add_argument("--motion-th", type=float, default=0.008,
                   help="运动判定阈值 0~1，默认 0.008（画面变化面积占比）")
    p.add_argument("--fps", type=float, default=30.0, help="最高处理帧率，0=不限速")
    p.add_argument("--stable", type=int, default=3,
                   help="连续 N 帧结果一致才切状态，默认 3（防抖动）")
    p.add_argument("--heartbeat", type=float, default=2.0,
                   help="状态没变时，每隔几秒也输出一行 JSON，默认 2 秒")
    p.add_argument("--verbose", action="store_true", help="每一帧都输出 JSON")
    return p.parse_args()


# ================================================================
#  2. 画面来源
# ================================================================
class DemoScene:
    """没有摄像头时的替身：画一个会动的小球，让整条链路能跑起来。

    它会「动 2 秒、停 1 秒」地循环，这样你能看到状态在 MOTION / IDLE 之间跳。
    （合成画面里没有人脸，所以 faces 永远是 0 —— 人脸那条路要用真摄像头测。）
    """

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.i = 0
        self.x, self.y = w // 2, h // 2

    def read(self):
        self.i += 1
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        img[:] = (28, 18, 10)                                  # 深蓝底（BGR）

        phase = self.i % 160
        if phase < 100:                                        # 前 100 帧：动
            self.x = int(self.w / 2 + (self.w / 3.0) * np.sin(self.i / 12.0))
            self.y = int(self.h / 2 + (self.h / 4.0) * np.sin(self.i / 9.0))
            tag = "MOVING"
        else:                                                  # 后 60 帧：停
            tag = "STILL"

        cv2.putText(img, "MOSS VISION  [DEMO]", (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (210, 230, 255), 2)
        cv2.putText(img, tag, (20, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
        cv2.circle(img, (self.x, self.y), 70, (255, 160, 60), -1)
        return True, img

    def release(self):
        pass


class FileScene:
    """视频文件 / 图片文件。图片会被反复复制成帧流。"""

    def __init__(self, path, w, h):
        self.is_image = path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        self.cap = None
        self.img = None
        if self.is_image:
            self.img = cv2.imread(path)
            if self.img is None:
                raise FileNotFoundError(path)
            self.img = cv2.resize(self.img, (w, h))
        else:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise FileNotFoundError(path)

    def read(self):
        if self.is_image:
            return True, self.img.copy()
        ok, frame = self.cap.read()
        if not ok:                       # 视频放完了 → 倒回开头，循环播
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return ok, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()


class CameraScene:
    """真实摄像头。"""

    def __init__(self, index, w, h):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == "nt" else 0)
        if not self.cap.isOpened():      # 有些机器用 CAP_DSHOW 打不开，退回默认后端
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"打不开摄像头 {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 只留 1 帧，避免画面延迟

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


def make_reader(args):
    src = args.source
    if src == "demo":
        print("[OK] 使用合成画面（demo 模式）", file=sys.stderr)
        return DemoScene(args.width, args.height)
    if src.isdigit():
        return CameraScene(int(src), args.width, args.height)
    return FileScene(src, args.width, args.height)


# ================================================================
#  3. 串口
# ================================================================
def open_serial(spec, baud):
    """连不上串口不算错误 —— 只输出 JSON 也能用，所以这里不 raise。"""
    if not HAS_SERIAL:
        print("[提示] 没装 pyserial，串口功能关闭。需要就执行：pip install pyserial",
              file=sys.stderr)
        return None
    try:
        if "://" in spec:                                   # 形如 rfc2217://localhost:4000
            ser = serial.serial_for_url(spec, baudrate=baud, timeout=0.2)
        else:                                               # 形如 COM3 / /dev/ttyUSB0
            ser = serial.Serial(spec, baud, timeout=0.2)
        time.sleep(0.3)
        print(f"[OK] 串口已连接：{spec}", file=sys.stderr)
        return ser
    except Exception as e:
        print(f"[提示] 串口连接失败（{spec}）：{e}", file=sys.stderr)
        print("       → 继续只输出 JSON，视觉识别不受影响。", file=sys.stderr)
        return None


def send_state(ser, mode, text):
    """把状态推给 ESP32。协议就是 firmware 里那几个串口命令。"""
    if ser is None:
        return False
    try:
        ser.write(f"MODE {mode}\n".encode("ascii", "ignore"))
        ser.write(f"TEXT {text}\n".encode("ascii", "ignore"))
        ser.flush()
        return True
    except Exception as e:
        print(f"[提示] 串口写入失败：{e}", file=sys.stderr)
        return False


# ================================================================
#  4. 主流程
# ================================================================
def load_face_cascade():
    path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    if not os.path.exists(path):
        print("[提示] 找不到人脸模型，本次只做运动检测。", file=sys.stderr)
        return None
    return cv2.CascadeClassifier(path)


def decide(n_faces, motion, motion_th):
    """业务规则：有脸 → 进入识物；画面在动 → 进入运动；都安静 → 待机。"""
    if n_faces > 0:
        return "VISION", f"face x{n_faces}"
    if motion > motion_th:
        return "MOTION", "moving"
    return "IDLE", "clear"


def main():
    args = parse_args()

    # ---- 打开画面来源 ----
    try:
        reader = make_reader(args)
    except Exception as e:
        print(f"[错误] 打不开画面来源 '{args.source}'：{e}", file=sys.stderr)
        print("       没有摄像头的话，先用：python moss_vision.py --source demo",
              file=sys.stderr)
        return 2

    cascade = load_face_cascade()
    ser = open_serial(args.port, args.baud) if args.port else None

    print("[OK] MOSS 视觉大脑启动。按 q 或 Ctrl+C 退出。", file=sys.stderr)
    print("-" * 62, file=sys.stderr)

    prev_gray = None
    cur_mode = None
    cur_text = ""
    pending = None
    stable_n = 0
    last_json = 0.0
    frame_i = 0
    t0 = time.time()
    window_ok = not args.no_window

    try:
        while True:
            iter_t0 = time.time()
            ok, frame = reader.read()
            if not ok or frame is None:
                print("[提示] 读不到画面，结束。", file=sys.stderr)
                break

            frame_i += 1

            # ---------- 灰度 + 降噪（人脸和运动都在灰度图上做）----------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            # ---------- 运动检测：和上一帧做差，统计"变了多少像素" ----------
            if prev_gray is None:
                motion = 0.0
            else:
                diff = cv2.absdiff(prev_gray, gray)
                _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                motion = float(np.count_nonzero(th)) / float(th.size)
            prev_gray = gray

            # ---------- 人脸检测 ----------
            faces = []
            if cascade is not None:
                faces = cascade.detectMultiScale(
                    gray, scaleFactor=1.15, minNeighbors=5, minSize=(48, 48)
                )

            # ---------- 决策 + 防抖动 ----------
            mode, text = decide(len(faces), motion, args.motion_th)
            if mode == pending:
                stable_n += 1
            else:
                pending, stable_n = mode, 1

            changed = False
            if stable_n >= args.stable and mode != cur_mode:
                cur_mode, cur_text, changed = mode, text, True

            # ---------- 输出 ----------
            now = time.time()
            if changed or args.verbose or (now - last_json) >= args.heartbeat:
                sent = send_state(ser, cur_mode, cur_text) if changed else False
                rec = {
                    "t": round(now - t0, 2),
                    "faces": int(len(faces)),
                    "motion": round(motion, 4),
                    "mode": cur_mode,
                    "text": cur_text,
                    "sent": bool(sent),
                }
                print(json.dumps(rec, ensure_ascii=False), flush=True)
                last_json = now

            # ---------- 预览窗 ----------
            if window_ok:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 170), 2)
                cv2.putText(frame, f"{cur_mode}  {cur_text}", (14, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
                cv2.putText(frame, f"faces {len(faces)}   motion {motion * 100:4.1f}%",
                            (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                cv2.putText(frame, "press q to quit", (14, frame.shape[0] - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                try:
                    cv2.imshow("MOSS Vision", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error as e:
                    # 没有图形界面（比如远程/无桌面环境）→ 自动关掉窗口，继续跑
                    print(f"[提示] 开不了预览窗，转为无窗口模式：{e}", file=sys.stderr)
                    window_ok = False

            # ---------- 帧率限速（让预览窗像视频，而不是刷屏）----------
            if args.fps > 0:
                dt = time.time() - iter_t0
                if dt < 1.0 / args.fps:
                    time.sleep(1.0 / args.fps - dt)

            # ---------- 帧数上限（自测用）----------
            if args.frames and frame_i >= args.frames:
                break

    except KeyboardInterrupt:
        print("\n[提示] 收到 Ctrl+C，退出。", file=sys.stderr)
    finally:
        reader.release()
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    print(f"[OK] 结束。共处理 {frame_i} 帧。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
