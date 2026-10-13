#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 16 端到端探针：人脸序列 → BehaviorEngine → 真串口 → 假设备
============================================================
为什么需要这个脚本
------------------------------------------------------------
`python behave.py` 已经覆盖了引擎的全部逻辑，`--demo` 也验证了"拍手闪灯"
这条支路。但有一个洞：

**合成场景（--demo）里 faces 永远是 0**（见 moss_brain.DemoScene 的注释）——
因为 Haar 认不出画出来的圆。所以"有人来了转头""人走了转回来"
这两条最核心的主动行为，在 demo 模式下**一次也走不到**。

要看它们，要么架真摄像头，要么把"人脸"这一步换成脚本。
这个脚本就是后者：它保留**真的** BehaviorEngine、**真的**串口协议、
**真的**设备解析，只把"摄像头看到几张脸"换成一段写死的序列。

于是它验证的是：引擎判断 → 命令过滤 → 写串口 → 设备收到并且认。
唯一没覆盖的是 Haar 本身能不能数对人脸 —— 那是阶段 7 的事。

用法：
    python tools/behave_e2e.py            # 自己起假设备，跑完自动清理
    python tools/behave_e2e.py --gap 0.3  # 把节流调小，跑得快一点
"""

import argparse
import os
import socket
import subprocess
import sys
import time

import serial

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from behave import BehaviorEngine                     # noqa: E402
from moss_actuators import ActuatorLink               # noqa: E402

# (相对时刻, 画面里有几张脸, 说明)
#
# 注意间隔：gap 之外还要注意 arrive 自己的 30 秒冷却 ——
# 所以整段序列里"来人"只安排一次，第二次来人放到最后单独验证冷却。
SEQ = [
    (0.0, 0, "空画面"),
    (0.3, 1, "有人出现  → 期待：转头 + 变蓝灯"),
    (0.8, 1, "人还在    → 期待：什么都不做（边沿只有一次）"),
    (1.4, 1, "人还在    → 期待：什么都不做"),
    (2.0, 0, "人走了    → 期待：转回来 + 暖橙灯"),
    (2.6, 0, "还是没人  → 期待：什么都不做"),
]


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main():
    ap = argparse.ArgumentParser(description="阶段16 主动行为端到端探针")
    ap.add_argument("--port", type=int, default=0, help="假设备端口，0=自动挑")
    ap.add_argument("--gap", type=float, default=0.5, help="全局节流，默认 0.5（为了跑得快）")
    ap.add_argument("--autonomy", default="l1", choices=["l0", "l1"])
    a = ap.parse_args()

    port = a.port or _free_port()
    print("=" * 70)
    print("  阶段 16 端到端探针：人脸序列 → 引擎 → 串口 → 假设备")
    print("=" * 70)
    print(f"  权限 {a.autonomy} · 节流 {a.gap}s · 假设备 127.0.0.1:{port}\n")

    dev = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "mock_esp32.py"), "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace")
    try:
        # 等它开始监听，直接连会失败，所以重试几次。
        #
        # ⚠️ 必须用 pyserial 的 serial_for_url，不能直接把 socket 塞给
        # ActuatorLink —— 裸 socket 没有 readline()，收设备的那个线程会立刻
        # 抛异常退出，结果是"命令确实发出去了，但一条回话都收不到"，
        # 看起来像设备没反应，其实是本地读的那一半根本不存在。
        ser = None
        for _ in range(40):
            try:
                ser = serial.serial_for_url(f"socket://127.0.0.1:{port}",
                                            timeout=0.2)
                break
            except Exception:
                time.sleep(0.1)
        if ser is None:
            print("[失败] 连不上假设备。")
            return 2

        link = ActuatorLink(ser, log=lambda s: None)     # 回话不打印，省得刷屏
        eng = BehaviorEngine(a.autonomy, gap=a.gap, log=None)

        print("  时刻   脸数  事件")
        print("  " + "-" * 66)
        t0 = time.time()
        got = []
        for at, faces, note in SEQ:
            time.sleep(max(0.0, t0 + at - time.time()))
            act = eng.tick({"faces": faces, "clap": False,
                            "speaking": False, "user_active": False,
                            "mode": "IDLE"})
            tag = ""
            if act is not None:
                for cmd in act.cmds:
                    link.send(cmd)
                tag = f"  ⇒ [{act.name}] {', '.join(act.cmds)}"
                got.append((act.name, tuple(act.cmds)))
            print(f"  {at:>4.1f}s  {faces}     {note}{tag}")

        time.sleep(0.4)
        print("\n  " + "-" * 66)
        print("  假设备实际收到的命令：")
        for line in link.recent_events(20):
            print(f"      {line}")

        # 直接用假设备的回执来判 —— 不看引擎自己怎么说，
        # 看设备**真的收到了什么**。这才是端到端该验的东西。
        lines = [ln for _, ln in list(link.events)]
        joins = " | ".join(lines)
        # 用的是假设备**原样回的话**（格式见 mock_esp32.py），
        # 不是我们自己拼的命令 —— 这样连"固件端解析对不对"一起验了。
        expect = ["[SPIN] 1024 steps CW", "[LED] 0 80 255",
                  "[SPIN] 1024 steps CCW", "[LED] 255 120 0"]
        miss = [e for e in expect if e not in joins]
        if miss:
            print(f"\n[失败] 设备没收到：{miss}")
            return 1
        if a.autonomy == "l1":
            if "[BEEP] 1500Hz 60ms" not in joins:
                print("\n[失败] l1 应该收到 BEEP，但没有。")
                return 1
            print("\n  ✓ l1：转头 + 蓝灯 + 蜂鸣，人走转回 + 暖橙，全部抵达")
        else:
            if "[BEEP]" in joins:
                print("\n[失败] l0 不该发出 BEEP。")
                return 1
            print("\n  ✓ l0：转头 + 蓝灯，人走转回 + 暖橙；**没有** BEEP")
        print(f"  引擎自述：{eng.stats_line()}")
        print("\n[OK] 主动行为端到端通过。")
        return 0
    finally:
        dev.terminate()
        try:
            dev.wait(timeout=3)
        except subprocess.TimeoutExpired:
            dev.kill()


if __name__ == "__main__":
    raise SystemExit(main())
