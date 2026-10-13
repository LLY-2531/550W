#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
假 ESP32 设备（离线验证工具）  ·  流浪地球 550W 复刻项目 · 阶段 14
============================================================
作用：在 PC 上扮演一台**跑着本项目固件的 ESP32**，
让"执行器工具"这条链路**不需要真板子、不需要 Wokwi** 就能验证。

它按 `src/main.cpp` 里 handleCommand() 的**同一套规则**解析串口命令：
    MODE / TEXT / DEMO / LED / SPIN / BEEP / STOP / PING / HELP / DUMP
并且回**同样的那几行**（[STATE]/[CMD]/[LED]/[SPIN]/[PONG]）。

为什么值得写这么个东西
------------------------------------------------------------
因为"发出去一条 LED 255 0 0"和"设备真的变红了"是两件事。
光看 PC 端日志，永远不知道指令是不是被设备**接受**了、参数有没有被**钳制**、
未知命令有没有被**兜住**。真板子又不好一边跑一边断言。

所以这里刻意做成"规则等价"：
- LED 值同样钳到 0..255、ms 钳到上限
- SPIN 同样有 200000 步上限、方向解析同样大小写都认
- 未知命令同样回 help（而不是默默吞掉）
- 多了一个固件没有的 DUMP：把内部状态吐出来，方便测试断言

------------------------------------------------------------
怎么接进 moss_brain
------------------------------------------------------------
Windows 上开虚拟串口很麻烦，但 pyserial 支持 `socket://` 伪 URL，
所以这个假设备直接监听 TCP，PC 端当成串口连就行：

    # 终端 1
    python tools/mock_esp32.py --port 4000

    # 终端 2（注意 --port 用的是 socket:// 而不是 COMx）
    python moss_brain.py --demo --port socket://localhost:4000 \
        --llm api --api-base http://localhost:8000/v1 --api-key fake --model mock-1

Wokwi 的 rfc2217 用的是 4000 端口，这里默认也开 4000 —— 但同一个端口
不能同时被 Wokwi 和这个假设备占用，二选一。

自检：
    python tools/mock_esp32.py --selftest
"""

import argparse
import socketserver
import sys
import threading
import time

# 与 include/moss_state.h 完全一致
STATES = ("BOOT", "IDLE", "LISTEN", "THINK",
          "SPEAK", "VISION", "MOTION", "ERROR")

HELP_LINES = [
    "---- commands ----",
    "MODE IDLE|LISTEN|THINK|SPEAK|VISION|MOTION|ERROR",
    "TEXT <text>       OLED third line",
    "DEMO ON|OFF       auto cycle states",
    "-- actuators (stage 14) --",
    "LED  <r> <g> <b> [ms]   ring solid color, auto-restore",
    "SPIN <steps> [CW|CCW]   turn motor N steps (4096 ~= 1 turn)",
    "BEEP <freq> <ms>        buzzer tone",
    "STOP                    stop motor + buzzer + light override",
    "PING                    -> [PONG] <state>",
    "HELP              show this",
    "------------------",
]


class DeviceState:
    """固件里那几个全局变量的等价物。"""

    def __init__(self):
        self.lock = threading.RLock()
        self.mode = "BOOT"
        self.line1 = ""
        self.demo = False
        # 灯环接管
        self.led = (0, 0, 0)
        self.led_until = 0.0
        self.led_on = False
        # 电机
        self.queue = 0
        self.queue_dir = 1
        self.pos = 0
        # 统计（只在假设备里有，用来做断言）
        self.received = []          # 收到的原始命令
        self.unknown = []           # 被兜住的未知命令
        self.rejected = []          # 参数不合法被拒的

    # ---------- 工具 ----------
    def led_expire(self):
        """到点归还（对应固件里的 lightTick）。返回是否刚刚归还。"""
        with self.lock:
            if self.led_on and time.time() >= self.led_until:
                self.led_on = False
                return True
        return False

    def snapshot(self):
        with self.lock:
            return {
                "mode": self.mode, "line1": self.line1, "demo": self.demo,
                "led": self.led, "led_on": self.led_on,
                "led_left_ms": (max(0, int((self.led_until - time.time()) * 1000))
                                if self.led_on else 0),
                "queue": self.queue, "queue_dir": self.queue_dir, "pos": self.pos,
                "received": len(self.received), "unknown": len(self.unknown),
                "rejected": len(self.rejected),
            }


STATE = DeviceState()


def _parse_ints(s, maxn):
    """等价于固件里的 parseInts()：按空格/逗号切，遇非数字停。"""
    out = []
    i, n = 0, len(s)
    while i < n and len(out) < maxn:
        while i < n and s[i] in " ,;":
            i += 1
        if i >= n:
            break
        j = i
        if s[j] in "+-":
            j += 1
        k = j
        while k < n and s[k].isdigit():
            k += 1
        if k == j:
            break                      # 一个数字都没前进 → 停（防死循环）
        out.append(int(s[i:k]))
        i = k
    return out


def handle_command(raw):
    """解析一条命令，返回要回给 PC 的行（list[str]）。

    规则与 src/main.cpp 的 handleCommand() 一一对应。
    """
    cmd, _, arg = raw.partition(" ")
    arg = arg.strip() or None
    cmd = cmd.strip().upper()

    with STATE.lock:
        STATE.received.append(raw)

    # ---------- MODE ----------
    if cmd == "MODE" and arg:
        m = arg.upper()
        if m in STATES:
            with STATE.lock:
                STATE.mode = m
            return [f"[STATE] {m}"]
        with STATE.lock:
            STATE.rejected.append(raw)
        return [f"[CMD] unknown mode: {arg}"]

    # ---------- TEXT ----------
    if cmd == "TEXT" and arg:
        with STATE.lock:
            STATE.line1 = arg[:23]                      # 固件 strncpy(buf,24)
        return [f"[CMD] TEXT = {STATE.line1}"]

    # ---------- DEMO ----------
    if cmd == "DEMO":
        on = bool(arg) and arg[0].upper() == "O"
        with STATE.lock:
            STATE.demo = on
        return [f"[CMD] DEMO {'ON' if on else 'OFF'}"]

    # ---------- LED ----------
    if cmd == "LED" and arg:
        v = _parse_ints(arg, 4)
        if len(v) < 3:
            with STATE.lock:
                STATE.rejected.append(raw)
            return ["[CMD] LED 用法: LED <r> <g> <b> [ms]   (0-255, ms 默认 3000)"]
        ms = v[3] if (len(v) >= 4 and v[3] > 0) else 3000
        ms = min(ms, 60000)
        r, g, b = (max(0, min(255, x)) for x in v[:3])
        with STATE.lock:
            STATE.led = (r, g, b)
            STATE.led_until = time.time() + ms / 1000.0
            STATE.led_on = True
        return [f"[LED] {r} {g} {b} hold {ms}ms"]

    # ---------- SPIN ----------
    if cmd == "SPIN" and arg:
        v = _parse_ints(arg, 2)
        if len(v) < 1:
            with STATE.lock:
                STATE.rejected.append(raw)
            return ["[CMD] SPIN 用法: SPIN <steps> [CW|CCW]   (4096 步 ≈ 一圈)"]
        up = arg.upper()
        d = -1 if "CCW" in up else 1
        steps = min(abs(v[0]), 200000)
        with STATE.lock:
            STATE.queue = steps
            STATE.queue_dir = d
        return [f"[SPIN] {steps} steps {'CW' if d > 0 else 'CCW'}"]

    # ---------- BEEP ----------
    if cmd == "BEEP" and arg:
        v = _parse_ints(arg, 2)
        f = max(50, min(8000, v[0])) if len(v) >= 1 else 880
        ms = max(10, min(3000, v[1])) if len(v) >= 2 else 200
        return [f"[BEEP] {f}Hz {ms}ms"]

    # ---------- STOP ----------
    if cmd == "STOP":
        with STATE.lock:
            STATE.queue = 0
            STATE.led_on = False
        return ["[STOP] motor off / buzzer off / light released"]

    # ---------- PING ----------
    if cmd == "PING":
        with STATE.lock:
            return [f"[PONG] {STATE.mode}"]

    # ---------- DUMP（固件没有，纯测试用）----------
    if cmd == "DUMP":
        s = STATE.snapshot()
        return ["[DUMP] mode=%s line1=%r led=%s led_on=%s led_left=%dms "
                "queue=%d dir=%d pos=%d recv=%d unknown=%d rejected=%d"
                % (s["mode"], s["line1"], s["led"], s["led_on"],
                   s["led_left_ms"], s["queue"], s["queue_dir"], s["pos"],
                   s["received"], s["unknown"], s["rejected"])]

    # ---------- HELP / 未知 ----------
    if cmd == "HELP":
        return list(HELP_LINES)
    with STATE.lock:
        STATE.unknown.append(raw)
    return list(HELP_LINES)


class Handler(socketserver.StreamRequestHandler):
    """一条 TCP 连接 = 一根"串口线"。"""

    def handle(self):
        peer = self.client_address
        print(f"[假ESP32] 客户端接入 {peer[0]}:{peer[1]}", file=sys.stderr, flush=True)
        # 对应固件 setup() 里那两行
        self._w("[OK] MOSS ready")
        for ln in HELP_LINES:
            self._w(ln)

        # 后台"走电机"：4ms 一步，走完打印一行（固件是静默的，这里打出来方便看）
        stop = threading.Event()
        t = threading.Thread(target=self._motor_loop, args=(stop,), daemon=True)
        t.start()
        try:
            while True:
                raw = self.rfile.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                print(f"[假ESP32] << {line}", file=sys.stderr, flush=True)
                for out in handle_command(line):
                    self._w(out)
                    time.sleep(0.01)          # 模拟串口逐个字符回来
        except (ConnectionResetError, OSError):
            pass
        finally:
            stop.set()
            print(f"[假ESP32] 断开 {peer[0]}:{peer[1]}", file=sys.stderr, flush=True)

    def _motor_loop(self, stop):
        """模拟电机把排队步数走完（每 4ms 一步）。"""
        while not stop.is_set():
            with STATE.lock:
                if STATE.queue > 0:
                    STATE.queue -= 1
                    STATE.pos += STATE.queue_dir
                    done = STATE.queue == 0
                    left, pos = STATE.queue, STATE.pos
                else:
                    done = False
            if done:
                self._safe_w(f"[SPIN] done, pos={pos}")
            time.sleep(0.004)

    def _w(self, line):
        self.wfile.write((line + "\n").encode("ascii", "replace"))
        self.wfile.flush()

    def _safe_w(self, line):
        try:
            self._w(line)
        except Exception:
            pass


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


# ================================================================
#  自检：python tools/mock_esp32.py --selftest
# ================================================================
def _selftest():
    print("=" * 66)
    print("  假 ESP32 自检（不联网、不开端口，直接测协议解析）")
    print("=" * 66)
    ok = True

    cases = [
        ("MODE SPEAK",   "[STATE] SPEAK"),
        ("MODE speak",   "[STATE] SPEAK"),
        ("MODE FLYING",  "[CMD] unknown mode: FLYING"),
        ("TEXT HELLO",   "[CMD] TEXT = HELLO"),
        ("LED 255 0 0 2000", "[LED] 255 0 0 hold 2000ms"),
        ("LED 999 -5 300",   "[LED] 255 0 255 hold 3000ms"),  # 钳制：999→255, -5→0, 300→255
        ("LED 1 2",      "[CMD] LED 用法"),                    # 参数不足 → 拒绝
        ("SPIN 4096",    "[SPIN] 4096 steps CW"),
        ("SPIN 4096 ccw", "[SPIN] 4096 steps CCW"),           # 小写也认
        ("SPIN 999999",  "[SPIN] 200000 steps CW"),           # 上限
        ("BEEP 880 200", "[BEEP] 880Hz 200ms"),
        ("BEEP",         None),                                # 没参数 → help
        ("PING",         "[PONG] "),                           # 前缀匹配
        ("STOP",         "[STOP] motor off"),
        ("WHAT",         "---- commands ----"),                # 未知 → help
    ]
    for cmd, expect in cases:
        out = handle_command(cmd)
        first = out[0]
        good = (expect is None and first == "---- commands ----") or \
               (expect is not None and first.startswith(expect))
        print(f"  {'✓' if good else '✗'} {cmd:<20} → {first}")
        ok = ok and good

    # LED 接管是真的有时限的
    handle_command("LED 0 255 0 300")
    assert STATE.led_on, "LED 应该处于接管中"
    time.sleep(0.35)
    assert STATE.led_expire(), "到点应该自动归还"
    print("  ✓ LED 接管 300ms 后自动归还（lightTick 等价逻辑）")

    # STOP 能把一切按平
    handle_command("SPIN 4096")
    handle_command("LED 255 0 0 5000")
    handle_command("STOP")
    s = STATE.snapshot()
    assert s["queue"] == 0 and not s["led_on"], s
    print("  ✓ STOP 之后：电机队列清零、灯环接管解除")

    # 统计：未知命令被记下来了（不是默默吞掉）
    assert STATE.unknown, "未知命令应该被记下来"
    print(f"  ✓ 未知命令被记录 {len(STATE.unknown)} 次（{STATE.unknown}）")
    print(f"  ✓ 参数不合法被拒 {len(STATE.rejected)} 次")

    print("\n[OK] 假 ESP32 自检通过。" if ok else "\n[失败] 有用例不符。")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description="假 ESP32 设备（离线验证执行器用）")
    p.add_argument("--port", type=int, default=4000,
                   help="监听端口，默认 4000（与 Wokwi rfc2217 相同，二选一）")
    p.add_argument("--host", default="127.0.0.1", help="监听地址")
    p.add_argument("--selftest", action="store_true", help="只跑协议自检，不开端口")
    a = p.parse_args()

    if a.selftest:
        return _selftest()

    srv = Server((a.host, a.port), Handler)
    print("=" * 74)
    print(f"  假 ESP32 已启动：tcp://{a.host}:{a.port}   （等价于一根串口线）")
    print("  用法示例：")
    print(f"    python moss_brain.py --demo --port socket://localhost:{a.port} \\")
    print("        --llm api --api-base http://localhost:8000/v1 \\")
    print("        --api-key fake --model mock-1")
    print("  提示：Wokwi 也用 4000，别同时开。")
    print("=" * 74, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    print("\n[假ESP32] 已退出。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
