#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rfc2217_mock_esp32.py —— 本机的「假 Wokwi」：一个最小 RFC2217 串口服务端
================================================================================
用途
----
阶段 7 里，moss_vision.py 通过 `rfc2217://localhost:4000` 把 MODE / TEXT 命令
发给 Wokwi 仿真里的 ESP32。但 Wokwi 仿真必须在 VS Code 图形界面里手动启动，
所以不方便自动化测试。

这个小工具在本机起一个 TCP 服务，**完全按 RFC2217 协议**跟 pyserial 客户端握手，
然后把它收到的每一行命令打印出来 —— 效果等同于「有人在 ESP32 那头接收」。
于是整条链路（视觉判定 → 串口写入 → 对端收到）就能离线验证。

它也可以当**排错工具**：
    如果你接了真机/Wokwi，但不确定是「Python 没发出去」还是「设备没收到」，
    先用这个假服务端跑一遍。能收到 → 说明 Python 端没问题。

运行
----
    python tools/rfc2217_mock_esp32.py                 # 监听 4000 端口
    python tools/rfc2217_mock_esp32.py --port 4000
    python tools/rfc2217_mock_esp32.py --once 20       # 20 秒后自动退出

协议说明（RFC2217 就是「Telnet + 串口参数协商」）
------------------------------------------------
    IAC=255  开始一个命令
    IAC IAC          → 转义，表示一个真正的 0xFF 数据字节
    IAC WILL/DONT X  → 我（/不）启用选项 X
    IAC DO/WONT X    → 请你（/不要）启用选项 X
    IAC SB X ... IAC SE → 子协商：选项 X 的参数
    BINARY(0) / COM_PORT_OPTION(44) 是本场景必须的两个选项
"""

import argparse
import socket
import sys
import time

# ------------------------------------------------------------------ Telnet 控制字节
IAC, DONT, DO, WONT, WILL, SB, SE = 0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF0
BINARY, ECHO, SGA, COM_PORT_OPTION = 0x00, 0x01, 0x03, 0x2C

# ------------------------------------------------------------------ RFC2217 子协商
# 客户端发来的「设置」选项 → 服务端要回的「应答」选项
SUB_ANSWER = {
    0x01: 0x65,   # SET_BAUDRATE    -> SERVER_SET_BAUDRATE
    0x02: 0x66,   # SET_DATASIZE    -> SERVER_SET_DATASIZE
    0x03: 0x67,   # SET_PARITY      -> SERVER_SET_PARITY
    0x04: 0x68,   # SET_STOPSIZE    -> SERVER_SET_STOPSIZE
    0x05: 0x69,   # SET_CONTROL     -> SERVER_SET_CONTROL
    0x06: 0x6A,   # NOTIFY_LINESTATE
    0x07: 0x6B,   # NOTIFY_MODEMSTATE
    0x0A: 0x6E,   # SET_LINESTATE_MASK
    0x0B: 0x6F,   # SET_MODEMSTATE_MASK
    0x0C: 0x70,   # PURGE_DATA      -> SERVER_PURGE_DATA
}


class Session:
    """一个客户端连接的 Telnet/RFC2217 状态机。"""

    def __init__(self, conn, addr, baud_hint, verbose):
        self.conn = conn
        self.addr = addr
        self.baud_hint = baud_hint
        self.verbose = verbose
        self.rx = bytearray()          # 收到「串口数据」（即 Python 发来的命令）
        self.line = bytearray()

    # ---------------------------------------------------------- 发送
    def send(self, data):
        try:
            self.conn.sendall(data)
        except OSError:
            pass

    def reply_neg(self, action, option):
        self.send(bytes((IAC, action, option)))

    def reply_sub(self, option, value):
        # 子协商里的 0xFF 必须转义成 0xFF 0xFF
        value = value.replace(b"\xff", b"\xff\xff")
        self.send(bytes((IAC, SB, COM_PORT_OPTION, option)) + value + bytes((IAC, SE)))

    # ---------------------------------------------------------- 主循环
    def run(self):
        state = "data"
        cmd = 0                 # 当前是 DO/DONT/WILL/WONT 中的哪个
        sub_opt = 0             # 当前子协商的 RFC2217 选项
        sub_val = bytearray()   # 当前子协商的值

        while True:
            try:
                chunk = self.conn.recv(4096)
            except OSError:
                break
            if not chunk:
                break

            for b in chunk:
                # ---------------- 普通数据 ----------------
                if state == "data":
                    if b == IAC:
                        state = "iac"
                    else:
                        self.on_serial_byte(b)

                # ---------------- 看到一个 IAC ----------------
                elif state == "iac":
                    if b == IAC:                       # 转义的 0xFF，是真数据
                        self.on_serial_byte(0xFF)
                        state = "data"
                    elif b in (DO, DONT, WILL, WONT):
                        cmd, state = b, "neg"
                    elif b == SB:
                        # 子协商结构：IAC SB <telnet选项=COM_PORT_OPTION>
                        #                    <RFC2217选项> <值> IAC SE
                        # 所以这里先进"读 telnet 选项"这一级，再读 RFC2217 选项。
                        state, sub_opt = "sub_topt", 0
                        sub_val.clear()
                    else:
                        state = "data"                 # 其它控制字（NOP 等）忽略

                # ---------------- 协商：IAC <cmd> <option> ----------------
                elif state == "neg":
                    self.handle_neg(cmd, b)
                    state = "data"

                # ---------------- 子协商：IAC SB <topt> <option> <value> IAC SE ----------------
                elif state == "sub_topt":              # 第 1 级：telnet 选项（应为 0x2C）
                    state = "sub_opt"

                elif state == "sub_opt":               # 第 2 级：RFC2217 子选项
                    sub_opt, state = b, "sub_val"

                elif state == "sub_val":
                    if b == IAC:
                        state = "sub_iac"
                    else:
                        sub_val.append(b)

                elif state == "sub_iac":
                    if b == SE:                        # 子协商结束
                        ans = SUB_ANSWER.get(sub_opt)
                        if ans is not None:
                            self.reply_sub(ans, bytes(sub_val))
                            if self.verbose:
                                print(f"    · 子协商 0x{sub_opt:02x} = "
                                      f"{bytes(sub_val).hex() or '-'} → 已应答 0x{ans:02x}")
                        elif self.verbose:
                            print(f"    · 子协商 0x{sub_opt:02x} 未知，忽略")
                        state = "data"
                    elif b == IAC:                     # 子协商里的转义 0xFF
                        sub_val.append(0xFF)
                        state = "sub_val"
                    else:
                        state = "sub_val"

    # ---------------------------------------------------------- 协商应答
    def handle_neg(self, cmd, option):
        if cmd == WILL:                 # 对端说「我启用 X」→ 我表示同意/不同意
            if option in (BINARY, COM_PORT_OPTION):
                self.reply_neg(DO, option)
            else:
                self.reply_neg(DONT, option)
        elif cmd == DO:                 # 对端要我「启用 X」
            if option in (BINARY, COM_PORT_OPTION, SGA):
                self.reply_neg(WILL, option)
            else:
                self.reply_neg(WONT, option)
        # WONT / DONT 无需回应

    # ---------------------------------------------------------- 串口数据
    def on_serial_byte(self, b):
        self.rx.append(b)
        if b in (0x0A, 0x0D):                       # 换行 → 凑齐一行命令
            line = bytes(self.line).decode("ascii", "replace").strip()
            self.line.clear()
            if line:
                # 这里就是「ESP32 收到命令」的时刻
                print(f"[ESP32 收到] {line}", flush=True)
                self.on_command(line)
        elif len(self.line) < 200:
            self.line.append(b)

    def on_command(self, line):
        """把命令翻译成人话，方便你一眼看出链路是否通了。"""
        up = line.upper()
        if up.startswith("MODE "):
            mode = line[5:].strip()
            print(f"             └─ 切换到状态：{mode}", flush=True)
        elif up.startswith("TEXT "):
            text = line[5:].strip()
            print(f"             └─ OLED 第三行显示：{text}", flush=True)
        else:
            print(f"             └─ （未知命令）", flush=True)


def serve(port, once_seconds, baud_hint, verbose):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(4)
    srv.settimeout(1.0)

    print("=" * 66)
    print(f"  假 Wokwi 串口服务已启动：rfc2217://localhost:{port}")
    print("  等 moss_vision.py / moss_audio.py 连进来……（Ctrl+C 退出）")
    print("=" * 66, flush=True)

    t0 = time.time()
    try:
        while True:
            if once_seconds and (time.time() - t0) >= once_seconds:
                print(f"[超时] 已运行 {once_seconds} 秒，退出。", flush=True)
                break
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue

            print(f"\n[连接] 客户端 {addr[0]}:{addr[1]} 已接入", flush=True)
            sess = Session(conn, addr, baud_hint, verbose)
            try:
                sess.run()                          # 阻塞直到客户端断开
            except Exception as e:
                print(f"[异常] {e}", flush=True)
            finally:
                conn.close()
                print(f"[断开] 客户端 {addr[0]}:{addr[1]} 已离开"
                      f"（共收到 {len(sess.rx)} 字节）", flush=True)
    except KeyboardInterrupt:
        print("\n[退出] 收到 Ctrl+C。", flush=True)
    finally:
        srv.close()


def main():
    p = argparse.ArgumentParser(description="最小 RFC2217 服务端：模拟 Wokwi 串口转发")
    p.add_argument("--port", type=int, default=4000, help="监听端口，默认 4000（要和 wokwi.toml 一致）")
    p.add_argument("--baud", type=int, default=115200, help="显示的波特率（仅提示用）")
    p.add_argument("--once", type=int, default=0, help="运行 N 秒后自动退出，0=一直运行")
    p.add_argument("--verbose", action="store_true", help="打印每一次子协商细节")
    args = p.parse_args()
    serve(args.port, args.once, args.baud, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
