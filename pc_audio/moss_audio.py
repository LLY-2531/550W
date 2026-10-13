#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOSS 听觉大脑  ·  流浪地球 550W 复刻项目 · 阶段 8
============================================================
它做的事：
    1. 打开电脑麦克风，按固定大小的「块」连续读取声音
    2. 算每一块的音量（RMS → 分贝 dBFS），并维护一条「自适应噪声底」
    3. 判断「现在有没有人在说话 / 有没有响声」（信噪比 + 绝对门限）
    4. 顺带识别「拍手」这种短促爆音
    5. 把结果整理成一行 JSON 输出（方便接大模型/存日志）
    6. （可选）通过串口把 MODE / TEXT 命令推给 ESP32，让 MOSS 进入「在听」

对应真机上的 INMP441 数字麦克风：真机是 ESP32 自己采音，这里换成
电脑麦克风先采、再通过串口把「听没听见」的结论告诉 ESP32。
逻辑跑通之后，把这一段搬到固件里即可，协议完全一致。

三种声音来源（--source）：
    默认          真实麦克风（默认设备，也可以 --device 指定）
    demo          合成音频（没有麦克风也能跑，用来验证整条链路）
    xxx.wav       录音文件（循环播放，适合用来测「识别准不准」）

典型用法：
    python moss_audio.py --list                    # 先看有哪些设备
    python moss_audio.py                           # 用默认麦克风
    python moss_audio.py --source demo             # 没麦克风时先试这个
    python moss_audio.py --port rfc2217://localhost:4000   # 发给 Wokwi 仿真
    python moss_audio.py --no-meter --frames 200   # 不显示电平条、跑 200 块就退

输出示例：
    {"t": 4.21, "dbfs": -27.4, "floor": -58.1, "snr": 30.7,
     "clap": false, "mode": "LISTEN", "text": "sound -27dB", "sent": true}
"""

import argparse
import collections
import json
import math
import os
import queue
import sys
import time
import wave

# ---------------------------------------------------------------- 依赖检查
try:
    import numpy as np
except ImportError:
    sys.exit("[缺少依赖] 没装 NumPy。请在终端执行：pip install numpy")

try:
    import sounddevice as sd
except ImportError:
    sys.exit("[缺少依赖] 没装 sounddevice。请在终端执行：pip install sounddevice")

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
        description="MOSS 听觉大脑 —— 读麦克风，判断有没有声音，输出 JSON，可选推送串口"
    )
    p.add_argument("--device", default=None,
                   help="输入设备：留空=系统默认；也可填编号(如 2)或名字的一部分")
    p.add_argument("--list", action="store_true", help="列出所有音频设备后退出")
    p.add_argument("--probe", action="store_true",
                   help="逐个试听所有输入设备、打印各自音量后退出（排查「没声音」用）")
    p.add_argument("--probe-seconds", type=float, default=0.4,
                   help="每个设备试听多久，默认 0.4 秒")
    p.add_argument("--auto", action="store_true",
                   help="自动试听并挑一个真正有信号的输入设备（见 README 第三节）")
    p.add_argument("--source", default="mic",
                   help="声音来源：mic=麦克风 / demo=合成音频 / 路径.wav=录音文件")
    p.add_argument("--samplerate", type=int, default=16000,
                   help="采样率，默认 16000（和 INMP441 常用值一致）")
    p.add_argument("--blocksize", type=int, default=1024,
                   help="每块样本数，默认 1024（16000Hz 下约 64 毫秒一块）")
    p.add_argument("--db-th", type=float, default=-50.0,
                   help="绝对门限(dBFS)：低于它一定算安静，默认 -50")
    p.add_argument("--snr-th", type=float, default=8.0,
                   help="信噪比门限(dB)：比噪声底高出这么多才算有声音，默认 8")
    p.add_argument("--clap-th", type=float, default=-12.0,
                   help="拍手判定门限(dBFS)：峰值超过它且前一块很安静，默认 -12")
    p.add_argument("--no-clap", action="store_true", help="关掉拍手检测")
    p.add_argument("--port", default=None,
                   help="串口。COM3=真机；rfc2217://localhost:4000=Wokwi 仿真；不给就不发")
    p.add_argument("--baud", type=int, default=115200, help="串口波特率，默认 115200")
    p.add_argument("--frames", type=int, default=0, help="跑多少块后退出，0=一直跑")
    p.add_argument("--stable", type=int, default=3,
                   help="连续 N 块结果一致才切状态，默认 3（防抖动）")
    p.add_argument("--heartbeat", type=float, default=2.0,
                   help="状态没变时，每隔几秒也输出一行 JSON，默认 2 秒")
    p.add_argument("--no-meter", action="store_true", help="不显示终端电平条")
    p.add_argument("--verbose", action="store_true", help="每一块都输出 JSON")
    return p.parse_args()


def list_devices():
    print("=" * 72)
    print("  可用的音频设备（★ = 系统默认输入）")
    print("=" * 72)
    try:
        default_in = sd.query_devices(kind="input")["name"]
    except Exception:
        default_in = None

    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] <= 0:
            continue                                    # 只列能录音的设备
        star = "★" if d["name"] == default_in else " "
        host = sd.query_hostapis(d["hostapi"])["name"]
        print(f" {star} [{i:2d}] {d['name']}")
        print(f"         {d['max_input_channels']} 通道 · 默认 {d['default_samplerate']:.0f} Hz"
              f" · {host}")
    print()
    print("用法：python moss_audio.py --device 2        # 按编号选")
    print("      python moss_audio.py --device 麦克风阵列   # 或者按名字的一部分选")
    print("      python moss_audio.py --probe           # 不知道选哪个？让它替你试听一遍")


def probe_inputs(seconds=0.4):
    """逐个试听所有输入设备，按实际音量从大到小返回。

    为什么要干这件事：Windows 上一个麦克风常常同时挂在 MME / DirectSound /
    WASAPI / WDM-KS 四套接口下，而**其中有些接口会给出「数字静音」**
    （一直是 -96 dBFS，读到的全是零）。这时你会以为"麦克风坏了"，
    其实只是选错了接口。挨个试一遍是最快的确诊办法。

    返回值里 rms 是 RMS 电平(dBFS)，越大说明越"听得见"。
    """
    results = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] <= 0:
            continue                                   # 只能放音的，跳过
        rate = int(d["default_samplerate"])
        ch = min(2, d["max_input_channels"])
        try:
            rec = sd.rec(int(seconds * rate), samplerate=rate, channels=ch,
                         device=i, dtype="float32", blocking=True)
            mono = rec.mean(axis=1)
        except Exception:
            continue                                   # 打不开的设备直接跳过
        if mono.size == 0:
            continue
        rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
        peak = float(np.max(np.abs(mono)))
        results.append({
            "index": i,
            "name": d["name"],
            "host": sd.query_hostapis(d["hostapi"])["name"],
            "rate": rate,
            "rms": 20.0 * math.log10(rms + 1e-9),
            "peak": 20.0 * math.log10(peak + 1e-9),
        })
    results.sort(key=lambda r: r["rms"], reverse=True)
    return results


def sane(entry):
    """判断一个设备的读数是否「可信」。

    数字静音大约 -96 dBFS，完全没有数据的会报到 -180；偶尔还有驱动 bug
    报出正几百 dB 的假值。这些都不是真实声音，自动挑选时要排除掉。
    """
    return -150.0 < entry["rms"] < 5.0


def cmd_probe(seconds):
    print("=" * 76)
    print(f"  逐个试听输入设备（每个 {seconds} 秒）—— 音量大的排前面")
    print("=" * 76)
    rows = probe_inputs(seconds)
    if not rows:
        print("  一个能打开的设备都没有。检查麦克风是否被系统禁用。")
        return 1

    print(f"{'排名':<5}{'编号':<6}{'名称':<34}{'接口':<14}{'RMS dBFS':>10}")
    print("-" * 76)
    for n, r in enumerate(rows, 1):
        flag = "" if sane(r) else "  ← 假值/静音，忽略"
        print(f"{n:<5}{r['index']:<6}{r['name'][:32]:<34}{r['host'][:12]:<14}{r['rms']:>10.1f}{flag}")

    best = next((r for r in rows if sane(r)), None)
    print("-" * 76)
    if best is None:
        print("  所有设备都是静音 —— 麦克风可能被系统静音了：")
        print("    Win 设置 → 系统 → 声音 → 输入 → 确认设备没被静音、音量不为 0")
        print("    另外检查 设置 → 隐私和安全性 → 麦克风 → 「允许应用访问麦克风」是否打开")
    else:
        print(f"  ✔ 建议使用：--device {best['index']}    （{best['name']} · {best['host']}）")
        print(f"    或者直接加 --auto，让程序每次自己挑。")
        if best["rms"] < -85:
            print("  ⚠ 最好的设备也只有 %.1f dBFS，接近静音。请对着麦克风说句话再试一次。" % best["rms"])
    return 0


# ================================================================
#  2. 声音来源
# ================================================================
class MicSource:
    """真实麦克风。用回调把数据丢进队列，主线程慢慢算 —— 这样不会丢样本。"""

    def __init__(self, device, samplerate, blocksize):
        self.q = queue.Queue(maxsize=64)
        self.samplerate = samplerate
        self.stream = sd.InputStream(
            device=device, channels=1, samplerate=samplerate,
            blocksize=blocksize, dtype="float32", callback=self._callback,
        )
        self.stream.start()

    def _callback(self, indata, frames, time_info, status):
        mono = indata[:, 0].copy()
        try:
            self.q.put_nowait(mono)
        except queue.Full:
            pass                                        # 算不过来就丢最老的，不让延迟堆积

    def read(self):
        try:
            return self.q.get(timeout=2.0)
        except queue.Empty:
            return None

    def release(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


def open_mic(device, samplerate, blocksize):
    """打开麦克风。请求的采样率不支持下就退回设备默认值 —— 别让新手卡在这。"""
    errors = []
    for rate in (samplerate, None):
        try:
            info = sd.query_devices(device, "input")
            r = rate if rate else info["default_samplerate"]
            sd.check_input_settings(device=device, channels=1,
                                    samplerate=r, dtype="float32")
            src = MicSource(device, r, blocksize)
            if rate is None:
                print(f"[提示] 设备不支持 {samplerate}Hz，已改用默认 {r:.0f}Hz",
                      file=sys.stderr)
            return src
        except Exception as e:
            errors.append(f"{e}")
    raise RuntimeError(errors[-1] if errors else "打不开麦克风")


class DemoSource:
    """没有麦克风时的替身：一段「安静 → 拍手 → 说话」的循环。

    一个周期 3.2 秒（按 16000Hz / 1024 样本算 = 50 块）：
        第 0~18 块     安静（接近静音）—— 先让噪声底学一下环境
        第 19 块       拍手（极短爆音，用来验证拍手检测）
        第 20~49 块    说话（带音节起伏的音调，听起来像"哇啦哇啦"）

    特意让"安静"排在最前面：这样启动时噪声底能先学到一个干净的本底，
    和真实场景一致（真实场景绝不会一开机就有人对着麦克风拍手）。
    """

    def __init__(self, samplerate, blocksize):
        self.sr, self.n = samplerate, blocksize
        self.i = 0
        self.cycle_blocks = max(6, int(3.2 * samplerate / blocksize))
        self.quiet_blocks = self.cycle_blocks * 19 // 50
        self.talk_blocks = self.quiet_blocks + 1          # 拍手后立刻开始说话

    def read(self):
        k = self.i % self.cycle_blocks
        t = (np.arange(self.n) + self.i * self.n) / self.sr

        if k < self.quiet_blocks:                        # 安静
            sig = 0.0006 * np.random.randn(self.n)
        elif k == self.quiet_blocks:                     # 拍手
            s = np.zeros(self.n, dtype=np.float32)
            s[: max(1, self.n // 40)] = 0.9
            sig = s
        else:                                            # 说话
            env = 0.45 + 0.55 * np.sin(2 * math.pi * 3.5 * t) ** 2
            tone = np.sin(2 * math.pi * 160 * t) + 0.5 * np.sin(2 * math.pi * 320 * t)
            sig = 0.30 * env * tone + 0.015 * np.random.randn(self.n)

        self.i += 1
        return sig.astype(np.float32)

    def release(self):
        pass


class WavSource:
    """录音文件。循环播放，用来测「识别得准不准」。"""

    def __init__(self, path, blocksize):
        with wave.open(path, "rb") as w:
            n_ch = w.getnchannels()
            width = w.getsampwidth()
            self.sr = w.getframerate()
            raw = w.readframes(w.getnframes())

        if width == 2:
            data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 1:
            data = (np.frombuffer(raw, dtype="<u1").astype(np.float32) - 128.0) / 128.0
        elif width == 4:
            data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise RuntimeError(f"不支持的位深：{width * 8} bit")

        if n_ch > 1:                                    # 多声道 → 取平均变单声道
            data = data.reshape(-1, n_ch).mean(axis=1)

        self.data = data
        self.n = blocksize
        self.pos = 0

    def read(self):
        if len(self.data) == 0:
            return None
        end = self.pos + self.n
        if end <= len(self.data):
            blk = self.data[self.pos:end]
            self.pos = end
        else:                                           # 到尾巴了 → 回开头补上
            blk = np.concatenate([self.data[self.pos:], self.data[: end - len(self.data)]])
            self.pos = end - len(self.data)
        return blk.astype(np.float32)

    def release(self):
        pass


def make_source(args):
    if args.source == "demo":
        print("[OK] 使用合成音频（demo 模式）", file=sys.stderr)
        return DemoSource(args.samplerate, args.blocksize)
    if args.source.lower().endswith(".wav"):
        if not os.path.exists(args.source):
            raise FileNotFoundError(args.source)
        src = WavSource(args.source, args.blocksize)
        print(f"[OK] 使用录音文件：{args.source}（{src.sr}Hz）", file=sys.stderr)
        return src
    return open_mic(args.device, args.samplerate, args.blocksize)


# ================================================================
#  3. 串口（和 moss_vision.py 用同一套协议）
# ================================================================
def open_serial(spec, baud):
    """连不上串口不算错误 —— 只输出 JSON 也能用，所以这里不 raise。"""
    if not HAS_SERIAL:
        print("[提示] 没装 pyserial，串口功能关闭。需要就执行：pip install pyserial",
              file=sys.stderr)
        return None
    try:
        if "://" in spec:
            ser = serial.serial_for_url(spec, baudrate=baud, timeout=0.2)
        else:
            ser = serial.Serial(spec, baud, timeout=0.2)
        time.sleep(0.3)
        print(f"[OK] 串口已连接：{spec}", file=sys.stderr)
        return ser
    except Exception as e:
        print(f"[提示] 串口连接失败（{spec}）：{e}", file=sys.stderr)
        print("       → 继续只输出 JSON，音频识别不受影响。", file=sys.stderr)
        return None


def send_state(ser, mode, text):
    """把状态推给 ESP32。协议就是固件里那几个串口命令。"""
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
#  4. 分析：音量 → 噪声底 → 有没有声音
# ================================================================
class NoiseFloor:
    """自适应噪声底：取「最近几秒音量」的 10% 分位数。

    为什么要这么绕？因为房间的本底噪声会变（开空调、开风扇），必须让它自己跟着走。
    但**不能**用平均值或简单的指数平滑 —— 那样人一说话、或者啪一下拍手，
    噪声底就被抬上去了，接下来反而听不见安静的差别。

    用「低分位数」的意思是：把最近几秒的音量从小到大排队，取第 10% 名那个值。
    这个值大概率落在"没人说话的那些瞬间"，于是天然抗干扰。

    trusted=False 的块（比如判定为拍手的瞬态）不会进窗口 —— 见 update()。
    """

    def __init__(self, window_seconds=4.0, block_seconds=0.064, percentile=10.0):
        cap = max(8, int(round(window_seconds / max(block_seconds, 1e-3))))
        self.buf = collections.deque(maxlen=cap)
        self.pct = percentile

    def update(self, level_db, trusted=True):
        if trusted:
            self.buf.append(level_db)
        if not self.buf:
            return level_db - 40.0        # 还没有可信数据 → 保守处理，先别急着"听见"
        return float(np.percentile(self.buf, self.pct))


def block_level(block):
    """返回 (RMS 电平 dBFS, 峰值 dBFS)。"""
    rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(block))) if block.size else 0.0
    level = 20.0 * math.log10(rms + 1e-9)
    peak_db = 20.0 * math.log10(peak + 1e-9)
    return level, peak_db


def decide(level, floor, args, is_clap):
    """业务规则：拍到响 → 在听；明显高于噪声底 → 在听；否则 → 待机。

    注意 text 用的是**稳定标签**（hearing / quiet / clap!），不是实时 dB 值。
    原因：实时 dB 每块都在变，如果拿它当 text，串口会每个块都发一次 TEXT，
    而 OLED 也一直在闪数字，人根本看不清。精确的电平放在 JSON 的 dbfs 字段里，
    给日志和上位机分析用。
    """
    if is_clap:
        return "LISTEN", "clap!", True
    if level >= args.db_th and (level - floor) >= args.snr_th:
        return "LISTEN", "hearing", False
    return "IDLE", "quiet", False


def draw_meter(level, floor, gate_db, mode, is_clap):
    """在终端画一条电平条（写到 stderr，不污染 stdout 的 JSON）。

    `=` 表示已超过判定门限的部分，`#` 是还没到的部分，`|` 标出门限位置。
    """
    width = 30
    lo, hi = -70.0, 0.0

    def pos(db):
        return max(0, min(width, int((db - lo) / (hi - lo) * width)))

    n = pos(level)
    gate = pos(gate_db)
    bar = ["-"] * width
    for i in range(n):
        bar[i] = "=" if i < gate else "#"
    if 0 < gate < width:
        bar[gate] = "|"

    tag = "CLAP" if is_clap else (mode or "-")
    sys.stderr.write(
        f"\r[{''.join(bar)}] {level:7.1f} dB  底 {floor:6.1f}  门限 {gate_db:6.1f}  {tag:<6}"
    )
    sys.stderr.flush()


# ================================================================
#  5. 主流程
# ================================================================
def resolve_device(spec):
    """把 --device 的值统一成设备编号。支持填编号，也支持填名字的一部分。"""
    if spec is None:
        return None
    s = str(spec).strip()
    if s.lstrip("-").isdigit():
        return int(s)
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and s.lower() in d["name"].lower():
            return i
    raise RuntimeError(f"找不到名字含「{spec}」的输入设备。用 --list 看看有哪些。")


def main():
    args = parse_args()

    if args.list:
        list_devices()
        return 0

    if args.probe:
        return cmd_probe(args.probe_seconds)

    # ---------- 自动挑设备：试听一遍，选真正有信号的那个 ----------
    if args.auto and args.source == "mic" and args.device is None:
        rows = probe_inputs(args.probe_seconds)
        best = next((r for r in rows if sane(r)), None)
        if best is None:
            print("[错误] 没找到有信号的输入设备。跑一次 --probe 看详情。", file=sys.stderr)
            return 2
        args.device = best["index"]
        print(f"[自动选设备] --device {best['index']}  {best['name']}"
              f"（{best['host']}，当前 {best['rms']:.1f} dBFS）", file=sys.stderr)

    if args.source == "mic":
        try:
            args.device = resolve_device(args.device)
        except RuntimeError as e:
            print(f"[错误] {e}", file=sys.stderr)
            return 2

    try:
        source = make_source(args)
    except Exception as e:
        print(f"[错误] 打不开声音来源 '{args.source}'：{e}", file=sys.stderr)
        print("       排查顺序：", file=sys.stderr)
        print("         1) python moss_audio.py --probe     # 看哪个设备真有信号",
              file=sys.stderr)
        print("         2) python moss_audio.py --auto      # 让程序自己挑",
              file=sys.stderr)
        print("         3) python moss_audio.py --source demo  # 先不管麦克风，验证链路",
              file=sys.stderr)
        return 2

    ser = open_serial(args.port, args.baud) if args.port else None
    sr_used = getattr(source, "samplerate", getattr(source, "sr", args.samplerate))

    print("[OK] MOSS 听觉大脑启动。按 Ctrl+C 退出。", file=sys.stderr)
    print("-" * 74, file=sys.stderr)

    floor_tracker = NoiseFloor(
        window_seconds=4.0,
        block_seconds=args.blocksize / float(sr_used or args.samplerate),
    )
    cur_mode = None
    cur_text = ""
    pending = None
    stable_n = 0
    prev_peak_db = -99.0
    last_json = 0.0
    block_i = 0
    t0 = time.time()
    last_dump = 0.0

    try:
        while True:
            iter_t0 = time.time()
            block = source.read()
            if block is None:
                print("\n[提示] 读不到声音，结束。", file=sys.stderr)
                break

            block_i += 1
            level, peak_db = block_level(block)

            # ---------- 拍手：峰值很高 + 前一块很安静（短促爆音的特征）----------
            is_clap = False
            if not args.no_clap:
                is_clap = (peak_db >= args.clap_th) and (prev_peak_db < args.clap_th - 12.0)
            prev_peak_db = peak_db

            # ---------- 噪声底：拍手这种瞬态不进窗口，免得把本底抬上去 ----------
            floor = floor_tracker.update(level, trusted=not is_clap)
            snr = level - floor
            gate_db = max(floor + args.snr_th, args.db_th)   # 实际起作用的判定门限

            mode, text, _ = decide(level, floor, args, is_clap)

            # ---------- 防抖动 ----------
            if mode == pending:
                stable_n += 1
            else:
                pending, stable_n = mode, 1

            changed = False
            if is_clap:
                # 拍手是"事件"，别被防抖动吃掉，立刻响应
                if cur_mode != mode or cur_text != text:
                    cur_mode, cur_text, changed = mode, text, True
                pending, stable_n = mode, 0
            elif stable_n >= args.stable and (mode != cur_mode or text != cur_text):
                cur_mode, cur_text, changed = mode, text, True

            # ---------- 输出 ----------
            now = time.time()
            if changed or args.verbose or (now - last_json) >= args.heartbeat:
                sent = send_state(ser, cur_mode, cur_text) if changed else False
                rec = {
                    "t": round(now - t0, 2),
                    "dbfs": round(level, 1),
                    "floor": round(floor, 1),
                    "snr": round(snr, 1),
                    "clap": bool(is_clap),
                    "mode": cur_mode,
                    "text": cur_text,
                    "sent": bool(sent),
                }
                print(json.dumps(rec, ensure_ascii=False), flush=True)
                last_json = now

            # ---------- 终端电平条 ----------
            if not args.no_meter:
                if now - last_dump >= 0.05:
                    draw_meter(level, floor, gate_db, cur_mode, is_clap)
                    last_dump = now

            # ---------- 限速：demo / wav 按真实时间播放，不空转 ----------
            if args.source != "mic":
                block_seconds = args.blocksize / float(sr_used or args.samplerate)
                sleep_left = block_seconds - (time.time() - iter_t0)
                if sleep_left > 0:
                    time.sleep(sleep_left)

            if args.frames and block_i >= args.frames:
                break

    except KeyboardInterrupt:
        print("\n[提示] 收到 Ctrl+C，退出。", file=sys.stderr)
    finally:
        source.release()
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
        if not args.no_meter:
            sys.stderr.write("\n")

    print(f"[OK] 结束。共处理 {block_i} 块。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
