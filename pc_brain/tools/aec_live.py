#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 17 实测探针：这台机器上，它说话时到底能减掉多少回声？
============================================================
`aec.py --selftest` 和 `tools/fullduplex_e2e.py` 验的都是**逻辑**：
在一个合成房间里，滤波器学不学得会、闸门该不该开。
它们回答不了唯一一个真机上才有的问题：

    **你这套音箱+麦克风，回声有多大、时延有多长？**

而这两个数恰恰决定 AEC 能不能工作：
  - 回声太小（比如你在用耳机）→ 压根不需要 AEC，开了纯属浪费
  - 时延太长（蓝牙音箱普遍 150~300ms）→ 默认的 250ms 抽头不够，
    要按实测值调 `--aec-taps-ms`

这个脚本就是去量这两个数：放一段已知的音频，同时录麦克风，
然后**用录到的真音频**跑一遍 AEC，把结果打出来。

它会做三件事：
  1. 先静默录 1 秒 → 本机噪声底
  2. 放一段 3 秒的"像人说话"的信号，同时录 → 回声电平、声学时延
  3. 把这段录音离线喂给真的 EchoCanceller → 实测 ERLE

用法：
    python tools/aec_live.py            # 只做第 1 步（**不出声**，先看看录音通不通）
    python tools/aec_live.py --yes      # 出声，跑完整三步

⚠️ 会真的从扬声器放出声音。跑的时候请保持环境安静，
   并且**别戴耳机**（戴耳机基本测不到回声，见下面的说明）。
"""

import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from aec import EchoCanceller, ReferenceBuffer, speechlike, _Clock     # noqa: E402

SR = 16000
BLOCK = 1024


def _db(x):
    return 20.0 * math.log10(max(float(x), 1e-12))


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)) + 1e-12)


def _device(sd, kind):
    try:
        d = sd.query_devices(kind=kind)
        return d["name"], int(d.get("default_samplerate", SR))
    except Exception as e:
        return "（取不到：%s）" % e, SR


def estimate_delay(far, rec, max_ms=400.0):
    """用互相关估"扬声器放出去"到"麦克风收到"之间的时延。

    返回 (时延毫秒, 相关系数)。找的是**绝对值最大**的那个峰 ——
    回声通常是录音里最强的、和远端相关的东西。
    """
    n = min(len(far), len(rec))
    if n < SR // 10:
        return 0.0, 0.0
    f = far[:n].astype(np.float64)
    r = rec[:n].astype(np.float64)
    f -= f.mean()
    r -= r.mean()
    max_lag = int(max_ms / 1000.0 * SR)
    # FFT 互相关：比逐点移位快得多
    nfft = 1
    while nfft < 2 * n:
        nfft <<= 1
    cc = np.fft.irfft(np.fft.rfft(r, nfft) * np.conj(np.fft.rfft(f, nfft)))[:max_lag + 1]
    denom = (np.linalg.norm(r) * np.linalg.norm(f) + 1e-12)
    cc = cc / denom
    k = int(np.argmax(np.abs(cc)))
    return k / float(SR) * 1000.0, float(abs(cc[k]))


def run_aec_on(far, rec, taps_ms, resid_db=10.0, dtd_ratio=2.0):
    """把录到的真音频离线喂给真的 EchoCanceller，返回 (ERLE, 逐块明细)。"""
    clk = _Clock()
    ref = ReferenceBuffer(SR, seconds=8.0)
    aec = EchoCanceller(SR, taps_ms=taps_ms, subblock=64,
                        resid_db=resid_db, dtd_ratio=dtd_ratio)
    t_play = clk.t
    ref.start(t_play, far, SR)

    rows = []
    n = min(len(rec), len(far))
    for s in range(0, n - BLOCK + 1, BLOCK):
        blk = rec[s:s + BLOCK]
        t_end = t_play + (s + BLOCK) / float(SR)
        clk.tick(BLOCK / float(SR))
        out, info = aec.process(blk, t_end=t_end, ref=ref, now=clk())
        rows.append({"t": (s + BLOCK) / float(SR),
                     "mic_db": _db(_rms(blk)), "out_db": _db(_rms(out)),
                     "dtd": info["dtd"], "suppressed": info["suppressed"]})
    return aec, rows


def main():
    ap = argparse.ArgumentParser(description="阶段17 真机回声实测")
    ap.add_argument("--yes", action="store_true", help="确认出声（不加这个只静默录音）")
    ap.add_argument("--seconds", type=float, default=3.0, help="测试音频时长，默认 3s")
    ap.add_argument("--level", type=float, default=-20.0,
                    help="播放电平 dBFS，默认 -20（怕吵可以再调小）")
    ap.add_argument("--taps-ms", type=float, default=250.0, help="按这个抽头数试")
    ap.add_argument("--input", default=None, help="输入设备名/编号，默认用系统默认")
    ap.add_argument("--output", default=None, help="输出设备名/编号，默认用系统默认")
    a = ap.parse_args()

    try:
        import sounddevice as sd
    except ImportError:
        print("[失败] 没装 sounddevice：pip install sounddevice")
        return 2

    print("=" * 72)
    print("  阶段 17 真机实测：这台机器上的回声有多大、时延有多长")
    print("=" * 72)
    in_name, _ = _device(sd, "input")
    out_name, _ = _device(sd, "output")
    print(f"  输入：{in_name}")
    print(f"  输出：{out_name}")
    if in_name == out_name and ("耳机" in in_name or "headset" in in_name.lower()):
        print("  ⚠ 输入输出是同一个设备、而且看起来是耳机 —— 大概率测不到回声。")
    print()

    # ---------- 1. 静默录音，量噪声底 ----------
    print("[1/3] 静默录 1 秒，量本机噪声底……")
    try:
        quiet = sd.rec(int(1.0 * SR), samplerate=SR, channels=1, dtype="float32",
                       device=a.input)
        sd.wait()
    except Exception as e:
        print(f"[失败] 录音打不开：{e}")
        return 2
    quiet = np.asarray(quiet).reshape(-1)
    floor_db = _db(_rms(quiet))
    print(f"      噪声底 {floor_db:.1f} dBFS"
          f"{'（正常）' if floor_db < -45 else '（偏高，环境有底噪或者麦克风增益太大）'}")

    if not a.yes:
        print("\n  [不出声模式] 录音通了。想看回声，加 --yes 重跑。")
        print("  （它会真的放 3 秒声音出来，请确保环境安静、别戴耳机。）")
        return 0

    # ---------- 2. 播放 + 录音 ----------
    far = (speechlike(SR, a.seconds, seed=5) * (10 ** (a.level / 20.0) / 0.35)
           ).astype(np.float32)
    print(f"\n[2/3] 放 {a.seconds:.1f} 秒测试声（{a.level:.0f}dBFS）并同时录音……")
    print("      就是现在，请注意听（也别说话，说话会污染这次测量）")
    time.sleep(0.6)
    try:
        rec = sd.playrec(far, samplerate=SR, channels=1, dtype="float32",
                         device=(a.input, a.output))
        sd.wait()
    except Exception as e:
        print(f"[失败] 播放/录音失败：{e}")
        return 2
    rec = np.asarray(rec).reshape(-1)

    delay_ms, corr = estimate_delay(far, rec)
    # 回声电平：取"延迟附近"那一段，避免把环境噪声算进去
    i0 = int(delay_ms / 1000.0 * SR)
    seg = rec[i0:i0 + int(0.5 * SR)] if i0 + int(0.5 * SR) <= len(rec) else rec
    echo_db = _db(_rms(seg))
    coupling = echo_db - _db(_rms(far))
    snr = echo_db - floor_db

    print(f"      远端（放出去的）电平 {_db(_rms(far)):.1f} dBFS")
    print(f"      回声（录回来的）电平 {echo_db:.1f} dBFS"
          f"   ← 比本底高 {snr:.1f} dB")
    print(f"      声学时延 ≈ {delay_ms:.0f} ms（相关系数 {corr:.2f}）")
    print()

    if coupling < -35.0 or snr < 6.0:
        print("  ⚠ 几乎听不到自己的声音（耦合 %.1f dB）。两种可能：" % coupling)
        print("      · 你在用耳机 —— 扬声器和麦克风之间没有空气路径，本来就没回声；")
        print("        这种配置下 AEC 大可关掉（--aec off），开着只是白跑。")
        print("      · 或者音量太小 —— 调大系统音量再跑一次。")
        print("    结论：**没法在这套设备上量 ERLE**。")
        return 0

    # ---------- 3. 离线跑 AEC ----------
    print("[3/3] 用这段真录音跑一遍 AEC（抽头 %.0fms）……" % a.taps_ms)
    aec, rows = run_aec_on(far, rec, a.taps_ms)
    body = rows[len(rows) // 3:]          # 前 1/3 是收敛期，不算
    if body:
        erle = sum(r["mic_db"] - r["out_db"] for r in body) / len(body)
        print(f"      实测 ERLE = {erle:.1f} dB（后 2/3 平均）")
        print(f"      收敛过程 → " + "  ".join(
            "%ds:%.0fdB" % (rows[i]["t"], rows[i]["mic_db"] - rows[i]["out_db"])
            for i in range(0, len(rows), max(1, len(rows) // 4))[:5]))
    print("      " + aec.stats_line())
    print()
    if aec.n_reset:
        print("      ⚠ 有发散复位。真机上通常意味着时延超过了抽头长度 ——")
        print("        加 --taps-ms 400 再试（蓝牙音箱尤其容易这样）。")
    print("  " + "-" * 68)
    if body and erle >= 15.0:
        print("  [OK] 这台机器上 AEC 可用：回声被压下去 %.0f dB。" % erle)
        print("       它说话的时候你不用等它说完，直接开口就行。")
    elif body:
        print("  [提示] ERLE 只有 %.0f dB，偏低。可以试：" % erle)
        print("       · --taps-ms 350（时延 %.0fms，留够余量）" % delay_ms)
        print("       · 把系统音量调小一点（回声小，更好减）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
