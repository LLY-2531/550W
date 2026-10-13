# -*- coding: utf-8 -*-
"""
make_speech.py —— 用 SAPI 合成一段"人说的话"，存成 WAV
------------------------------------------------------------
【为什么需要它】
没有麦克风的人（比如本机：Windows 麦克风隐私权限关着，所有设备都录到
-96dBFS 的数字静音）根本演示不了"耳朵"这条链 —— 嘴巴能说话、眼睛能看，
但你说什么它听不见。

而 `moss_brain.py --audio-source xxx.wav` 本来就支持**把 WAV 当麦克风**。
所以只要先造出一段"人真的在说话"的 WAV，整条听觉链路（切句 → whisper
识别 → 唤醒门 → 工具调用 → 回答）就能**完全离线跑通**，一个字都不少。

这比"让用户修好麦克风权限"实用得多：
  - 权限问题各人机器不一样，说不清楚
  - 而 WAV 这条路径永远能用，还能**重复**跑（同一句话每次识别结果一样，
    这一点在做断言时很关键 —— 真人说话每次都不一样，测不了）

【为什么不用 PowerShell 的 Add-Type + System.Speech】
那条路更短，但 `Add-Type` 会在运行时编译并加载 .NET 代码，
很多环境的安全策略会直接拦掉（本机就拦了）。
`voice.py` 的 `SapiVoice` 走的是 COM（`SAPI.SpVoice` + `SAPI.SpFileStream`），
不编译任何东西，能过。这里复用同一套办法 —— **不抄第二份实现**。

【为什么刻意不设 Format.Type】
各版本 Windows 的 SAPI 默认输出格式不一样（有的是 22.05kHz 单声道，
有的是 16kHz）。硬写一个枚举值反而容易在别的机器上踩坑。
所以：写完照 WAV 头**读实际格式**打印出来，让 ASR 那边自己去适配。
（ASR 本来就是按 WAV 头解析的，不依赖某个固定采样率。）

用法：
    python tools/make_speech.py --say "摩丝，现在几点了" --out tools/assets/ask_time.wav
    python tools/make_speech.py --batch            # 生成一组演示用的固定句子
    python tools/make_speech.py --selftest         # 自检：造一段再读回来验格式
"""

import argparse
import base64
import os
import shutil
import subprocess
import sys
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")

# 演示用的一组固定句子。
# 选这四句不是随便挑的 —— 它们分别打到**四个不同的功能**上：
#   1. 带唤醒词 + 工具调用（get_time）
#   2. 工具调用 + 视觉（look）
#   3. 记忆写入（remember）
#   4. **不带唤醒词** —— 用来演示唤醒门真的会"不理你"
DEMO_LINES = [
    ("ask_time.wav", "摩丝，现在几点了"),
    ("ask_look.wav", "摩丝，看看画面里有什么"),
    ("ask_fact.wav", "摩丝，记住我住在杭州"),
    ("no_wake.wav", "现在几点了"),
]


def find_powershell():
    return shutil.which("powershell") or shutil.which("pwsh")


def psq(s):
    """PowerShell 单引号字符串里，单引号写成两个。"""
    return (s or "").replace("'", "''")


def synthesize(text, out_path, rate=0, volume=100, exe=None):
    """调 SAPI 说一句话，写进 out_path。返回 (采样率, 声道数, 位深, 秒数)。"""
    exe = exe or find_powershell()
    if not exe:
        raise RuntimeError("找不到 powershell.exe —— SAPI 合成不可用")

    script = (
        "$ErrorActionPreference = 'Stop'\n"
        "$v = New-Object -ComObject SAPI.SpVoice\n"
        "$v.Rate = %d\n"
        "$v.Volume = %d\n"
        # SpFileStream.Open(文件名, 模式, DoEvents)；模式 3 = SSFMCreateForWrite
        "$s = New-Object -ComObject SAPI.SpFileStream\n"
        "$s.Open('%s', 3, $false)\n"
        "$v.AudioOutputStream = $s\n"
        "$null = $v.Speak('%s', 0)\n"
        "$s.Close()\n" % (int(rate), int(volume), psq(out_path), psq(text))
    )

    # -EncodedCommand 收 base64(UTF-16LE)：中文路径/中文句子不会因为
    # 命令行编码被搞坏（这一条踩过，直接传中文会变成问号）
    enc = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    cmd = [exe, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
           "-EncodedCommand", enc]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0:
        err = (r.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError("SAPI 退出码 %s：%s" % (r.returncode, err[:200]))
    if not os.path.exists(out_path) or os.path.getsize(out_path) < 64:
        size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
        raise RuntimeError("没生成有效的 WAV（%s 字节）" % size)

    # 照 WAV 头读**实际**格式 —— 不假设它一定是 16kHz 单声道
    with wave.open(out_path, "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        secs = w.getnframes() / float(sr or 1)
    return sr, ch, sw * 8, secs


def _selftest():
    print("=" * 74)
    print("  make_speech.py 自检 —— 不验「合成得好不好听」，只验「能不能用」")
    print("=" * 74)

    tmp = os.path.join(ASSETS, "_selftest.wav")
    os.makedirs(ASSETS, exist_ok=True)

    print("\n[1/3] 造一段再读回来")
    sr, ch, bits, secs = synthesize("自检，一二三四五。", tmp)
    print("       ✓ 合成了 %.2f 秒 · %d Hz · %d 声道 · %d 位" % (secs, sr, ch, bits))
    assert secs > 0.3, "太短了，八成没真的说话（%.2fs）" % secs
    assert os.path.getsize(tmp) > 4000, "文件太小"

    print("\n[2/3] 格式必须是 ASR 能吃的那种")
    assert bits == 16, "SAPI 应当给 16 位 PCM，实际 %d 位" % bits
    assert ch in (1, 2), "声道数离谱：%d" % ch
    # 采样率不做硬断言：各版本 Windows 默认不一样（见文件头说明）。
    # 只要 ≥8kHz 就够 whisper 用，ASR 那边会自己重采样。
    assert sr >= 8000, "采样率 %d 太低，whisper 认不出来" % sr
    print("       ✓ %d 位 · %d 声道 · %d Hz（≥8kHz 即可，ASR 自己重采样）"
          % (bits, ch, sr))

    print("\n[3/3] 内容是裸 PCM，不是压缩格式")
    with wave.open(tmp, "rb") as w:
        assert w.getcomptype() == "NONE", "竟然是压缩 WAV：%s" % w.getcomptype()
        n, ch, sw = w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)

    # ⚠️ 别只读开头几帧就断言"有声音" —— SAPI 会在前面垫一段静音，
    # 第一版就是这么写的，前 45ms 全是 0，于是自检冤枉了代码。
    # 顺便把这段静音**量出来**：它其实有用，ASR 那边的切句器正好拿它
    # 当"噪声底"的参考（跟真人开口前必然的那点安静是一回事）。
    step = sw * ch
    import array
    samples = array.array("h")
    samples.frombytes(raw)
    assert len(samples) > 0, "一个采样都没有"
    # 短路 any()：只要找到一个非零就走人，不用扫完 6 万个样本
    first = next((i for i, v in enumerate(samples) if v != 0), None)
    assert first is not None, "整段全是 0 —— 那是数字静音，不是语音"
    lead_ms = first / float(sr) * 1000.0
    peak = max(abs(v) for v in samples)
    print("       ✓ 裸 PCM · 峰值 %d/32767 · 开头垫了 %.0fms 静音" % (peak, lead_ms))
    assert peak > 1000, "峰值只有 %d，基本等于没出声" % peak

    os.remove(tmp)
    print("\n[OK] make_speech.py 自检通过。")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="用 SAPI 合成一段语音存成 WAV（没有麦克风也能演示耳朵）")
    ap.add_argument("--say", help="要说的话")
    ap.add_argument("--out", help="输出路径（--say 时用）")
    ap.add_argument("--batch", action="store_true",
                    help="生成一组演示用的固定句子到 tools/assets/")
    ap.add_argument("--rate", type=int, default=0, help="SAPI 语速 -10~10，默认 0")
    ap.add_argument("--volume", type=int, default=100, help="音量 0~100")
    ap.add_argument("--exe", help="powershell.exe 路径（默认自动找）")
    ap.add_argument("--selftest", action="store_true", help="跑自检")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    if args.batch:
        os.makedirs(ASSETS, exist_ok=True)
        print("生成演示语音到 %s" % ASSETS)
        for name, text in DEMO_LINES:
            out = os.path.join(ASSETS, name)
            try:
                sr, ch, bits, secs = synthesize(text, out, args.rate,
                                                args.volume, args.exe)
                print("  ✓ %-14s %.2fs  %dHz/%dch/%dbit   %s"
                      % (name, secs, sr, ch, bits, text))
            except Exception as e:
                print("  ✗ %-14s %s" % (name, e))
                return 1
        print("\n下一步（不用麦克风，整条耳朵链路）：")
        print("  python moss_brain.py --asr whisper --audio-source "
              "tools/assets/ask_time.wav --asr-test")
        return 0

    if not args.say:
        ap.print_help()
        return 2
    if not args.out:
        print("--say 要和 --out 一起用", file=sys.stderr)
        return 2
    d = os.path.dirname(os.path.abspath(args.out))
    if d:
        os.makedirs(d, exist_ok=True)
    sr, ch, bits, secs = synthesize(args.say, args.out, args.rate,
                                    args.volume, args.exe)
    print("%s  ←  %.2f 秒 · %dHz · %d 声道 · %d 位   %s"
          % (args.out, secs, sr, ch, bits, args.say))
    return 0


if __name__ == "__main__":
    sys.exit(main())
