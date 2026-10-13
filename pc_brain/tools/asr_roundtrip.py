#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS → ASR 闭环自测  ·  流浪地球 550W 复刻项目 · 阶段 12
============================================================
这是阶段 12 最有用的一个工具，解决一个实际问题：

    **没有麦克风（或者麦克风坏的），怎么验证"听得懂人话"？**

答案：自己造人来听。用阶段 11 那套 TTS（edge-tts）合成几句话，
拼成一段带静音间隔的音频 —— 正好模拟"说一句、停一下、再说一句"，
然后喂进真实的 ASR 链路（分段器 + whisper），看能不能原样认回来。

因为我们**知道标准答案**，所以这还是个准确率测试，不是"能跑就行"。

它一次验证了三件事：
    1. 分段器有没有把两句正确切开（不能切碎、不能并成一句）
    2. 前置缓冲够不够（第一个字有没有被切掉）
    3. whisper 认不认得出

生成的那段音频还能顺手存下来（`--keep-wav`），拿去喂给完整大脑：
    python moss_brain.py --audio-source _roundtrip.wav --asr whisper --llm api --tts edge

**edge-tts 不可用怎么办**（它依赖微软在线服务，会周期性改鉴权、报 403）：
用 `--from-files` 直接拿本地录音当音源，测试就不卡在合成这一步了。
音源文件和 `--text` 一一对应，顺序不能乱：

    python tools/asr_roundtrip.py --from-files a.mp3,b.wav --text "第一句|第二句"

用法：
    python tools/asr_roundtrip.py                                 # 用内置三句话
    python tools/asr_roundtrip.py --text "现在几点了|打开舱门"
    python tools/asr_roundtrip.py --model tiny --keep-wav _roundtrip.wav
    python tools/asr_roundtrip.py --noise 0.002                   # 加点背景噪声，更接近真实
    python tools/asr_roundtrip.py --from-files x.mp3 --text "本地录音内容"
"""

import argparse
import asyncio
import os
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asr import (TARGET_SR, UtteranceSegmenter, build_engine,   # noqa: E402
                 clean_text, decode_audio_file)

DEFAULT_TEXTS = ["现在几点了", "帮我看看画面里有什么", "你是谁"]


# ---------------------------------------------------------------- 合成
def synth(text, voice, rate, pitch, tmp_mp3):
    import edge_tts

    async def go():
        c = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        await c.save(tmp_mp3)
    asyncio.run(go())
    return decode_audio_file(tmp_mp3, TARGET_SR)


def load_local(path):
    """本地音频文件当音源（mp3 / wav / m4a 都行，走 PyAV 解码）。"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return decode_audio_file(path, TARGET_SR)


def normalize(s):
    return "".join(ch for ch in (s or "") if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def similarity(a, b):
    """很朴素的字级相似度：用 difflib，够用了。"""
    import difflib
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def write_wav(path, pcm, sr=TARGET_SR):
    data = np.clip(pcm, -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((data * 32767.0).astype("<i2").tobytes())


def main():
    p = argparse.ArgumentParser(description="TTS → ASR 闭环自测")
    p.add_argument("--text", default=None, help="用 | 分隔多句，默认内置 3 句")
    p.add_argument("--from-files", default=None,
                   help="用本地音频文件当音源（逗号分隔，与 --text 一一对应）。"
                        "edge-tts 报 403 时用这个绕过合成")
    p.add_argument("--voice", default="zh-CN-YunxiNeural")
    p.add_argument("--rate", default="-10%")
    p.add_argument("--pitch", default="-15Hz")
    p.add_argument("--model", default="small", help="whisper 模型")
    p.add_argument("--engine", default="whisper", choices=["whisper", "api"])
    p.add_argument("--api-base", default="https://api.openai.com/v1")
    p.add_argument("--api-key", default=None)
    p.add_argument("--silence", type=float, default=1.2, help="句间静音秒数，默认 1.2")
    p.add_argument("--lead", type=float, default=1.0, help="开头静音秒数，默认 1.0")
    p.add_argument("--noise", type=float, default=0.0,
                   help="叠加的高斯噪声幅度，如 0.002（约 -54dBFS）")
    p.add_argument("--keep-wav", default=None, help="把拼好的音频存成 WAV")
    p.add_argument("--dir", default=None, help="模型目录")
    p.add_argument("--mirror", default=None)
    a = p.parse_args()

    texts = a.text.split("|") if a.text else list(DEFAULT_TEXTS)

    use_files = bool(a.from_files)
    files = [f.strip() for f in a.from_files.split(",")] if use_files else []

    if use_files:
        if len(files) != len(texts):
            print("[错误] --from-files 给了 %d 个文件，--text 给了 %d 句，必须一一对应"
                  % (len(files), len(texts)), file=sys.stderr)
            return 2
    else:
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            print("[错误] 需要 edge-tts 才能合成测试语音：pip install edge-tts\n"
                  "       或者用 --from-files 拿本地录音当音源", file=sys.stderr)
            return 2

    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_synth.mp3")
    print("=" * 70)
    print("  TTS → ASR 闭环自测" + ("（本地文件音源）" if use_files else ""))
    print("=" * 70)

    gap = np.zeros(int(a.silence * TARGET_SR), dtype=np.float32)
    lead = np.zeros(int(a.lead * TARGET_SR), dtype=np.float32)

    pieces = [lead]
    print("  %s %d 句：" % ("载入" if use_files else "合成", len(texts)))
    for i, t in enumerate(texts):
        t0 = time.time()
        src = files[i] if use_files else t
        try:
            pcm = load_local(src) if use_files else synth(t, a.voice, a.rate, a.pitch, tmp)
        except Exception as e:
            print("    %d. [失败] %r：%s" % (i + 1, src, e), file=sys.stderr)
            if not use_files:
                print("        如果是 403 / WSServerHandshakeError，说明 edge-tts\n"
                      "        的在线合成被服务端拒了（外部问题）。用 --from-files\n"
                      "        拿一段本地录音当音源即可继续测。", file=sys.stderr)
            return 2
        print("    %d. %-24s 音频 %.2fs（%s %.1fs）"
              % (i + 1, t, len(pcm) / float(TARGET_SR),
                 "解码" if use_files else "合成", time.time() - t0))
        pieces.append(pcm)
        pieces.append(gap)
    try:
        os.remove(tmp)
    except OSError:
        pass

    audio = np.concatenate(pieces)
    if a.noise > 0:
        rng = np.random.default_rng(1)
        audio = audio + rng.normal(0, a.noise, len(audio)).astype(np.float32)
        print("  已叠加噪声幅度 %.4f" % a.noise)
    print("  拼好：总长 %.2f 秒（含 %.1fs 开头静音、%.1fs 句间静音）"
          % (len(audio) / float(TARGET_SR), a.lead, a.silence))

    if a.keep_wav:
        write_wav(a.keep_wav, audio)
        print("  已存：%s  ← 可以直接喂给 moss_brain --audio-source" % a.keep_wav)

    # ---------- 分段 ----------
    print("-" * 70)
    bs = 1024
    seg = UtteranceSegmenter(TARGET_SR, bs)
    utts = []
    for i in range(0, len(audio) - bs + 1, bs):
        utts.extend(seg.feed(audio[i:i + bs]))
    tail = seg.flush()
    if tail is not None:
        utts.append(tail)

    print("  【分段】期望 %d 段，实际切出 %d 段，时长 %s 秒"
          % (len(texts), len(utts), ["%.2f" % (len(u) / float(TARGET_SR)) for u in utts]))
    if len(utts) != len(texts):
        print("  ⚠️ 段数对不上 —— 调 --silence，或看 asr.py 里的 end_ms/min_speech_ms")
    print("         分段器统计 = %s" % seg.stats)

    # ---------- 识别 ----------
    print("-" * 70)
    opts = {"model": a.model, "dir": a.dir, "mirror": a.mirror or "https://hf-mirror.com",
            "api_base": a.api_base, "api_key": a.api_key, "lang": "zh"}
    engine, desc = build_engine(a.engine, opts)
    print("  【引擎】%s" % desc)

    hit = 0
    print("  【识别】")
    for i, u in enumerate(utts):
        t0 = time.time()
        got = clean_text(engine.transcribe(u, TARGET_SR))
        want = texts[i] if i < len(texts) else "（多出来的段）"
        sim = similarity(want, got)
        ok = sim >= 0.75
        hit += 1 if ok else 0
        print("    %s %d  期望 %-22s 得到 %-22s 相似度 %.2f  (%.2fs)"
              % ("✓" if ok else "✗", i + 1, want, got or "（空）", sim, time.time() - t0))
    engine.close()

    print("-" * 70)
    total = min(len(utts), len(texts))
    rate = (hit / float(total) * 100.0) if total else 0.0
    print("  结果：%d/%d 句识别正确（相似度≥0.75）→ %.0f%%" % (hit, total, rate))
    if total and hit != total:
        print("  建议：换 --model small，或去掉 --noise，或调大 --silence")
    print("=" * 70)
    return 0 if hit == total and len(utts) == len(texts) else 1


if __name__ == "__main__":
    sys.exit(main())
