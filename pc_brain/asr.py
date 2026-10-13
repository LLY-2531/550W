#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOSS 听觉理解层（ASR）  ·  流浪地球 550W 复刻项目 · 阶段 12
============================================================
阶段 8 给了 MOSS「耳朵」，但那只耳朵只能听出三件事：
    「有声音 / 多大声 / 是不是拍手」
它不知道你在说什么，所以阶段 10/11 的 MOSS 只会自顾自地汇报
——「我看见你了」「我在待机」。它是**广播**，不是**对话**。

这一步把"声音"变成"文字"，MOSS 才能真的接话。

四块拼图
------------------------------------------------------------
    1. UtteranceSegmenter   把连续音频流切成一"句"（VAD 状态机 + 前置缓冲）
    2. AsrEngine            把一句音频转成文字
       ├─ WhisperEngine     本地 faster-whisper：离线、免费、首次下模型
       ├─ ApiEngine         OpenAI 兼容 /audio/transcriptions：不下模型、要联网
       └─ NullEngine        关闭
    3. AsrWorker            独立线程 —— 识别几百毫秒，绝不能卡住 20Hz 融合循环
    4. 接进 moss_brain       heard 文本 → mind.py → 大模型回应 → voice.py 说出来

半双工：MOSS 说话时，它听不见你
------------------------------------------------------------
阶段 11 为了防「自说自听」，说话期间把麦克风按静音算。ASR 是同一个物理限制，
而且更严重 —— 如果不禁，MOSS 会把自己说的话录下来、识别成文字、再当成
"用户说的话"回给自己，**无限自我对话**。

彻底解法是 AEC（回声消除，像手机免提那样从麦克风信号里减掉自己的声音），
那是另一个量级的工程。所以这里选择**半双工**（对讲机模式）：

    MOSS 说话  →  你的话它听不见（麦克风屏蔽）
    它说完     →  麦克风打开，你说，它听

这是绝大多数 DIY 语音助手的实际做法。`--tts off` 时没有这个问题。

延迟与"什么时候算说完"
------------------------------------------------------------
VAD 判断"一句话结束"靠的是**静音持续多久**（`--asr-end-ms`，默认 700ms）。
这个数字有取舍：
    太小 → 你说话中间一停顿，就被切成两句
    太大 → 你说完要等很久它才反应
700ms 是"正常说话换气"和"真的说完了"之间的常用分界。

另外**前置缓冲（preroll）**很关键：VAD 从"听到声音"到"确认人开始说话"
要连续几块超阈值，这几块的音频如果不留，第一个字就被切掉了 ——
表现就是"MOSS 老是听漏我开头"。所以环形缓冲一直保留最近 300ms。

模型从哪来（重要，国内必看）
------------------------------------------------------------
huggingface.co 在国内基本不可达，但 **hf-mirror.com** 通。所以这里
**不依赖 huggingface_hub**（它在 Windows 上还会因为建不了软链而留下 0 字节文件），
而是直接从镜像把这 4 个文件拉下来：

    config.json / model.bin / tokenizer.json / vocabulary.txt

拉完就是一个普通目录，`WhisperModel(目录)` 直接加载。透明、可控、好排查。

用法
------------------------------------------------------------
    python asr.py --list-models                      # 看有哪些模型、多大
    python asr.py --get-model small                  # 只下模型
    python asr.py --file 录音.mp3                    # 识别一个音频文件
    python asr.py --selftest                         # 分段器自测（不用模型）
"""

import argparse
import collections
import io
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import wave

try:
    import numpy as np
except ImportError:
    sys.exit("[缺少依赖] 没装 NumPy。请在终端执行：pip install numpy")

# whisper 只认 16kHz
TARGET_SR = 16000

# ---------------------------------------------------------------- 模型清单
WHISPER_FILES = ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")

# name -> (仓库名, 大约多少 MB)
WHISPER_MODELS = {
    "tiny":   ("Systran/faster-whisper-tiny", 75),
    "base":   ("Systran/faster-whisper-base", 145),
    "small":  ("Systran/faster-whisper-small", 484),
    "medium": ("Systran/faster-whisper-medium", 1530),
    "large-v3": ("Systran/faster-whisper-large-v3", 3090),
}

DEFAULT_MIRROR = "https://hf-mirror.com"
DEFAULT_ROOT = os.path.join(os.path.expanduser("~"), ".cache", "moss_asr")


def model_dir(name, root=None):
    return os.path.join(root or DEFAULT_ROOT, "faster-whisper-%s" % name)


def model_ready(path):
    """4 个文件都在、而且都不是 0 字节，才算下好了。"""
    for f in WHISPER_FILES:
        p = os.path.join(path, f)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            return False
    return True


def fetch_model(name, root=None, mirror=DEFAULT_MIRROR, quiet=False):
    """从镜像把模型 4 个文件拉到本地目录。已经下好就直接返回路径。

    为什么要自己下、不用 huggingface_hub：
      - 国内 huggingface.co 不可达，需要额外设 HF_ENDPOINT 和 HF_HUB_DISABLE_XET
      - 实测 hf-hub 2.2.0 在 Windows 上建快照软链失败，会留下 **0 字节的 model.bin**，
        报错却是 "model.bin is incomplete"，很难查
    自己下就 4 个文件，出问题一眼能看出来。
    """
    if name not in WHISPER_MODELS:
        raise ValueError("没有这个模型：%s（可选：%s）"
                         % (name, "、".join(WHISPER_MODELS)))
    dest = model_dir(name, root)
    if model_ready(dest):
        return dest

    repo = WHISPER_MODELS[name][0]
    os.makedirs(dest, exist_ok=True)
    if not quiet:
        print("[模型] %s 不在本地，从 %s 下载（约 %d MB）"
              % (name, mirror, WHISPER_MODELS[name][1]), file=sys.stderr)

    for f in WHISPER_FILES:
        url = "%s/%s/resolve/main/%s" % (mirror.rstrip("/"), repo, f)
        tmp = os.path.join(dest, f + ".part")
        _download(url, tmp, f, quiet)
        os.replace(tmp, os.path.join(dest, f))     # 原子替换：中断也不会留半个文件
    if not quiet:
        print("[模型] 下好了 → %s" % dest, file=sys.stderr)
    return dest


def _download(url, dest, name, quiet=False):
    t0 = time.time()
    last = [0.0]
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = r.read(262144)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if not quiet and time.time() - last[0] > 0.5:
                        last[0] = time.time()
                        if total:
                            pct = done * 100.0 / total
                            mb = done / 1048576.0
                            print("        %-18s %5.1f%%  %6.1f MB  %.1f MB/s"
                                  % (name, pct, mb, mb / max(time.time() - t0, 1e-6)),
                                  file=sys.stderr, end="\r")
    except Exception:
        try:
            os.remove(dest)
        except OSError:
            pass
        raise
    if not quiet:
        print(" " * 70, end="\r", file=sys.stderr)


# ================================================================
#  1. 把连续音频流切成一"句"
# ================================================================
class _NoiseFloor:
    """自适应噪声底。和 moss_brain.py 里那个同一套思路（低分位数），
    这里单独放一份是为了让 asr.py 能独立运行、独立自测。"""

    def __init__(self, window_seconds=4.0, block_seconds=0.064, percentile=10.0):
        cap = max(8, int(round(window_seconds / max(block_seconds, 1e-3))))
        self.buf = collections.deque(maxlen=cap)
        self.pct = percentile

    def update(self, level_db):
        self.buf.append(level_db)
        if not self.buf:
            return level_db - 40.0
        return float(np.percentile(self.buf, self.pct))


class UtteranceSegmenter:
    """VAD 状态机：喂它一块块音频，它吐出"一整句"。

        silence ──(连续 start_blocks 块有声)──► speech
        speech  ──(连续 end_blocks 块无声)────► 吐出一句，回到 silence
        speech  ──(超过 max_ms)───────────────► 强制吐出（防止有人说个不停把缓冲撑爆）

    两个容易忽略但很关键的细节：
      - **前置缓冲**：确认"开始说话"之前的 300ms 也要算进去，否则第一个字会被切掉
      - **最短时长看的是"有效说话时长"，不是总长度**：前置缓冲（300ms）+ 尾部保留
        （150ms）会把一声 130ms 的咳嗽撑成 450ms 的片段，骗过"总长度≥400ms"的检查。
        所以额外统计**真正有声的块数**，不足 `min_speech_ms`（默认 250ms）的直接丢弃，
        否则咳嗽、关门声都会被送去识别，而 whisper 最爱在这种片段上瞎编。
    """

    def __init__(self, samplerate=TARGET_SR, blocksize=1024,
                 preroll_ms=300.0, start_ms=160.0, end_ms=700.0,
                 min_ms=400.0, max_ms=15000.0, keep_tail_ms=150.0,
                 min_speech_ms=250.0, db_th=-50.0, snr_th=8.0):
        self.sr = samplerate
        self.bs = blocksize
        self.block_s = blocksize / float(samplerate)

        self.preroll_n = max(1, int(round(preroll_ms / 1000.0 / self.block_s)))
        self.start_n = max(1, int(round(start_ms / 1000.0 / self.block_s)))
        self.end_n = max(1, int(round(end_ms / 1000.0 / self.block_s)))
        self.min_n = max(1, int(round(min_ms / 1000.0 / self.block_s)))
        self.max_n = max(1, int(round(max_ms / 1000.0 / self.block_s)))
        self.keep_tail_n = max(0, int(round(keep_tail_ms / 1000.0 / self.block_s)))
        self.min_speech_n = max(1, int(round(min_speech_ms / 1000.0 / self.block_s)))

        self.db_th = db_th
        self.snr_th = snr_th
        self.floor = _NoiseFloor(block_seconds=self.block_s)

        self.ring = collections.deque(maxlen=self.preroll_n)
        self.buf = []
        self.state = "silence"
        self.run_voiced = 0
        self.run_silent = 0
        self.voiced_n = 0          # 这一句里"真正有声"的块数（不含前置缓冲和尾部静音）

        self.stats = {"utterances": 0, "too_short": 0, "forced_cut": 0, "blocks": 0}

    # ---------- 内部 ----------
    @staticmethod
    def level(block):
        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
        return 20.0 * math.log10(rms + 1e-9)

    def reset(self):
        """丢弃正在积累的音频（比如 MOSS 自己开始说话了）。"""
        self.ring.clear()
        self.buf = []
        self.state = "silence"
        self.run_voiced = 0
        self.run_silent = 0
        self.voiced_n = 0

    @property
    def in_speech(self):
        return self.state == "speech"

    # ---------- 对外 ----------
    def feed(self, block, voiced=None):
        """喂一块音频。返回本次产生的完整句子列表（0 个或 1 个，元素是 float32 ndarray）。

        voiced 一般留空让它自己判断；测试时可以显式传 True/False 跳过 VAD。
        """
        self.stats["blocks"] += 1
        block = np.asarray(block, dtype=np.float32).reshape(-1)

        if voiced is None:
            lvl = self.level(block)
            floor = self.floor.update(lvl)
            voiced = (lvl >= self.db_th) and ((lvl - floor) >= self.snr_th)

        out = []

        if self.state == "silence":
            # 环形缓冲一直在存，所以要开始说话时能"回到过去"取前置音频
            self.ring.append(block)
            if voiced:
                self.run_voiced += 1
            else:
                self.run_voiced = 0

            if self.run_voiced >= self.start_n:
                self.state = "speech"
                self.buf = list(self.ring)      # ← 前置缓冲在这里生效
                self.ring.clear()
                self.run_silent = 0
                self.voiced_n = self.run_voiced
        else:                                   # speech
            self.buf.append(block)
            self.run_silent = 0 if voiced else self.run_silent + 1
            if voiced:
                self.voiced_n += 1

            done = False
            if self.run_silent >= self.end_n:
                done = True
            elif len(self.buf) >= self.max_n:
                done = True
                self.stats["forced_cut"] += 1

            if done:
                self.state = "silence"
                self.run_voiced = 0
                # 掐掉尾巴上多余的静音，只留 keep_tail_ms —— 每次省下 500ms 左右，
                # whisper 也少算一点。（"强制切"时 run_silent 是 0，不会误砍）
                if self.run_silent > self.keep_tail_n:
                    drop = self.run_silent - self.keep_tail_n
                    if drop < len(self.buf):
                        del self.buf[-drop:]
                n = len(self.buf)
                if n >= self.min_n and self.voiced_n >= self.min_speech_n:
                    out.append(np.concatenate(self.buf))
                    self.stats["utterances"] += 1
                else:
                    self.stats["too_short"] += 1
                self.buf = []
                self.run_silent = 0
                self.voiced_n = 0

        return out

    def flush(self):
        """收尾时把还在缓冲里的半句话吐出来（可选）。"""
        if (self.state == "speech" and len(self.buf) >= self.min_n
                and self.voiced_n >= self.min_speech_n):
            self.state = "silence"
            self.stats["utterances"] += 1
            seg = np.concatenate(self.buf)
            self.buf = []
            self.voiced_n = 0
            return seg
        self.reset()
        return None


# ================================================================
#  2. 识别引擎
# ================================================================
# whisper 在"没有语音的音频"上会一本正经地胡说，而且全国统一 ——
# 它拿视频字幕训练过，所以静音/噪声常常被听成"谢谢大家""字幕由XX提供"。
# 这些结果直接丢掉，否则 MOSS 会在没人说话时突然开始感谢观众。
# ⚠️ 这张表里混着两类东西，判法完全不同（见 is_hallucination）：
#   • 整句匹配：短词、常见词 —— 它们**可能出现在正常话里**，只能整句比。
#   • 句中匹配：长而独特的套话 —— 正常口语根本不会那么说，出现即是幻觉。
# 原来不分青红皂白统一用 `h in text` 子串匹配，造成了严重误杀（见下）。
HALLUCINATIONS = (
    "谢谢大家", "谢谢观看", "谢谢大家观看", "感谢观看", "感谢大家",
    "字幕组", "订阅", "转发", "打赏", "moss",
)

# 长而独特的字幕套话：句中出现即可判定（正常口语不会这么说）
_PHRASE_MARKERS = ("请不吝点赞", "明镜与点点栏目", "字幕由")

# 网址/字幕残留。用**未去标点的原文**匹配，因为点号是特征的一部分 ——
# 也正因如此不能拿 "com" 去子串匹配："welcome" 里就有 com。
_RAW_MARKERS = ("www.", ".com", ".org", ".net", "http://", "https://",
                "amara.org", "subtitle", "ming pao")

# 字幕署名类幻觉的**形态**。固定词表追不完 —— 实测又冒出过
# "字幕by索兰娅"。但"以字幕打头 + 署名标记"这个形状很稳定。
# 刻意**不**写成 `^字幕`：那样"字幕看不清"这种正常话也会被误杀。
#
# 注意这两条要拿**保留标点**的文本匹配（下面用 flat）：冒号本身就是特征，
# 去标点之后 "字幕：XX制作组" 会变成 "字幕xx制作组"，形状就丢了。
_HALLUCINATION_RE = (
    re.compile(r"^字幕\s*[:：]?\s*(by|由|组|君|菌)"),
    re.compile(r"^字幕\s*[:：]?.{0,6}(组|制作|提供|翻译|校对|压制|出品)"),
)


# 判等前先抹掉标点和空白，只留实词
_PUNCT_RE = re.compile(r"""[\s，。！？、,.!?：:;；~～\-—_…\[\]【】（）()"'"'《》<>*#|]+""")


def _norm_for_match(s):
    return _PUNCT_RE.sub("", s or "").lower()


def is_hallucination(text):
    """整句话是不是一句"幻觉"。

    ⚠️ 必须**整句比对，不能用子串匹配**。这里踩过一个很隐蔽的坑：
    HALLUCINATIONS 里有 "moss"，而原来的写法是 `if h in low: return ""`，
    于是任何**含它自己名字**的短句都会被整句丢掉 ——
    "屏幕上写 MOSS"、"moss 你好"、"MOSS 在吗" 全都听不见。
    症状是"MOSS 听不见你在叫它"，而且只在你说它名字的时候复现，
    极难往"过滤表"上联想。

    判据分三层，各管各的：
      (1) 原文里有网址/字幕标记（".com"、"amara.org"…）——— 正常口语不可能有。
      (2) 去掉标点后**整句**等于表里某句 —— whisper 的幻觉特征是
          "整段输出就是那句话"（没语音时它把训练见过的字幕背出来）。
      (3) 出现长而独特的字幕套话 —— 正常口语不会那么说。
    """
    # (1) 网址/字幕残留：拿原文比（点号是特征）
    raw = (text or "").lower()
    for mk in _RAW_MARKERS:
        if mk in raw:
            return True

    core = _norm_for_match(text)
    if not core:
        return True

    # (2) 整句就是这句幻觉（whisper 的幻觉是"整段输出就是它"）
    for h in HALLUCINATIONS:
        if core == _norm_for_match(h):
            return True

    # (3) 长而独特的字幕套话，出现在句中也是幻觉
    for h in _PHRASE_MARKERS:
        if _norm_for_match(h) in core:
            return True

    # (4) 字幕署名类的形态（词表追不完，按形状判）。
    #     用保留标点的 flat —— 冒号是形状的一部分。
    flat = re.sub(r"\s+", "", (text or "").lower())
    for pat in _HALLUCINATION_RE:
        if pat.match(flat):
            return True
    return False


def clean_text(text, simplify=True):
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return ""
    if simplify:
        try:
            import zhconv
            text = zhconv.convert(text, "zh-cn")     # whisper 中文常出繁体
        except ImportError:
            pass
    if is_hallucination(text):
        return ""
    if not re.search(r"[\w\u4e00-\u9fff]", text):
        return ""
    return text


def to_float32(pcm):
    """把各种输入统一成 float32 一维数组（-1.0 ~ 1.0）。"""
    a = np.asarray(pcm)
    if a.dtype == np.float32:
        return a.reshape(-1)
    if a.dtype == np.int16:
        return (a.reshape(-1).astype(np.float32) / 32768.0)
    if a.dtype == np.int32:
        return (a.reshape(-1).astype(np.float32) / 2147483648.0)
    if a.dtype == np.uint8:
        return ((a.reshape(-1).astype(np.float32) - 128.0) / 128.0)
    return a.reshape(-1).astype(np.float32)


def resample_linear(pcm, sr_in, sr_out=TARGET_SR):
    """线性插值重采样。麦克风基本都是 16k，用不上；但用 48k 的设备时必须有。"""
    if sr_in == sr_out or len(pcm) == 0:
        return pcm.astype(np.float32)
    n_out = int(round(len(pcm) * float(sr_out) / sr_in))
    x_in = np.arange(len(pcm), dtype=np.float64)
    x_out = np.linspace(0, len(pcm) - 1, n_out)
    return np.interp(x_out, x_in, pcm.astype(np.float64)).astype(np.float32)


def pcm_to_wav_bytes(pcm, samplerate=TARGET_SR):
    """float32 → 16bit PCM 的 WAV 字节流（给 API 用）。"""
    data = np.clip(pcm, -1.0, 1.0)
    pcm16 = (data * 32767.0).astype("<i2")
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(samplerate)
        w.writeframes(pcm16.tobytes())
    return bio.getvalue()


class AsrEngine:
    name = "-"

    def transcribe(self, pcm, samplerate=TARGET_SR):
        raise NotImplementedError

    def close(self):
        pass


class NullEngine(AsrEngine):
    """关闭状态。"""

    name = "off"

    def transcribe(self, pcm, samplerate=TARGET_SR):
        return ""


class WhisperEngine(AsrEngine):
    """本地 faster-whisper。第一次用要下模型（见 fetch_model）。"""

    name = "whisper"

    def __init__(self, model="small", root=None, mirror=DEFAULT_MIRROR,
                 device="cpu", compute_type="int8", language="zh",
                 beam_size=5, simplify=True, quiet=False):
        path = fetch_model(model, root, mirror, quiet=quiet)
        from faster_whisper import WhisperModel        # 延迟 import：没装也不影响启动
        t0 = time.time()
        self.model = WhisperModel(path, device=device, compute_type=compute_type)
        self.model_name = model
        self.load_ms = int((time.time() - t0) * 1000)
        self.language = language or None
        self.beam_size = beam_size
        self.simplify = simplify

    def transcribe(self, pcm, samplerate=TARGET_SR):
        audio = to_float32(pcm)
        if samplerate != TARGET_SR:
            audio = resample_linear(audio, samplerate, TARGET_SR)
        if audio.size == 0:
            return ""
        segs, _info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            # 我们自己已经做过 VAD 切句了，再来一层会把开头结尾切没
            vad_filter=False,
            # 关键：不把上一句当上下文。短音频上开着它，whisper 会陷入
            # "复读上一句"的死循环 —— 这是 faster-whisper 最常见的坑。
            condition_on_previous_text=False,
            # 不让它靠"升温"乱猜。短句上降温到 0 更稳、更可复现。
            temperature=0.0,
        )
        text = "".join(s.text for s in segs)
        return clean_text(text, self.simplify)


class ApiEngine(AsrEngine):
    """OpenAI 兼容的 /audio/transcriptions。不用下模型，但要联网、要 Key。

    multipart/form-data 是手写拼的 —— 和 mind.py 一样，能不多一个依赖就不加。
    """

    name = "api"

    def __init__(self, base, key, model="whisper-1", language="zh", timeout=30.0):
        self.url = base.rstrip("/") + "/audio/transcriptions"
        self.key = key
        self.model = model
        self.language = language
        self.timeout = timeout

    def transcribe(self, pcm, samplerate=TARGET_SR):
        audio = to_float32(pcm)
        if samplerate != TARGET_SR:
            audio = resample_linear(audio, samplerate, TARGET_SR)
        if audio.size == 0:
            return ""
        wav = pcm_to_wav_bytes(audio, TARGET_SR)

        boundary = "----moss%d" % int(time.time() * 1000)
        parts = []

        def field(name, value):
            parts.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                          % (boundary, name, value)).encode("utf-8"))

        field("model", self.model)
        if self.language:
            field("language", self.language)
        field("response_format", "json")

        parts.append(("--%s\r\nContent-Disposition: form-data; name=\"file\"; "
                      "filename=\"moss.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
                      % boundary).encode("utf-8"))
        parts.append(wav)
        parts.append(b"\r\n")
        parts.append(("--%s--\r\n" % boundary).encode("utf-8"))
        body = b"".join(parts)

        req = urllib.request.Request(
            self.url, data=body, method="POST",
            headers={
                "Content-Type": "multipart/form-data; boundary=%s" % boundary,
                "Content-Length": str(len(body)),
                "Authorization": "Bearer %s" % self.key,
            })
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            raw = r.read().decode("utf-8", "replace")
        data = json.loads(raw)
        return clean_text((data.get("text") or "").strip())


class AutoEngine(AsrEngine):
    """先试本地 whisper（离线、免费），加载失败再退 API。只切一次。"""

    name = "auto"

    def __init__(self, primary_factory, fallback, on_fallback=None):
        self.primary = primary_factory()
        self.fallback = fallback
        self.on_fallback = on_fallback

    def transcribe(self, pcm, samplerate=TARGET_SR):
        return self.primary.transcribe(pcm, samplerate)

    def close(self):
        self.primary.close()
        if self.fallback:
            self.fallback.close()


def build_engine(kind, opts, on_note=None):
    """按 --asr 造引擎。返回 (engine, 说明)。"""
    kind = (kind or "off").lower()
    note = on_note or (lambda s: print(s, file=sys.stderr))

    if kind in ("off", "none"):
        return NullEngine(), "关闭"

    def make_whisper():
        e = WhisperEngine(model=opts.get("model", "small"),
                          root=opts.get("dir"), mirror=opts.get("mirror", DEFAULT_MIRROR),
                          device=opts.get("device", "cpu"),
                          compute_type=opts.get("compute_type", "int8"),
                          language=opts.get("lang", "zh"),
                          beam_size=opts.get("beam", 5))
        note("[语音识别] 本地 whisper-%s 已加载（%dms）" % (e.model_name, e.load_ms))
        return e

    if kind == "whisper":
        e = make_whisper()
        return e, "本地 whisper-%s" % e.model_name

    if kind == "api":
        key = opts.get("api_key") or ""
        if not key:
            raise RuntimeError("--asr api 需要 --asr-api-key 或环境变量 MOSS_ASR_KEY")
        return ApiEngine(opts["api_base"], key, opts.get("api_model", "whisper-1"),
                         opts.get("lang", "zh")), "API %s" % opts["api_base"]

    if kind == "auto":
        fb = None
        try:
            primary = make_whisper()
        except Exception as e:
            note("[提示] 本地 whisper 起不来（%s），改用 API。" % e)
            key = opts.get("api_key") or os.environ.get("MOSS_ASR_KEY") or ""
            if not key:
                note("[提示] 也没给 --asr-api-key，语音识别关闭。")
                return NullEngine(), "关闭（whisper 与 api 都不可用）"
            a = ApiEngine(opts["api_base"], key, opts.get("api_model", "whisper-1"),
                          opts.get("lang", "zh"))
            return a, "API %s（whisper 不可用）" % opts["api_base"]
        key = opts.get("api_key") or os.environ.get("MOSS_ASR_KEY") or ""
        if key:
            fb = ApiEngine(opts["api_base"], key, opts.get("api_model", "whisper-1"),
                           opts.get("lang", "zh"))
        return AutoEngine(lambda: primary, fb,
                          lambda e: note("[提示] 本地识别出错，改用 API：%s" % e)), \
            "自动（本地 whisper-%s）" % primary.model_name

    raise ValueError("未知的 --asr 值：%s" % kind)


# ================================================================
#  3. 识别线程
# ================================================================
class AsrWorker(threading.Thread):
    """独立线程做识别。

    队列策略：**有限队列 + 丢最旧**。
    这一点和 SpeakWorker（只留最新）不同，因为语义反了：
      - 说话：旧的台词已经过时，念出来是错的 → 只留最新
      - 听话：用户说的每一句都有意义，只是怕积压 → 留几句，实在排不下才丢最旧的
    默认留 2 段。实测 whisper 识别 RTF≈0.05~0.3（比实时快 3~20 倍），
    正常对话根本排不满。
    """

    # 短于这个秒数的段直接判空，不去调引擎（见 _work 里的说明）
    MIN_AUDIO_S = 0.05

    def __init__(self, engine, samplerate=TARGET_SR, queue_size=2,
                 on_error=None, on_result=None, on_empty=None):
        super().__init__(daemon=True, name="asr")
        self.engine = engine
        self.samplerate = samplerate
        self.on_error = on_error
        self.on_result = on_result
        self.on_empty = on_empty
        self.min_audio_s = self.MIN_AUDIO_S

        self.stop_ev = threading.Event()
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.q = collections.deque()
        self.qmax = max(1, int(queue_size))

        self.result = None
        self.seq = 0
        self.busy_flag = False
        self.stats = {"ok": 0, "empty": 0, "dropped": 0, "error": 0, "ms": 0, "audio_s": 0.0}

    # ---------- 外部接口 ----------
    def submit(self, pcm):
        """投递一段音频。返回 True 表示排上了（False = 排不下，被丢了）。"""
        with self.cv:
            self.seq += 1
            dropped = False
            if len(self.q) >= self.qmax:
                self.q.popleft()          # 丢最旧
                self.stats["dropped"] += 1
                dropped = True
            self.q.append((self.seq, np.asarray(pcm, dtype=np.float32)))
            self.cv.notify()
            return not dropped

    def latest(self):
        with self.lock:
            return dict(self.result) if self.result else None

    def busy(self):
        """还有音频在排队 / 正在识别 —— 大脑据此判断"用户这一轮还没结束"。"""
        with self.lock:
            return self.busy_flag or len(self.q) > 0

    def pending_count(self):
        with self.lock:
            return len(self.q)

    def shutdown(self):
        self.stop_ev.set()
        with self.cv:
            self.cv.notify_all()

    # ---------- 线程主体 ----------
    def run(self):
        while not self.stop_ev.is_set():
            with self.cv:
                while not self.q and not self.stop_ev.is_set():
                    self.cv.wait(0.1)
                if not self.q:
                    continue
                seq, pcm = self.q.popleft()
            self._work(seq, pcm)
        # 收尾：把队列里剩下的识别完，用户最后一句话别丢了
        while self.q:
            with self.cv:
                if not self.q:
                    break
                seq, pcm = self.q.popleft()
            self._work(seq, pcm)

    def _work(self, seq, pcm):
        with self.lock:
            self.busy_flag = True
        dur = len(pcm) / float(self.samplerate)

        # 空/极短的段直接判空，别去打网络或跑模型 —— 白白花一次请求，
        # 而且结果必然是空串（WhisperEngine/ApiSpeechEngine 里都有
        # `if audio.size == 0: return ""` 这条早退）。
        # 实测会碰上：分段器在静音↔说话的边界上偶尔吐出一段没有样本的"句子"。
        if dur < self.min_audio_s:
            with self.lock:
                self.stats["empty"] += 1
                self.busy_flag = False
                self.result = {"seq": seq, "text": "", "ok": True, "err": "",
                               "ms": 0, "audio_s": round(dur, 2), "rtf": 0.0,
                               "engine": getattr(self.engine, "name", "asr")}
            if self.on_empty:
                self.on_empty(dur)
            return

        t0 = time.time()
        try:
            text = self.engine.transcribe(pcm, self.samplerate)
            ok, err = True, ""
        except Exception as e:
            text, ok, err = "", False, "%s" % e
            if self.on_error:
                self.on_error(e)
        ms = int((time.time() - t0) * 1000)

        rec = {"seq": seq, "text": text, "ok": ok, "err": err,
               "ms": ms, "audio_s": round(dur, 2),
               "rtf": round(ms / max(dur * 1000.0, 1.0), 3),
               "engine": getattr(self.engine, "name", "asr")}

        with self.lock:
            if ok and text:
                self.stats["ok"] += 1
                self.stats["ms"] += ms
                self.stats["audio_s"] += dur
            elif ok:
                self.stats["empty"] += 1
            else:
                self.stats["error"] += 1
            self.result = rec
            self.busy_flag = False

        if self.on_result and ok and text:
            self.on_result(rec)


# ================================================================
#  4. 音频文件解码（给自测/回放用，靠 PyAV —— faster-whisper 已经带了它）
# ================================================================
def decode_audio_file(path, samplerate=TARGET_SR):
    """读 mp3/wav/m4a/... → 16kHz 单声道 float32。

    用 PyAV 而不是自己解 mp3：PyAV 是 faster-whisper 的依赖，装它的时候就自带了，
    而且它内置 ffmpeg，什么格式都能读。**没有它也不影响主流程**（主流程吃麦克风 PCM）。
    """
    try:
        import av
    except ImportError:
        raise RuntimeError("读音频文件需要 PyAV（pip install av）。"
                           "主流程用麦克风时不需要它。")
    with av.open(path) as c:
        rs = av.AudioResampler(format="fltp", layout="mono", rate=samplerate)
        chunks = []
        for fr in c.decode(audio=0):
            for out in rs.resample(fr):
                chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


# ================================================================
#  5. 自检 / 命令行
# ================================================================
def _selftest_segmenter():
    """纯逻辑自测：造一段"说话-停顿-说话"的信号，看分段器切得对不对。
    不需要模型、不需要联网。"""
    sr, bs = TARGET_SR, 1024
    seg = UtteranceSegmenter(sr, bs, end_ms=700.0, min_ms=400.0, start_ms=160.0)
    rng = np.random.default_rng(7)

    def quiet(n):
        return [rng.normal(0, 10 ** (-70 / 20.0), bs).astype(np.float32) for _ in range(n)]

    def talk(n):
        # 用固定幅度的噪声冒充语音：对 VAD 来说和真人说话没区别（都要看电平和信噪比）
        return [(rng.normal(0, 1, bs) * 0.12).astype(np.float32) for _ in range(n)]

    print("[自测] 造信号：静音0.83s + 说话1.22s + 静音1.02s + 说话0.58s + 静音1.02s")
    blocks = quiet(13) + talk(19) + quiet(16) + talk(9) + quiet(16)
    got = []
    for b in blocks:
        for utt in seg.feed(b):
            got.append(len(utt) / float(sr))

    print("       切出 %d 段，时长 %s 秒" % (len(got), ["%.2f" % x for x in got]))
    ok = len(got) == 2 and 1.3 <= got[0] <= 1.8 and 0.7 <= got[1] <= 1.2
    # 前置缓冲 300ms + 说话 1.2s ≈ 1.5s；第二段 300+600 ≈ 0.9s
    print("       → %s（两段，且都含 300ms 前置缓冲）" % ("正确" if ok else "不对"))
    print("       stats = %s" % seg.stats)

    # 极短噪声不该成句
    seg.reset()
    short = 0
    for b in quiet(13) + talk(2) + quiet(16):
        short += len(seg.feed(b))
    print("[自测] 2 块（约 130ms）的短噪声 → 切出 %d 段（应为 0，否则咳嗽会被送去识别）"
          % short)
    return 0 if (ok and short == 0) else 1


def _selftest_clean_text():
    """纯逻辑自测：假话要丢掉，真话一句都不能误杀。"""
    print()
    print("[自测] 幻觉过滤（clean_text）")
    # 该丢的
    bad = ["谢谢大家。", "谢谢观看", "字幕由 Amara.org 提供", "请不吝点赞订阅转发打赏",
           "字幕by索兰娅", "字幕：XX制作组", "  ", "", "。。。"]
    # 一句都不能丢的 —— 尤其那些**含 "moss"** 的正常话
    good = ["屏幕上写 MOSS", "MOSS 你好", "moss 在吗", "把灯变成红色",
            "你好 MOSS，现在几点了", "我订阅了这个频道", "帮我记住，我叫张三",
            # 下面这些一被误杀就说明"去标点后子串匹配"的老毛病又回来了
            "welcome", "comfort zone", "他打了个赏", "请转发给同事",
            # 跟"字幕"沾边但不该被丢的
            "字幕看不清", "这个视频有字幕吗"]
    bad_ok = all(clean_text(t) == "" for t in bad)
    good_ok = all(clean_text(t) != "" for t in good)

    for t, ok in [(t, clean_text(t) == "") for t in bad]:
        print("       该丢 %-28r → %s" % (t, "丢了 ✓" if ok else "没丢 ✗"))
    for t, ok in [(t, clean_text(t) != "") for t in good]:
        print("       该留 %-28r → %s" % (t, "留了 ✓" if ok else "**被误杀 ✗**"))
    print("       → %s" % ("正确" if (bad_ok and good_ok) else "不对"))
    return 0 if (bad_ok and good_ok) else 1


def _main():
    p = argparse.ArgumentParser(description="MOSS 听觉理解层（ASR）")
    p.add_argument("--file", default=None, help="识别一个音频文件（mp3/wav/m4a…）")
    p.add_argument("--engine", default="whisper", choices=["whisper", "api", "off"])
    p.add_argument("--model", default="small", help="whisper 模型：tiny/base/small/medium")
    p.add_argument("--dir", default=None, help="模型存放目录，默认 ~/.cache/moss_asr")
    p.add_argument("--mirror", default=DEFAULT_MIRROR, help="模型下载镜像")
    p.add_argument("--device", default="cpu", help="cpu 或 cuda")
    p.add_argument("--compute-type", default="int8")
    p.add_argument("--lang", default="zh", help="语言，zh / en / auto(留空)")
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--no-simplify", action="store_true", help="不做繁转简")
    p.add_argument("--api-base", default="https://api.openai.com/v1")
    p.add_argument("--api-key", default=None)
    p.add_argument("--api-model", default="whisper-1")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--get-model", default=None, help="只下载某个模型然后退出")
    p.add_argument("--selftest", action="store_true", help="分段器自测（不需要模型）")
    a = p.parse_args()

    if a.list_models:
        print("%-10s %-42s %s" % ("名字", "仓库", "约多少 MB"))
        for n, (repo, mb) in WHISPER_MODELS.items():
            mark = "  [已下载]" if model_ready(model_dir(n, a.dir)) else ""
            print("%-10s %-42s %5d%s" % (n, repo, mb, mark))
        print("\n存放目录：%s" % (a.dir or DEFAULT_ROOT))
        return 0

    if a.get_model:
        fetch_model(a.get_model, a.dir, a.mirror)
        return 0

    if a.selftest:
        # 两个都不需要模型、不需要联网
        rc1 = _selftest_segmenter()
        rc2 = _selftest_clean_text()
        return 0 if (rc1 == 0 and rc2 == 0) else 1

    if not a.file:
        p.print_help()
        return 2

    if not os.path.exists(a.file):
        print("[错误] 找不到文件：%s" % a.file, file=sys.stderr)
        return 2

    pcm = decode_audio_file(a.file)
    print("[文件] %s  时长 %.2f 秒" % (a.file, len(pcm) / float(TARGET_SR)))

    if a.engine == "off":
        print("[提示] --engine off，不识别。")
        return 0

    opts = {"model": a.model, "dir": a.dir, "mirror": a.mirror, "device": a.device,
            "compute_type": a.compute_type, "lang": a.lang, "beam": a.beam,
            "api_base": a.api_base, "api_key": a.api_key, "api_model": a.api_model}
    engine, desc = build_engine(a.engine, opts)
    print("[引擎] %s" % desc)

    t0 = time.time()
    text = engine.transcribe(pcm, TARGET_SR)
    dt = time.time() - t0
    print("[识别] %.2fs（RTF %.3f）" % (dt, dt / max(len(pcm) / float(TARGET_SR), 1e-6)))
    print("[结果] %s" % (text or "（空）"))
    engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(_main())
