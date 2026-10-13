#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOSS 回声消除（AEC）  ·  流浪地球 550W 复刻项目 · 阶段 17
============================================================
阶段 12 的 MOSS 有个很难受的毛病：**它一开口就聋了**。

原因在 asr.py 开头写过 —— 音箱放出来的声音会被自己的麦克风收进去，
识别成"用户在说话"，再当成新问题回给自己，无限自我对话。
当时的解法是**半双工**（对讲机模式）：说话期间把麦克风按静音算。

代价你现在能直接体会到：**它念台词的时候你插不进话**。
你说"停"，它还在念；你说"不对，是红色"，它念完才轮到听。
这不是对话，是轮流发言。

正确的解法只有一个：**从麦克风信号里，把自己刚才放出去的那份减掉**。
这就是 AEC（Acoustic Echo Cancellation）。手机免提、视频会议全靠它。

AEC 到底在干什么
------------------------------------------------------------
房间是一个**线性系统**：扬声器发出的信号 `x`，经过空气、桌子、墙壁的
反射叠加，到达麦克风时变成 `y = h * x`（`h` 是房间冲激响应，`*` 是卷积）。

麦克风实际收到的 `d = y + s + n`：
    y = 自己的回声（要去掉的）
    s = 用户真正在说的话（要留下的）
    n = 环境噪声

既然 `h` 未知，那就**边听边学**：用一个自适应滤波器 `w` 去逼近 `h`，
用 `y_hat = w * x` 估出回声，输出 `e = d - y_hat`。
`w` 收敛到 `h` 之后，`e ≈ s + n` —— 剩下的就是用户的声音。

滤波器用 NLMS（归一化最小均方）更新，这是自适应滤波里最皮实的一种：
    w ← w + μ · (x ⊗ e) / ‖x‖²
`μ` 是步长。步长大收敛快但会抖，步长小稳但慢。

三个必须处理的工程问题
------------------------------------------------------------
**(1) 时间对齐 —— 这一步比算法本身难**

要让 `d - y_hat` 真的减得掉，`x` 和 `d` 必须在**同一根时间轴上**。
但我们不知道扬声器缓冲、声卡缓冲、空气传播加起来是多少毫秒。

这里不猜、也不做显式的时延估计，而是**把时延交给滤波器自己吸收**：
滤波器长度取 200ms 出头，远大于可能的时延（通常 20~120ms）。
于是 `w` 里"开头一段接近 0 + 后面一段是真实回声路径"，
时延自然被学进零抽头里。鲁棒，代价是滤波器长一点。

真正难的是**"麦克风这一刻听到的，对应扬声器哪一刻放的"**。
这靠 `ReferenceBuffer` 解决：播放侧把"这段 PCM 是从 t_play 开始放的"
登记进去，缓冲只对外暴露**已经播出去的那部分**（未来的样本一律当 0）。

一个反直觉的结论写在这里，免得以后又绕回来：
**不需要去问播放器"现在播到第几毫秒了"**。
声卡的时钟精度是 ppm 级，3 秒的句子累计漂移不到 1ms；而"开始播"
这个动作到"真正出声"之间的固定延迟是个常数，被滤波器当成时延吸收掉。
反而是频繁查询播放位置会引入 20ms 级的抖动 —— 那才是滤波器学不会的东西。

**(2) 双讲（double-talk）—— 最容易把滤波器搞崩的地方**

你插话的瞬间，麦克风里是"自己的回声 + 你的声音"。这时候如果还在更新 `w`，
NLMS 会以为"误差这么大，说明回声路径错了"，把好不容易学到的 `h` 改坏。
一改坏，回声漏得更厉害，误差更大 → **发散**。

所以在双讲期间**冻结更新**，只减不学。

判据用最简单也最不容易在冷启动时翻车的一种：
    麦克风能量 > 阈值 × 参考能量  →  认为有人插话
为什么不用"回声估计能量"来判：冷启动时 `w=0`，回声估计恒为 0，
任何声音都像"插话"，于是**永远冻结、永远学不会** —— 死锁。
用参考能量就绕开了这个鸡生蛋问题。

**(3) 残余回声 —— 学不完的那部分**

`w` 收敛后也总有一点减不干净（房间是非线性/时变的，滤波器长度有限）。
而且双讲期间冻结更新，漏音会回升。
所以最后再挂一级**残余抑制**：确认"只有远端在响"时，把输出压掉若干 dB。
代价是这段里用户的声音也会被压 —— 但那时本来就没有用户在说话。

为什么不用现成的库
------------------------------------------------------------
`webrtc-audio-processing` / `speexdsp` 是更成熟的选择，但：
  - Windows 上基本没有可用的 wheel，要自己编译 C++；
  - 装不上就是"整条链路起不来"，违背本项目"离线、可验证、零折腾"的原则；
  - 最要紧的是：**它们的内部状态是黑盒，没法在这个项目里做确定性的自检**。

所以这里用 NumPy 自己写。代价是不如 WebRTC 成熟（尤其是非线性处理），
好处是每一行都能测：`--selftest` 会造一个假的房间、量出真实的 ERLE（dB）。

用法
------------------------------------------------------------
    python aec.py --selftest          # 全套自检（合成房间，不用麦克风）
    python aec.py --bench             # 看一眼每块音频的开销

接进主程序：`python moss_brain.py --aec on ...`（见 README 第六节）
"""

import argparse
import collections
import math
import os
import sys
import threading
import time

import numpy as np

# 滤波器默认长度。250ms @16k = 4000 抽头。
#
# 长度怎么定的 —— 实测的抽头数 × 设备时延 二维表（8 秒收敛，ERLE dB）：
#
#     时延 \ 抽头    150ms   200ms   250ms   300ms
#        20ms        18.7    18.0    16.9    16.3
#        55ms        17.3    17.6    16.9    15.7
#        90ms         4.7    15.1    18.0    17.1
#       130ms         2.1     5.3    17.8    16.7
#
# 看那两行断崖：抽头一旦覆盖不住"时延 + 混响"，ERLE 不是慢慢降，是**塌掉**。
# 200ms 在 55ms 时延下比 250ms 还高 0.7dB，但时延一到 90ms 就跌到 15dB。
# 而设备时延恰恰是这个项目里**最不可控**的东西（声卡驱动、蓝牙、USB 各不相同）。
#
# 所以选 250ms：用 1 dB 的峰值换掉整条断崖。多出来的那点收敛慢，
# 靠频域的逐 bin 归一化早就赚回来了。
DEFAULT_TAPS_MS = 250.0

# 内部分块长度（样本）。每块做一次 NLMS 更新。
# 64 样本 @16k = 4ms —— 比麦克风块（默认 1024 样本 / 64ms）细得多，
# 这样滤波器在 64ms 里能更新 16 次，收敛快得多，而总计算量不变。
DEFAULT_SUBBLOCK = 64

# 「时间轴跳变」的判定门槛（秒）。详见 process() 里那段注释 ——
# 简单说：声卡回调给的时间戳本身有 ±10ms 抖动，比子块步长（4ms）还大，
# 门槛按子块长取（12ms）会被正常抖动打穿，于是每几块就误清一次参考历史，
# 滤波器收敛反复从头开始。真正的跳变（丢块/卡顿/换音源）至少是一个音频块。
GAP_TOL_S = 0.03


# ================================================================
#  1. 参考信号缓冲：扬声器"此刻在放什么"
# ================================================================
class _Segment:
    """一段正在播（或播过）的音频，带它自己的时间轴。

    时间轴的定义是整个对齐的基础，写清楚：
        pcm 里的第 n 个样本，**理想的**播放时刻是 t0 + n / sr。
    实际出声会晚一个固定量（声卡缓冲），但只要它是常数，
    就会被自适应滤波器当成时延吸收掉，不影响减掉回声。
    """

    __slots__ = ("t0", "pcm", "sr", "taxis", "limit")

    def __init__(self, t0, pcm, sr):
        self.t0 = float(t0)
        self.pcm = pcm
        self.sr = int(sr)
        self.taxis = self.t0 + np.arange(len(pcm), dtype=np.float64) / float(self.sr)
        # 这个片段最多允许被暴露到哪个时刻。
        # 中途被打断时会被截断到"真正播到的地方"，
        # 否则缓冲会以为后面还在响，把不存在的回声也"减"掉。
        self.limit = float("inf")


class ReferenceBuffer:
    """"扬声器在放什么"的时间轴缓冲。线程安全。

    播放侧（SpeakWorker 线程）登记片段，采集侧（AudioWorker 线程）按时间取。
    两边是不同线程，所以内部上锁；但锁的持有时间都是微秒级，不会互相拖死。
    """

    def __init__(self, sr, seconds=4.0, keep=4):
        self.sr = int(sr)
        self.keep = max(1, int(keep))
        self.lock = threading.Lock()
        self.segments = collections.deque(maxlen=self.keep)
        self.n_start = 0
        self.n_window = 0

    # ---------- 播放侧 ----------
    def start(self, t_play, pcm, sr=None):
        """登记一段即将播出的音频。返回登记成功的样本数。"""
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if pcm.size == 0:
            return 0
        sr = int(sr or self.sr)
        if sr != self.sr:
            pcm = resample_linear(pcm, sr, self.sr)
        with self.lock:
            self.segments.append(_Segment(t_play, pcm, self.sr))
            self.n_start += 1
        return pcm.size

    def finish(self, t_end=None):
        """播放结束（含被中途打断）：把当前片段的时间轴截到 t_end。

        没有这一步的话，被打断的句子会继续"虚拟地播下去"，
        AEC 会拿一段根本不存在的声音去减，反而把用户的下一句削掉。
        """
        t_end = time.time() if t_end is None else float(t_end)
        with self.lock:
            if self.segments:
                seg = self.segments[-1]
                if t_end < seg.limit:
                    seg.limit = t_end

    def clear(self):
        with self.lock:
            self.segments.clear()

    @property
    def active(self):
        """此刻手上有没有"正在播（或刚播完）"的参考信号。

        这个是**半双工守卫的判据**，比它看起来重要：
        主程序用它来回答"MOSS 正在说话，而我知道它在放什么吗？"
          - 知道 → 交给 AEC 去减，麦克风保持打开（**全双工**）
          - 不知道 → 退回老办法，说话期间屏蔽麦克风（半双工）

        为什么要有这条：参考信号可能拿不到 —— SAPI 合成到文件失败、
        edge 的 mp3 解码失败、临时文件权限问题……这时候如果还傻乎乎地
        开着麦克风，MOSS 就会听见自己 → 自己回答自己 → 无限自我对话。
        那正是阶段 11/12 花大力气堵掉的坑。所以这里给了个兜底：
        **拿不到参考就不是"全双工"，只是"退化回半双工"**，功能不丢。

        上界的取法：标称时长内一律算 active；finish() 会把 limit 截到真实停下的时刻。
        """
        now = time.time()
        with self.lock:
            for seg in reversed(self.segments):
                end = min(seg.limit, seg.t0 + len(seg.pcm) / float(seg.sr))
                if seg.t0 - 0.05 <= now <= end + 0.25:
                    return True
        return False

    # ---------- 采集侧 ----------
    def window(self, t_end, nsamp, now=None):
        """取"结束于 t_end 的那 nsamp 个样本"。

        返回数组是**时间升序**的：`out[-1]` 对应时刻 t_end，
        `out[0]` 对应 t_end - (nsamp-1)/sr。
        还没播到的时刻（> now 或 > 片段的 limit）一律填 0 ——
        对 AEC 来说"还没放"和"放的是静音"是一回事。
        """
        nsamp = int(nsamp)
        if nsamp <= 0:
            return np.zeros(0, dtype=np.float32)
        now = time.time() if now is None else float(now)
        times = t_end - np.arange(nsamp - 1, -1, -1, dtype=np.float64) / float(self.sr)

        out = np.zeros(nsamp, dtype=np.float32)
        with self.lock:
            segs = list(self.segments)
            self.n_window += 1

        hi = nsamp                     # 还没被填的最高下标（左闭右开）
        for seg in reversed(segs):     # 新的先填，然后老片段补下面那段
            end = min(now, seg.limit)
            if end <= seg.t0:
                continue
            i0 = min(int(np.searchsorted(times, seg.t0, "left")), hi)
            i1 = min(int(np.searchsorted(times, end, "right")), hi)
            if i1 > i0:
                out[i0:i1] = np.interp(times[i0:i1], seg.taxis, seg.pcm)
                hi = i0
            if hi <= 0:
                break
        return out

    def describe(self):
        with self.lock:
            n = len(self.segments)
            total = sum(len(s.pcm) for s in self.segments) / float(self.sr)
        return "参考缓冲 %d 段 / 共 %.1fs" % (n, total)


def resample_linear(pcm, sr_from, sr_to):
    """线性重采样。只在采样率不一致时用。

    为什么敢用这么糙的重采样：AEC 的参考信号不需要"好听"，只需要
    **和真实回声线性相关**。线性插值引入的高频镜像很小，
    而且它本身也是线性的 —— 会被自适应滤波器一起学进去。
    （真正影响的是"参考带宽不足"，那由重采样前的一句交代解决：
    TTS 是 24kHz、麦克风是 16kHz，属于降采样，不会丢信息。）
    """
    if sr_from == sr_to or len(pcm) == 0:
        return np.asarray(pcm, dtype=np.float32)
    n_out = max(1, int(round(len(pcm) * sr_to / float(sr_from))))
    src = np.arange(len(pcm), dtype=np.float64) / float(sr_from)
    dst = np.arange(n_out, dtype=np.float64) / float(sr_to)
    return np.interp(dst, src, pcm).astype(np.float32)


# ================================================================
#  2. 解码：把即将播出的音频变成参考 PCM
# ================================================================
def decode_reference(path, target_sr=16000):
    """把一个音频文件解码成单声道 float32 PCM（目标采样率）。

    为什么非得解码：MCI 只负责"放"，不告诉我们"放的是什么波形"。
    而 AEC 恰恰需要那个波形。edge-tts 给的是 mp3，所以要解一次。

    用 PyAV：它本来就是 faster-whisper 的依赖，已经在环境里；
    而且它内置 libswresample，重采样质量比手写插值好得多
    —— 参考信号越接近真实回声，能减掉的越多。
    """
    try:
        import av                                    # 延迟导入
    except ImportError:
        return None

    frames = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=int(target_sr))
        for frame in container.decode(stream):
            for out in _resample(resampler, frame):
                frames.append(out.to_ndarray().reshape(-1))
        for out in _resample(resampler, None):       # 冲干净重采样器里剩的尾巴
            frames.append(out.to_ndarray().reshape(-1))

    if not frames:
        return None
    pcm = np.concatenate(frames).astype(np.float32)
    return pcm if pcm.size else None


def _resample(resampler, frame):
    """PyAV 各个版本的 resample() 返回值不一样（有时是帧，有时是列表）。抹平它。"""
    try:
        out = resampler.resample(frame)
    except Exception:
        return []
    if out is None:
        return []
    return list(out) if isinstance(out, (list, tuple)) else [out]


# ================================================================
#  3. 回声消除本体
# ================================================================
class EchoCanceller:
    """自适应回声消除器。两种更新方式，默认频域。

    一次 `process()` 处理一个麦克风块（默认 1024 样本 / 64ms），
    内部切成 `subblock`（默认 64 样本 / 4ms）小块逐次更新。

    ── 为什么默认不是时域 NLMS ──────────────────────────────
    时域 NLMS 的收敛速度取决于输入自相关矩阵的**特征值散布**。
    白噪声散布小，收敛快；而语音是强有色信号（低频能量比高频高十几个 dB），
    散布极大 —— 实测同样 6 秒，白噪声能压 30dB，语音只有 9dB。

    频域（PBFDAF，分块频域自适应滤波）把这件事解决了：它在**每个频点
    单独归一化步长**。低频 bin 功率大、步长自动变小，高频 bin 功率小、
    步长自动变大 —— 等效于把输入白化了，收敛速度回到接近最优。

        时域 20 秒 → 15 dB ；频域 6 秒 → 20 dB 以上（实测见 --selftest）

    ── 一个让开销变得可以忽略的巧劲 ──────────────────────────
    PBFDAF 有 M 个分块（比如 63 个），朴素实现每帧要做 M 次 FFT。
    但观察一下：第 k 帧的第 p 块，用的是参考信号的第 (k-p-1, k-p) 两段；
    而第 k-1 帧的第 p-1 块用的正是同样两段。也就是说：

        Uf[k][p] == Uf[k-1][p-1]

    所以每帧只需要算 **1 次** FFT（最新那一块），其余整体右移即可。
    开销从 M 次 FFT 降到 1 次，62 次复数乘加成了大头 —— 依然很便宜。

    ── 时域实现保留着 ────────────────────────────────────────
    `mode="td"` 还能用。它的价值不在性能，而在**对照**：
    自检里两条路径跑同一个假房间，能立刻看出"问题出在算法还是出在别处"。
    """

    def __init__(self, sr=16000, taps_ms=DEFAULT_TAPS_MS, subblock=DEFAULT_SUBBLOCK,
                 mu=None, dtd_ratio=2.0, resid_db=10.0, adapt=True, mode="fd",
                 leak=0.9997, ref_floor_db=-60.0, log=None):
        self.sr = int(sr)
        self.subblock = int(subblock)
        self.mode = mode
        # 抽头数取 subblock 的整数倍，省掉边界判断
        taps = max(self.subblock, int(round(taps_ms / 1000.0 * self.sr)))
        self.L = (taps + self.subblock - 1) // self.subblock * self.subblock
        self.taps_ms = self.L / float(self.sr) * 1000.0

        # 两种模式的步长含义不同（时域按整块能量归一，频域按 bin 归一），
        # 所以各给各的默认值 —— 混用一个数字会让人调不明白。
        self.mu = float(mu) if mu is not None else (0.5 if mode == "fd" else 0.9)
        # 泄漏因子：每一步把系数乘掉一点点。防的是"输入长期不激励某个频点，
        # 那里的系数被噪声慢慢顶到很大"。0.9997 在 250 帧/秒下时间常数约 13 秒。
        self.leak = float(leak)
        self.dtd_ratio = float(dtd_ratio)
        # 细判据（"麦克风比回声估计大好几倍 → 有人"）只有在滤波器**确实收敛过**
        # 之后才敢用。怎么判"收敛过"：看 `rho = 残差 / 麦克风`。
        # rho 越小说明抵消得越好；有一阵子压到 conv_rho 以下就置位，之后不再复位
        # （除非发散复位）。理由见 _dtd 的注释。
        #
        # 门槛取 0.3（≈10dB）。这一条是实测调出来的，不是拍的：
        # 一开始取 0.5（6dB），结果滤波器刚压掉 8dB 就被认定"收敛了"，
        # 而那时 `echo_est` 还比真回声低十几 dB，于是 `mic > 2×echo_est`
        # 一路误判成双讲 → 闸门大开 → 自己的残差被当成用户（S4 翻车）。
        # 用 EMA（而不是取最小值）也是同一个原因：单个瞬间的低残差
        # 不能当作"收敛了"，得**持续**低下来才算数。
        self.conv_rho = 0.3
        self.rho = None                # None = 还没测过
        self.converged = False
        self.n_warm = 0                # 预热期（还没收敛）被按"只有回声"处理的块数
        # 「时间轴跳变」的判定门槛（秒）。取"真正的跳变有多长"的量级：
        # 丢块/卡顿/换音源都 ≥ 一个音频块（几十毫秒），而声卡回调查时间戳的
        # 抖动只有 ±10ms。再和子块长取大，免得子块调大时误判（见 process 里的注释）。
        self.gap_tol = max(GAP_TOL_S, 3.0 * self.subblock / float(self.sr))
        # 余音保护：扬声器停了以后，房间混响还会响一阵。这段时间照样按
        # "有回声"对待（继续抑制、继续关 VAD 的门），否则尾巴会被当成用户。
        # 时长取滤波器长度 —— 它本来就该覆盖"直达声 + 混响"，用它当余音上界最自然。
        self.tail_s = self.L / float(self.sr)
        self._last_far_t = None
        self.n_tail = 0                # 余音期（远端已停、但还按有回声处理）的块数
        self.resid_gain = 10.0 ** (-abs(float(resid_db)) / 20.0)
        self.adapt = bool(adapt)
        self.ref_floor = 10.0 ** (ref_floor_db / 20.0)
        self.log = log
        self.eps = 1e-9

        self.B = self.subblock
        if mode == "fd":
            self.N = 2 * self.B
            self.nbins = self.N // 2 + 1
            self.M = self.L // self.B                      # 分块数
            self.Wf = np.zeros((self.M, self.nbins), dtype=np.complex64)
            self.Uf = np.zeros((self.M, self.nbins), dtype=np.complex64)
            self.P = np.full(self.nbins, 1e-8, dtype=np.float64)   # 逐 bin 功率
            self.smooth = 0.7                              # 功率估计的平滑系数
            self.p_floor = 1e-3                            # 功率的相对地板
            self._p_ready = False
            self.prev_blk = np.zeros(self.B, dtype=np.float32)
        else:
            # 时域：滤波器**反向存**（wr[k] 对应 w[L-1-k]），
            # 这样前向卷积和梯度各是一句 np.correlate，不用铺 (B×L) 的大矩阵。
            self.wr = np.zeros(self.L, dtype=np.float32)

        # 内部样本时钟的"上一小块末尾时刻"。None = 还没对过表，用声卡的时间戳起头。
        # 之后每一块都按 `上一块末尾 + 块时长` 推进，不再看时间戳 —— 详见 process()。
        self._last_t = None
        self._frame = np.zeros(self.N if mode == "fd" else 0, dtype=np.float32)

        # 统计
        self.n_frames = 0
        self.n_sub = 0
        self.n_adapt = 0
        self.n_freeze = 0
        self.n_dtd = 0
        self.n_resid = 0
        self.n_cold = 0
        self.n_reset = 0
        self.n_gap = 0
        self._erle_sum = 0.0
        self._erle_n = 0
        self.erle_db = 0.0            # 只有"纯远端"的块才有意义
        self.last = {"erle_db": 0.0, "dtd": False, "adapted": False,
                     "ref_db": -99.0, "mic_db": -99.0, "out_db": -99.0}

    # ---------- 对外 ----------
    @property
    def ready(self):
        """滤波器是否已经「学到东西」（用于告诉用户 AEC 到底有没有在工作）。"""
        if self.mode == "fd":
            return float(np.sum(np.abs(self.Wf) ** 2)) > 1e-6
        return float(np.dot(self.wr, self.wr)) > 1e-6

    def reset(self):
        if self.mode == "fd":
            self.Wf[:] = 0.0
            self.Uf[:] = 0.0
            self.prev_blk[:] = 0.0
            self.P[:] = 1e-8
            self._p_ready = False
        else:
            self.wr[:] = 0.0
        self._last_t = None
        self.n_frames = self.n_sub = self.n_adapt = self.n_freeze = 0
        self.n_dtd = self.n_resid = self.n_cold = self.n_reset = self.n_gap = 0
        self.n_warm = 0
        self.n_tail = 0
        self._last_far_t = None
        self.rho = None
        self.converged = False
        self._erle_sum = 0.0
        self._erle_n = 0
        self.erle_db = 0.0

    def process(self, mic_block, t_end=None, ref=None, now=None):
        """处理一个麦克风块。

        mic_block : float32 数组，本block的音频（时间上"最后"的样本是 t_end）
        t_end     : 本块**最后一个样本**对应的时刻（不传就用当前时间）
        ref       : ReferenceBuffer；不传就整个直通（等价于关掉 AEC）
        返回 (处理后的块, info)

        ⚠️ `t_end` 只当**第一次对表**用，之后内部按样本数自己走（见下面那段长注释）。
        所以调用方给的时间戳抖一点没关系；但**别给错**（比如队列积压时拿最新块的
        时刻配老块音频），错得超过 gap_tol 会被当成"时间轴跳变"，白清一次参考历史。
        """
        mic = np.asarray(mic_block, dtype=np.float32).reshape(-1)
        if ref is None or mic.size == 0:
            return mic, dict(self.last, passthrough=True)

        now = time.time() if now is None else float(now)
        t_end = now if t_end is None else float(t_end)
        # 时间戳跑到"现在"后面去了（回调早于音频内容、或时钟微倒挂）→ 拉回来。
        # 这一刀只会动 t_end 最多一个抖动幅度（±10ms），远小于 gap_tol（30ms），
        # 所以不会把正常的抖动放大成一次假跳变。
        if t_end > now:
            t_end = now

        B = self.subblock
        n = mic.size
        dur = n / float(self.sr)
        out = np.empty(n, dtype=np.float32)
        last = None

        # ---------- 时间轴：自己养一条**样本驱动**的时钟 ----------
        # ⚠️ 这一整段是阶段 17 收尾时补的，起因就是自检里那条 [6/7]「时间戳抖动」：
        # 抖动 ±8ms（真机 MME 实测就是这个量级）时 ERLE 从 23dB 掉到 8dB。
        # 根因不在滤波器，在**时间轴**。声卡回调给的时间戳抖动 ±10ms，
        # 而 t_sub 既要判跳变、又要拿去 `ref.window(t_sub, ...)` 取参考 ——
        # 时间戳抖多少，取到的参考就整体错位多少。±8ms 就是 ±128 个样本，
        # 参考和麦克风错开 128 个样本 = 彻底去相关，滤波器根本无从下手。
        #
        # 关键认识：**抖的只是时间戳，音频本身是连续的**。声卡每块交上来 n 个样本，
        # 它们的间隔严格等于 1/sr，一分不差；抖的是"回调什么时候被执行"。
        # 所以正确做法是：锚定一次之后，就按**样本数**推进内部时钟，
        # 完全不理后面每一块的时间戳；时间戳只留着"对表" ——
        # 差得离谱（> gap_tol）才认账，那是真丢块 / 卡顿 / 换设备，不是抖动。
        #
        # 收益不止抗抖动：跨块那一步从「4ms ± 10ms」变成严格 4ms，
        # 判跳变从"跟抖动比大小"变成"跟噪音比大小"，也就稳了 ——
        # 这才是上一版把门槛从 12ms 放宽到 30ms 想解决的同一个问题的**正解**，
        # 放宽门槛只是让它别误报，样本时钟是让它根本不需要判。
        #
        # 为什么整体差一个常数无所谓：回放 → 采集之间本来就有几十毫秒的固定延迟
        # （驱动缓冲 + 空气 + 采集缓冲），它从头到尾不变，直接落进滤波器的
        # 第一个抽头里被吸收掉。我们要的只是它**不变**，不是它等于多少 ——
        # [5/7] 那组测试（12/55/90/130ms 四档时延，全都不用给参数）验的就是这一点。
        if self._last_t is None:
            t_use = t_end                     # 第一次：先认声卡的时间戳
        else:
            pred = self._last_t + dur         # 内部时钟：上一块末尾 + 这一块时长
            if abs(t_end - pred) > self.gap_tol:
                self._drop_history()          # 真跳了 → 参考历史作废、重新对表
                self.n_gap += 1
                t_use = t_end
            else:
                t_use = pred                  # 正常：不信这次的时间戳，按样本数走

        for start in range(0, n, B):
            blk = mic[start:start + B]
            take = min(B, n - start)
            if blk.size < B:                  # 尾巴不满一块就补零（输出只取前 blk.size 个）
                pad = np.zeros(B, dtype=np.float32)
                pad[:blk.size] = blk
                blk = pad
            # 这一小块的「最后一个样本」时刻。由内部时钟派生 → 严格等距，
            # 不再有任何抖动；滤波器和参考缓冲看到的是一条干净的时间轴。
            t_sub = t_use - (n - (start + take)) / float(self.sr)
            self._last_t = t_sub
            last = self._step(blk, t_sub, ref, now)
            # ⚠️ 必须截回 take 个。末尾不满一个子块时音频被补了零，
            # 滤波器照样吐 B 个样本出来 —— 直接赋给长度 take 的槽会炸。
            # 这个坑一直潜伏着：默认块长 1024 刚好整除子块 64，永远走不到那条支路。
            out[start:start + take] = last.pop("_y")[:take]

        self.n_frames += 1
        if last:
            self.last = last
        # ⚠️ 返回的 info 是**最后一个小块**的判决，不是整块的汇总。
        # 调用方（AudioWorker）拿它决定"这一块要不要按静音算"，
        # 所以这个约定很关键：粒度是"块末尾那 4ms 的结论"。
        # 它天然带了点惯性——块内前半段判成双讲、末尾没判到，这一块就仍按
        # 有回声处理（偏保守的方向），这正合我们的意。
        return out, dict(last)

    def _drop_history(self):
        """丢掉参考历史（保留已学到的滤波器系数）。

        刻意**不动内部时钟** `_last_t` —— 清参考历史是因为"某一段时刻对不上了"，
        但时间轴本身还在往前走。调用方紧接着就会用重新对过表的时间把它写回去；
        在这里先抹掉只会让中间的判断失去依据。
        """
        if self.mode == "fd":
            self.Uf[:] = 0.0
            self.prev_blk[:] = 0.0

    def _step(self, d, t_sub, ref, now):
        self.n_sub += 1
        return self._step_fd(d, t_sub, ref, now) if self.mode == "fd" \
            else self._step_td(d, t_sub, ref, now)

    # ---------- 收尾：两种模式共用的 DTD / 抑制 / 统计 ----------
    def _dtd(self, mic_rms, ref_rms, echo_rms, far_now):
        """双讲判据。两条，一粗一细。

        粗判据（跟房间耦合无关，但很钝）：`麦克风 > ratio × 参考`。
        参考信号送出去只会被衰减，不可能在麦克风那里变得比原信号还响 ——
        所以这条一旦成立，一定是有人在说话。冷启动时它是**唯一**能用的判据。

        细判据（跟房间耦合无关，且灵敏得多）：`麦克风 > ratio × 回声估计`。
        滤波器收敛之后，`y` 就是"这个房间里回声本来该有多大"。
        麦克风比它还大好几倍，多的那部分就是人。

        ⚠️ 细判据必须由 `converged` 把关，这一步走过一段弯路，记下来：
        最初的把关条件是 `echo_rms > 0.05 × ref_rms`（"回声估计有点样子了"）。
        看着合理，其实是错的 —— 它把**收敛过程中间**那一大段直接放行了：
            echo_est/麦克风  0 → 0.05   不用细判据，正常收敛
                             0.05 → 0.5 **误判成双讲 → 冻结 → 收敛停住**
                             > 0.5       不用细判据，继续收敛
        中间那段"死区"正好卡在收敛的半路上。结果是：滤波器学到一半被冻住，
        残差一直很高；而"残差高"又让 AudioWorker 那边把它当成"有人在说话"。
        症状就是 **MOSS 把自己没消干净的回声当成用户**（工具里 S4 场景翻车）。
        换句话说：判据里那个"我够不够格用这条判据"的问题本身，
        不能用判据要保护的那个量来衡量 —— 又是一个自指的坑（同 asr.py 那个）。

        所以改成看**收敛度 rho = 残差/麦克风**：它跟"有没有人"无关，
        纯粹是"滤波器抵消得怎么样"。压到 0.5（6dB）以下就认定学过一遍了。

        参数是 `far_now`（此刻扬声器真的在放）而不是 `far_active`（含余音）：
        余音期没有远端信号，谈不上"分不清谁在说"，也就没有双讲可言。
        """
        if not far_now:
            return False
        if mic_rms > self.dtd_ratio * ref_rms:
            return True
        if self.converged and mic_rms > self.dtd_ratio * echo_rms:
            return True
        return False

    def _finish(self, e, d, mic_rms, ref_rms, echo_rms, adapted, now=None):
        # (3) 双讲检测
        #
        # ⚠️ 这里有两个**不一样**的概念，混用会出一个很隐蔽的 bug，所以拆开：
        #
        #   far_now  —— 参考缓冲里**此刻**确实有波形（扬声器正在放）
        #   far_tail —— 扬声器刚停，房间混响还在响（余音）
        #
        # 一开始只有一个 far_active = (ref_rms > 门槛)，于是扬声器一停它就变 False。
        # 但声音不会跟着电信号一起断：实测（tools/fullduplex_e2e.py 的 S4）
        # 播放在 3.00s 结束，3.04/3.11/3.17s 三块的麦克风里还躺着 -31/-27/-37dBFS
        # 的混响尾巴，而参考缓冲已经是**精确的零**（-234dB）——
        # 于是"只有回声"的那道闸在最需要它的时候关掉了，尾巴被当成用户说话，
        # 切出一句 0.58s 的"自问自答"。真机上音箱一停也是一样的道理。
        #
        # 所以：余音期内**照样**按"有回声"对待（继续抑制、继续关门），
        # 只是不再自适应（参考都是零了，学不到东西），也不判双讲
        # （没有远端在放，就不存在"分不清谁在说"的问题）。
        far_now = ref_rms > self.ref_floor
        if far_now:
            self._last_far_t = now if now is not None else time.time()
        far_tail = (self._last_far_t is not None
                    and (now if now is not None else time.time())
                    - self._last_far_t < self.tail_s)
        far_active = far_now or far_tail

        # 收敛度：残差 / 麦克风。只在"真的在放"的时候才有意义。
        # 用 EMA（不取 min）：单个瞬间的低残差不算收敛，要持续低下来才算。
        # 取 min 的写法试过 —— 会被一次偶然的低残差骗到，过早置位（见 conv_rho 注释）。
        if far_now:
            out_rms0 = float(np.sqrt(np.mean(e * e)) + 1e-12)
            rho_now = out_rms0 / mic_rms
            self.rho = rho_now if self.rho is None else (
                0.8 * self.rho + 0.2 * rho_now)
            if not self.converged and self.rho < self.conv_rho:
                self.converged = True

        dtd = self._dtd(mic_rms, ref_rms, echo_rms, far_now)

        # 预热期：远端在响，但滤波器还没证明自己学过一遍 → 这段的残差不可信，
        # 既不该拿去判双讲，也不该拿去判"有人在说话"。如实记个数备查。
        warm = far_active and not self.converged
        if warm:
            self.n_warm += 1

        # 自适应只发生在"远端确实在响、且没双讲"的时候。
        # 余音期虽然 far_active 为真，但参考是零，学不到东西 —— 所以用 far_now。
        if not (self.adapt and far_now and not dtd):
            if not far_active:
                self.n_cold += 1
            else:
                self.n_freeze += 1
                if dtd:
                    self.n_dtd += 1
                elif far_tail and not far_now:
                    self.n_tail += 1
            adapted = False

        # (4) 残余回声抑制：确认"只有远端在响"时再压一把。
        #     双讲期间**不能压** —— 那正是用户在说话的时候。
        #     余音期要压：那几块滤波器已经压不动了（参考是零），
        #     不额外压一把，混响尾巴就会顶穿 VAD 门限。
        suppress = far_active and not dtd and self.resid_gain < 1.0
        if suppress:
            e = e * self.resid_gain
            self.n_resid += 1

        # (5) 量 ERLE：只有"纯远端"的块才有意义
        #     （双讲时残差里混着用户的声音，算出来会虚高，不能自欺欺人）
        if far_now and not dtd and self.n_adapt > 4:
            out_rms = max(float(np.sqrt(np.mean(e * e)) + 1e-12), 1e-12)
            erle = 20.0 * math.log10(mic_rms / out_rms)
            if -5.0 < erle < 90.0:                       # 明显抽风的样本直接扔
                self._erle_sum += erle
                self._erle_n += 1
                self.erle_db = self._erle_sum / self._erle_n

        return e.astype(np.float32), {
            "erle_db": round(self.erle_db, 1),
            "dtd": dtd,
            "far_active": bool(far_active),
            "far_now": bool(far_now),
            "far_tail": bool(far_tail),
            "warm": bool(warm),
            "converged": bool(self.converged),
            "adapted": bool(adapted),
            "suppressed": bool(suppress),
            "ref_db": round(20.0 * math.log10(ref_rms + 1e-12), 1),
            "mic_db": round(20.0 * math.log10(mic_rms + 1e-12), 1),
            "out_db": round(20.0 * math.log10(float(np.sqrt(np.mean(e * e)) + 1e-12)), 1),
        }

    def _bail(self, e, d, mic_rms):
        """发散兜底：输出比输入还大得多 → 说明减错了，不是减对了。

        自适应滤波器总有翻车的可能（数值、异常输入、房间突变）。
        这时候**宁可退回半双工**（清零系数、原样放行），
        也不能让爆音进到后面的识别和播放里。
        """
        rms = float(np.sqrt(np.mean(e * e)))
        if math.isfinite(rms) and rms <= 8.0 * mic_rms:
            return e, False
        self.n_reset += 1
        # 只清滤波器状态，**不动统计和时钟** —— 这里还在一次 process() 内部，
        # 把 _last_t 抹掉会让后面的小块误判成"时间轴跳变"。
        if self.mode == "fd":
            self.Wf[:] = 0.0
            self.Uf[:] = 0.0
            self.prev_blk[:] = 0.0
            self.P[:] = 1e-8
            self._p_ready = False
        else:
            self.wr[:] = 0.0
        # 系数都清零了，"我学过一遍"这句话也就不成立了 —— 收敛位跟着撤掉，
        # 让后面的块重新走一遍预热，别拿着旧的信心去做双讲判断。
        self.converged = False
        self.rho = None
        return d.copy(), True

    # ---------- 频域（PBFDAF）：默认 ----------
    def _step_fd(self, d, t_sub, ref, now):
        B, N, M = self.B, self.N, self.M

        # (1) 只要"最新那一块"参考。更早的都躺在 Uf 里（见类注释里的滑移技巧）。
        new = ref.window(t_sub, B, now=now)
        ref_rms = float(np.sqrt(np.mean(new * new)) + 1e-12)
        mic_rms = float(np.sqrt(np.mean(d * d)) + 1e-12)

        # (2) 滑移：第 k 帧的第 p 块 == 第 k-1 帧的第 p-1 块
        if M > 1:
            self.Uf[1:] = self.Uf[:-1]
        frame = self._frame
        frame[:B] = self.prev_blk
        frame[B:] = new
        self.Uf[0] = np.fft.rfft(frame)
        self.prev_blk = new

        # (3) 回声估计：频域相乘、跨分块求和，再变回时域取后半段
        #     （前半段被循环卷积污染了，这正是 overlap-save 要丢掉的部分）
        Yf = np.sum(self.Wf * self.Uf, axis=0)
        y = np.fft.irfft(Yf, N)[B:]
        e = d - y
        echo_rms = float(np.sqrt(np.mean(y * y)) + 1e-12)

        e, diverged = self._bail(e, d, mic_rms)

        # (4) 逐 bin 功率：**这就是频域收敛快的全部秘密**。
        #     低频 bin 功率大 → 步长自动变小；高频 bin 功率小 → 步长自动变大。
        #     时域只有一个全局步长，被低频能量绑死，高频永远学不完。
        #
        #     ⚠️ 必须对**所有分块求和**，不能只取 Uf[0]。
        #     因为回声估计是 y = Σ_p U_p·W_p，而每一步对每个 W_p 都更新一次，
        #     y 的净变化是 Σ_p |U_p|²/P 倍。只归一化一个分块时这个倍数是 M（本例 63），
        #     于是 e ← (1 - M·mu)·e —— 一步就把误差放大 62 倍，立刻发散。
        #     求和之后倍数是 1，退化成教科书里最标准的单分块形式（e ← (1-mu)·e）。
        p_now = np.sum(np.abs(self.Uf) ** 2, axis=0)
        if not self._p_ready:
            # 第一次有远端时**直接把 P 设成观测值**，别从 1e-8 慢慢爬。
            # 否则第一帧 P 只有观测值的 (1-smooth) 倍（0.15），
            # 等效步长变成 mu/0.15 = 6.7μ —— 冷启动那一下就把滤波器打飞了。
            self.P = p_now.copy()
            self._p_ready = True
        else:
            self.P = self.smooth * self.P + (1.0 - self.smooth) * p_now
            # 相对地板：某个频点长期没有能量（比如男声在 6kHz 以上几乎为零）时，
            # 分母趋近 0 会让那里的步长炸掉。用一个相对均值的地板兜住。
            np.maximum(self.P, self.p_floor * float(np.mean(self.P)), out=self.P)

        far_active = ref_rms > self.ref_floor
        dtd = self._dtd(mic_rms, ref_rms, echo_rms, far_active)
        adapted = False
        if self.adapt and far_active and not dtd:
            # 误差放在**前半段**、后半段补零，而且要和 U 取**共轭** ——
            # overlap-save 的约定。这两种选择一共 4 个组合，实测只有这一个收敛：
            #
            #     Ef 位置   共轭   6 秒后 ERLE
            #     前半段    ✓        12.8 dB   ← 就是这个
            #     前半段    ✗       -10.1 dB
            #     后半段    ✓        -7.7 dB
            #     后半段    ✗        -9.6 dB
            #
            # 关键在输出取的是 irfft 的**后半段**（前半段被循环卷积污染了），
            # 误差就必须放在对应的前半段，梯度才对得上位。
            # 另外三种不是"效果差一点"，是**发散** —— 残留误差比回声还大。
            Ef = np.fft.rfft(np.concatenate([np.zeros(B, np.float32), e]))
            upd = (np.conj(self.Uf) * Ef) / (self.P + self.eps)
            self.Wf = (self.Wf * self.leak
                       + (self.mu * upd).astype(np.complex64)).astype(np.complex64)
            self.n_adapt += 1
            adapted = True

        out, info = self._finish(e, d, mic_rms, ref_rms, echo_rms, adapted, now)
        info["diverged"] = bool(diverged)
        info["echo_db"] = round(20.0 * math.log10(echo_rms + 1e-12), 1)
        info["_y"] = out
        return info

    # ---------- 时域分块 NLMS：留作对照 ----------
    def _step_td(self, d, t_sub, ref, now):
        L, B = self.L, self.B

        # (1) 取参考窗口：[t_sub - (L+B-1)/sr, t_sub]，共 L+B 个样本。
        #     多要 B 个是因为要算 B 个输出（每个输出用一段 L 长的参考）。
        x = ref.window(t_sub, L + B, now=now)
        ref_rms = float(np.sqrt(np.mean(x * x)) + 1e-12)
        mic_rms = float(np.sqrt(np.mean(d * d)) + 1e-12)

        # (2) 回声估计 + 残差
        #     np.correlate valid 的输出长度是 len(x)-len(wr)+1 = B+1，
        #     第 j 个输出对应"以 x[j] 结尾的 L 个样本"与 wr 的内积。
        #     我们要的是 j = 1..B（对齐关系见 README），所以 [1:B+1]。
        corr = np.correlate(x, self.wr, "valid")
        y = corr[1:B + 1]
        e = d - y
        echo_rms = float(np.sqrt(np.mean(y * y)) + 1e-12)

        e, diverged = self._bail(e, d, mic_rms)

        far_active = ref_rms > self.ref_floor
        dtd = self._dtd(mic_rms, ref_rms, echo_rms, far_active)
        adapted = False
        if self.adapt and far_active and not dtd:
            grad = np.correlate(x, e, "valid")[1:L + 1]
            # NLMS 的分母必须是"参与这一批输出的那些输入向量"的能量和。
            # 我们有 B 个输出，每个用一段 L 长的参考；把它们逐个求和是 B×L 次运算，
            # 而它们高度重叠，所以 Σ‖x_i‖² ≈ B·‖x‖²（误差 (L+B)/L，本例 1.6%）。
            # 一开始这里只写了 ‖x‖²，漏掉那个 B —— 等效步长被放大 63 倍，直接发散。
            denom = B * float(np.dot(x, x)) + self.eps
            self.wr = (self.wr * self.leak
                       + (self.mu / denom) * grad.astype(np.float32)).astype(np.float32)
            self.n_adapt += 1
            adapted = True

        out, info = self._finish(e, d, mic_rms, ref_rms, echo_rms, adapted, now)
        info["diverged"] = bool(diverged)
        info["echo_db"] = round(20.0 * math.log10(echo_rms + 1e-12), 1)
        info["_y"] = out
        return info

    # ---------- 展示 ----------
    def stats_line(self):
        s = ("回声消除：ERLE %.1f dB · 更新 %d 次 · 双讲冻结 %d 次 · "
             "残余抑制 %d 块 · 参考静音 %d 块"
             % (self.erle_db, self.n_adapt, self.n_dtd, self.n_resid, self.n_cold))
        if self.n_warm:
            s += " · 预热 %d 块" % self.n_warm
        if self.n_tail:
            s += " · 余音 %d 块" % self.n_tail
        if self.n_gap:
            s += " · 时间轴跳变 %d 次" % self.n_gap
        if self.n_reset:
            s += " · ⚠ 发散复位 %d 次" % self.n_reset
        return s

    def describe(self):
        how = ("频域分块 PBFDAF（%d 分块 × %d 点 FFT）" % (self.M, self.N)
               if self.mode == "fd" else "时域分块 NLMS")
        return ("%s · %d 抽头(%.0fms) · 步长 %.2f · 双讲门 %.1f× · 残余抑制 %.0fdB"
                % (how, self.L, self.taps_ms, self.mu, self.dtd_ratio,
                   -20.0 * math.log10(self.resid_gain)))


# ================================================================
#  4. 假房间：离线验证用
# ================================================================
def room_impulse(sr, delay_ms=55.0, tail_ms=70.0, direct=0.62,
                 reflections=((0.0, 0.34), (13.0, 0.21), (27.0, 0.15),
                              (46.0, 0.10), (72.0, 0.062)),
                 seed=7):
    """合成一个房间冲激响应。

    结构 = 直达声（延迟 delay_ms 到达）+ 几个早期反射 + 指数衰减的混响尾巴。
    这比"纯延迟"更接近真实，也更能暴露滤波器长度不够的问题。
    """
    n = int(round((delay_ms + tail_ms + 20.0) / 1000.0 * sr))
    h = np.zeros(max(1, n), dtype=np.float64)
    d0 = int(round(delay_ms / 1000.0 * sr))

    for dt_ms, g in reflections:
        i = d0 + int(round(dt_ms / 1000.0 * sr))
        if i < len(h):
            h[i] += g

    if tail_ms > 0:
        rng = np.random.default_rng(seed)
        k0 = d0 + int(round(reflections[-1][0] / 1000.0 * sr)) + 1
        m = len(h) - k0
        if m > 0:
            t = np.arange(m) / float(sr) * 1000.0
            env = np.exp(-t / (tail_ms / 3.0))
            h[k0:] += 0.045 * env * rng.standard_normal(m)

    # 归一化到单位能量，这样"耦合强度"就由调用方单独控制，好推理
    e = math.sqrt(float(np.dot(h, h)))
    if e > 0:
        h /= e
    return h


def speechlike(sr, seconds, seed=0, f0=118.0):
    """合成一段"像人在说话"的测试信号。

    为什么要费这个劲而不是用白噪声：白噪声的频谱太平，
    自适应滤波器在它上面收敛得**异常好**，测出来的 ERLE 会虚高，
    到了真人语音（低通、有声/无声交替、非平稳）就原形毕露。
    所以这里故意做得像语音：浊音谐波 + 音节包络 + 清音噪声。
    """
    rng = np.random.default_rng(seed)
    n = max(1, int(sr * seconds))
    t = np.arange(n) / float(sr)

    # 音节包络：3~4Hz 的起伏（正常语速的音节率），再叠一层慢速语调
    env = 0.28 + 0.72 * np.abs(np.sin(2 * math.pi * 3.3 * t + 0.8 * np.sin(2 * math.pi * 0.6 * t)))
    env *= 0.55 + 0.45 * np.sin(2 * math.pi * 0.31 * t) ** 2

    # 浊音：基频随时间漂移（说话有语调），叠 5 个谐波
    f = f0 * (1.0 + 0.13 * np.sin(2 * math.pi * 0.47 * t) + 0.05 * np.sin(2 * math.pi * 1.9 * t))
    phase = 2 * math.pi * np.cumsum(f) / float(sr)
    voiced = np.zeros(n)
    for k, a in ((1, 1.0), (2, 0.62), (3, 0.4), (4, 0.26), (5, 0.16), (6, 0.1)):
        voiced += a * np.sin(k * phase)

    # 清音：低幅噪声，随机地"出现在某些音节上"
    noise = rng.standard_normal(n)
    fric = np.clip(np.sin(2 * math.pi * 2.7 * t + 1.7) - 0.3, 0, None) * noise

    sig = env * (voiced + 1.4 * fric)
    peak = float(np.max(np.abs(sig))) + 1e-12
    return (0.35 * sig / peak).astype(np.float32)


class FakeRoom:
    """把"扬声器在放什么"变成"麦克风收到什么"。

    每个样本块调用一次 `echo_of()` 得到回声，再叠上近端语音和噪声 ——
    这就是离线世界里能造出的、最接近真机的东西。
    """

    def __init__(self, sr, ir, coupling=0.5, noise_db=-62.0, seed=3):
        self.sr = int(sr)
        self.ir = np.asarray(ir, dtype=np.float64)
        self.coupling = float(coupling)
        self.noise = 10.0 ** (noise_db / 20.0)
        self.rng = np.random.default_rng(seed)
        self._tail = np.zeros(len(self.ir) - 1, dtype=np.float64)   # 块间卷积状态

    def echo_of(self, far):
        """far = 本块扬声器发出的信号 → 本块麦克风收到的回声。"""
        far = np.asarray(far, dtype=np.float64)
        buf = np.concatenate([self._tail, far])
        full = np.convolve(buf, self.ir)
        out = full[len(self._tail):len(self._tail) + len(far)]
        self._tail = buf[-(len(self.ir) - 1):] if len(self.ir) > 1 else np.zeros(0)
        return (out * self.coupling).astype(np.float32)

    def noise_like(self, n):
        return (self.noise * self.rng.standard_normal(n)).astype(np.float32)


# ================================================================
#  5. 自检：造一个假房间，量真实的 ERLE
# ================================================================
class _Clock:
    """可控时钟。用真实 time.time() 会让自检跑得像蜗牛 —— 而且不可复现。"""

    def __init__(self, t0=1000.0):
        self.t = t0

    def __call__(self):
        return self.t

    def tick(self, dt):
        self.t += dt


def _db(x):
    return 20.0 * math.log10(float(np.sqrt(np.mean(np.asarray(x) ** 2))) + 1e-12)


def _run_scene(sr=16000, block=1024, sub=64, taps_ms=DEFAULT_TAPS_MS, mu=None,
               resid_db=10.0, dtd_ratio=2.0, seconds=6.0, delay_ms=55.0,
               coupling=0.5, noise_db=-62.0, interrupt_at=None, interrupt_ms=0.0,
               mode="fd", jitter_ms=0.0):
    """跑一个完整场景，返回逐块的账单。

    interrupt_at : 第几秒开始"用户插话"（None = 全程只有远端）
    jitter_ms    : 时间戳抖动幅度。真实声卡回调给的时间戳**不是**等距的，
                   实测 MME 下有 ±10ms 的抖动；传这个参数就能把它模拟出来。
    """
    clk = _Clock()
    rng = np.random.default_rng(17)
    far = speechlike(sr, seconds, seed=11)
    near = speechlike(sr, seconds, seed=99, f0=196.0)   # 换个基频，和远端去相关
    if interrupt_at is None:
        near = np.zeros_like(near)
    else:
        i0 = int(interrupt_at * sr)
        i1 = min(len(near), i0 + int(interrupt_ms / 1000.0 * sr))
        near = np.concatenate([np.zeros(i0, np.float32), near[i0:i1],
                               np.zeros(max(0, len(near) - i1), np.float32)])

    ir = room_impulse(sr, delay_ms=delay_ms)
    room = FakeRoom(sr, ir, coupling=coupling, noise_db=noise_db)

    ref = ReferenceBuffer(sr)
    aec = EchoCanceller(sr, taps_ms=taps_ms, subblock=sub, mu=mu, mode=mode,
                        resid_db=resid_db, dtd_ratio=dtd_ratio)

    # 整段"播放"在 t=0 登记，之后按块推进时间轴。
    # 注意：这里刻意**不**告诉 AEC 任何时延信息 —— 就是要验证它能自己吸收掉。
    t_play = clk.t
    ref.start(t_play, far, sr)

    rows = []
    for s in range(0, seconds and int(seconds * sr) - block + 1, block):
        blk_far = far[s:s + block]
        blk_near = near[s:s + block]
        # ⚠️ 回声必须**取这次 echo_of() 的真实返回值**，不能事后自己再卷一遍。
        # 一开始就是自己在外层重算的，忘了带上"上一块的卷积尾巴"，
        # 结果回声能量少算约 12 dB，ERLE 被算成 2 dB，白白怀疑了半天滤波器。
        echo = room.echo_of(blk_far)
        d = echo + blk_near + room.noise_like(block)

        t_end = t_play + (s + block) / float(sr)
        if jitter_ms:
            # 只抖"块的时间戳"，音频本身是等距采到的 —— 这正是真机的样子
            t_end += float(rng.uniform(-jitter_ms, jitter_ms)) / 1000.0
        clk.tick(block / float(sr))
        out, info = aec.process(d, t_end=t_end, ref=ref, now=clk())

        rows.append({
            "t": (s + block) / float(sr),
            "mic_db": _db(d),
            "out_db": _db(out),
            "echo_db": _db(echo),
            "near_db": _db(blk_near),
            "active_far": bool(np.sqrt(np.mean(blk_far ** 2)) > 0.01),
            "active_near": bool(np.sqrt(np.mean(blk_near ** 2)) > 0.005),
            "dtd": info["dtd"],
            "adapted": info["adapted"],
            "suppressed": info["suppressed"],
        })
    return rows, aec


def _selftest():
    ok = True
    sr, block = 16000, 1024
    clk = _Clock()

    # ---------------- [1/7] 参考缓冲的时间轴 ----------------
    print("\n[1/7] 参考缓冲：只暴露「已经播出去」的部分")
    rb = ReferenceBuffer(sr)
    pcm = np.ones(sr, dtype=np.float32)                # 1 秒的直流 1.0
    rb.start(100.0, pcm, sr)
    w = rb.window(100.5, 100, now=100.5)
    print("      播了 0.5s，取 0.5s 处的窗口 → 均值 %.3f（应为 1.0）" % float(w.mean()))
    assert abs(w.mean() - 1.0) < 1e-6, "已经播出去的部分必须是真实波形"
    w = rb.window(100.9, 100, now=100.5)
    print("      只播到 0.5s，却去取 0.9s 的窗口 → 均值 %.3f（应为 0.0）" % float(w.mean()))
    assert abs(w.mean()) < 1e-6, "还没播到的部分必须按 0 算（否则会拿未来的声音去减）"
    rb.finish(100.5)
    w = rb.window(100.7, 100, now=100.9)
    print("      0.5s 处被打断，再取 0.7s 的窗口 → 均值 %.3f（应为 0.0）" % float(w.mean()))
    assert abs(w.mean()) < 1e-6, "被打断后不能继续拿「虚拟播放」的样本去减"
    print("      ✓ 时间轴三条规则都对（已播=真值 / 未播=0 / 打断后=0）")

    # ---------------- [2/7] 假房间 ----------------
    print("\n[2/7] 假房间：合成冲激响应 + 回声")
    ir = room_impulse(sr, delay_ms=55.0, tail_ms=70.0)
    print("      冲激响应 %d 抽头（%.0fms）· 直达声在第 %d 抽头"
          % (len(ir), len(ir) / sr * 1000, int(np.argmax(np.abs(ir)))))
    assert len(ir) > 0.1 * sr, "混响尾巴要够长，不然测不出滤波器长度不够的问题"
    room = FakeRoom(sr, ir, coupling=0.5, noise_db=-99.0, seed=1)
    far = speechlike(sr, 1.0, seed=5)
    echo = room.echo_of(far)
    ratio_db = 20 * math.log10(float(np.sqrt(np.mean(echo ** 2))) /
                               float(np.sqrt(np.mean(far ** 2))))
    print("      回声/远端 电平比 %.1f dB（耦合 0.5 → 理论约 -6 dB）" % ratio_db)
    assert -12 < ratio_db < -2, "假房间的耦合强度不对，后面的 ERLE 就没意义了"
    print("      ✓ 假房间可用（回声确实是远端的线性函数）")

    # ---------------- [3/7] ERLE：纯远端 ----------------
    print("\n[3/7] ERLE —— 只有远端在响时，能减掉多少回声")
    rows, aec = _run_scene(seconds=6.0, interrupt_at=None)
    tail = [r for r in rows if r["active_far"]][len(rows) // 3:]
    erle = sum(r["echo_db"] - r["out_db"] for r in tail) / max(1, len(tail))
    print("      滤波器 %d 抽头 · %s" % (aec.L, aec.describe()))
    print("      后半段平均 ERLE = %.1f dB" % erle)
    print("      （参考：手机免提一般 20~35dB；>20dB 已经能让人正常插话）")
    assert erle > 20.0, "ERLE 只有 %.1f dB，AEC 没在工作" % erle
    print("      ✓ ERLE %.1f dB ≥ 20 dB" % erle)

    # 收敛曲线（让人看到"要多久才好用"）
    curve = [(r["t"], r["echo_db"] - r["out_db"]) for r in rows if r["active_far"]]
    marks = []
    for target in (1.0, 2.0, 4.0):
        got = [e for t, e in curve if t >= target]
        if got:
            marks.append("%.0fs:%.0fdB" % (target, got[len(got) // 2]))
    print("      收敛过程 → " + "  ".join(marks))

    # 为什么频域值得多写一百行：同样时长，两条路径的差距是量级上的
    print("      对照（同时长，ERLE 不含残余抑制）：")
    for secs in (2.0, 6.0):
        line = []
        for mode in ("td", "fd"):
            r2, _ = _run_scene(seconds=secs, interrupt_at=None, mode=mode, resid_db=0.0)
            t2 = [x for x in r2 if x["active_far"]][len(r2) // 3:]
            line.append("%s %.1f dB" % ("时域" if mode == "td" else "频域",
                                        sum(x["echo_db"] - x["out_db"] for x in t2) / max(1, len(t2))))
        print("        %3.0fs  " % secs + "   |   ".join(line))

    # ---------------- [4/7] 双讲：用户插话 ----------------
    print("\n[4/7] 双讲 —— 用户在它说话时插进去")
    rows, aec = _run_scene(seconds=8.0, interrupt_at=3.0, interrupt_ms=1500.0)
    dt = [r for r in rows if r["active_near"]]
    print("      插话期间：%d 块，其中判成双讲的 %d 块（%.0f%%）"
          % (len(dt), sum(1 for r in dt if r["dtd"]),
             100.0 * sum(1 for r in dt if r["dtd"]) / max(1, len(dt))))
    assert sum(1 for r in dt if r["dtd"]) > 0.6 * len(dt), "双讲检测没检出来，滤波器会被带坏"
    # 插话时输出里必须还有用户的声音（不能被当回声减掉）
    keep = sum(r["out_db"] for r in dt) / len(dt)
    print("      插话期间输出电平 %.1f dBFS（要明显高于静音，说明人的声音留住了）" % keep)
    assert keep > -45.0, "插话的声音被一起减掉了，那还不如半双工"
    # 滤波器不能被带发散：插话结束后 ERLE 要能回去
    after = [r for r in rows if r["active_far"] and not r["active_near"] and r["t"] > 5.0]
    back = sum(r["echo_db"] - r["out_db"] for r in after) / max(1, len(after))
    print("      插话结束后的 ERLE = %.1f dB（要回到 15dB 以上，说明滤波器没被带崩）" % back)
    assert back > 15.0, "双讲把滤波器带坏了（ERLE 掉到 %.1f dB），说明更新没冻结住" % back
    print("      ✓ 双讲：检得出、人的声音留住、滤波器没发散")

    # ⚠️ 只测"该检的时候检得出"是不够的，还得测"**不该检的时候别乱检**"。
    #    这一条是事后补的 —— 因为漏了它，一个很严重的 bug 溜了过去：
    #    误判双讲 → 闸门大开 → 自己的残留回声被当成用户 → 自问自答。
    #    前三条只看了"插话那 25 块有没有检出来"，而这个 bug 恰恰发生在
    #    **没人插话**的时候，前三条一个字都没说。测试的盲区就是这么来的：
    #    它只测了你想到的那一面。工具里 S4 场景翻车才发现。
    rows2, _ = _run_scene(seconds=8.0, interrupt_at=None)     # 全程只有远端，没人在场
    fo = [r for r in rows2 if r["active_far"]]
    fp = sum(1 for r in fo if r["dtd"])
    print("      纯远端（没人在场）：%d 块，误判成双讲的 %d 块（%.1f%%）"
          % (len(fo), fp, 100.0 * fp / max(1, len(fo))))
    assert fp <= 0.10 * len(fo), \
        "双讲误报 %.0f%%，滤波器会被反复冻住、收敛不下去" % (100.0 * fp / max(1, len(fo)))
    # 更关键的一条：纯远端时，输出得**压到 VAD 门限（-50dBFS）以下**。
    # 只要高于它，下游的分段器就会在自己说的话上切出一整句。
    # 开头那段（滤波器预热）不算 —— 真链路里 AudioWorker 会用 warm 标志兜住。
    body = fo[len(fo) // 4:]
    loud = [r for r in body if r["out_db"] > -50.0]
    print("      纯远端后 3/4：输出高于 -50dBFS 的 %d / %d 块（要少，否则被 VAD 当人声）"
          % (len(loud), len(body)))
    assert len(loud) <= 0.10 * len(body), \
        "残留回声高过 -50dBFS 的块太多，MOSS 会听见自己说话"
    print("      ✓ 双讲不误报：没人在场时，残留回声压在 VAD 门限之下")

    # ---------------- [5/7] 时延鲁棒性 ----------------
    print("\n[5/7] 时延鲁棒性 —— 不给任何时延信息，看滤波器能不能自己吸收")
    print("      时延 = 声卡输出缓冲 + 空气传播 + 声卡输入缓冲，是这套系统里")
    print("      最不可控的一段（驱动/蓝牙/USB 各不相同，几十毫秒到一百多）。")
    for d_ms in (12.0, 55.0, 90.0, 130.0):
        rows, _ = _run_scene(seconds=5.0, delay_ms=d_ms, interrupt_at=None)
        tail = [r for r in rows if r["active_far"]][len(rows) // 3:]
        erle = sum(r["echo_db"] - r["out_db"] for r in tail) / max(1, len(tail))
        print("      时延 %5.0fms → ERLE %.1f dB" % (d_ms, erle))
        assert erle > 16.0, ("%.0fms 时延下只有 %.1f dB —— 滤波器长度不够了。"
                             "真机上遇到时加 --aec-taps-ms" % (d_ms, erle))
    print("      ✓ 四档时延全过，一行时延参数都不用给")

    # ---------------- [6/7] 时间轴：抖动 vs 真跳变 ----------------
    # ⚠️ 上面整段用的都是 `_Clock`，时间戳**精确等距**（4ms 一步）——
    #    真机不是这样。声卡回调的时间戳实测有 ±10ms 抖动（MME 尤其明显），
    #    比子块步长（4ms）还大。这件事被漏掉过一次：门槛按子块长取 12ms，
    #    结果正常抖动就把门槛打穿，每两三块误报一次"时间轴跳变" →
    #    参考历史被反复清空 → 滤波器收敛一直从头开始。
    #    真机上不会崩，只会**悄悄变差**，所以必须有个测试盯着。
    #
    #    后来发现比"放宽门槛"更彻底的解法：门槛放宽只是让它别误报，
    #    真正的正解是**别再拿抖动的时间戳当时间轴**——自己按样本数走。
    #    于是这一组要盯两个方向：抖动不能误报，真跳变不能漏报。
    print("\n[6/7] 时间轴 —— 抖动不能误判，真丢块不能漏判")
    jer, jaec = _run_scene(seconds=5.0, interrupt_at=None, jitter_ms=8.0)
    # 用"精确时钟"那一组做对照，抖动组不该明显变差
    ref_rows, raec = _run_scene(seconds=5.0, interrupt_at=None, jitter_ms=0.0)
    jt = [r for r in jer if r["active_far"]][len(jer) // 3:]
    rt = [r for r in ref_rows if r["active_far"]][len(ref_rows) // 3:]
    je = sum(r["echo_db"] - r["out_db"] for r in jt) / max(1, len(jt))
    re_ = sum(r["echo_db"] - r["out_db"] for r in rt) / max(1, len(rt))
    print("      抖动 ±8ms：ERLE %.1f dB · 误报时间轴跳变 %d 次"
          % (je, jaec.n_gap))
    print("      精确时钟 ：ERLE %.1f dB · 误报时间轴跳变 %d 次"
          % (re_, raec.n_gap))
    assert jaec.n_gap == 0, (
        "±8ms 的时间戳抖动被当成了时间轴跳变 %d 次 —— 门槛定得比抖动还小。"
        "后果是参考历史被反复清空、收敛反复从头开始" % jaec.n_gap)
    assert je > re_ - 3.0, ("抖动下 ERLE 掉了 %.1f dB，说明抖动真的破坏了滤波器"
                            % (re_ - je))
    print("      ✓ 抖动 ±8ms：不误判、ERLE 和精确时钟一样（%.1f dB）" % je)

    # (b) 反向：**真**丢一块必须抓到。
    # 这一条是在给 (a) 上保险：如果为了让 (a) 过去而把门槛一路放宽（或者
    # 干脆不判跳变），丢块就会漏过 —— 那时参考历史里存的是错位的参考，
    # 滤波器拿一段对不上的声音去减，比不复位还糟。
    # 模拟：音频和时间戳**一起**往前跳一个块（这才是丢块的样子；
    # 光跳时间戳、音频不动，那是另一种故障）。
    gap_aec = EchoCanceller(sr)
    gap_ref = ReferenceBuffer(sr)
    far5 = speechlike(sr, 3.0, seed=5)
    t0 = 1000.0
    gap_ref.start(t0, far5, sr)
    s = 0
    for _ in range(20):
        gap_aec.process(far5[s:s + block], t_end=t0 + (s + block) / float(sr),
                        ref=gap_ref, now=t0 + 10.0)
        s += block
    assert gap_aec.n_gap == 0, "时间戳完全等距却报了 %d 次跳变" % gap_aec.n_gap
    s += block                                   # ← 丢掉这一块
    gap_aec.process(far5[s:s + block], t_end=t0 + (s + block) / float(sr),
                    ref=gap_ref, now=t0 + 10.0)
    print("      丢一块后：报出时间轴跳变 %d 次（应为 1）" % gap_aec.n_gap)
    assert gap_aec.n_gap == 1, (
        "丢了一块却报了 %d 次跳变 —— 门槛宽过头了，参考历史会留着错位的参考"
        % gap_aec.n_gap)
    print("      ✓ 真丢块：抓到了（不多不少 1 次）")

    # ---------------- [7/7] 边界 ----------------
    print("\n[7/7] 边界：不该动的时候一点都不动")
    aec2 = EchoCanceller(sr, subblock=256)
    sig = speechlike(sr, 0.5, seed=2)
    # (a) 没有参考缓冲 → 直通
    out, info = aec2.process(sig, t_end=0.5, ref=None)
    assert np.allclose(out, sig), "没接参考缓冲时必须原样直通"
    print("      ✓ 没有参考缓冲 → 原样直通（等价于关掉 AEC）")
    # (b) 有缓冲但从来没播过东西 → 参考全 0 → 输出也不该被改
    rb2 = ReferenceBuffer(sr)
    small = EchoCanceller(sr, subblock=256, resid_db=0.0)
    out, info = small.process(sig, t_end=0.5, ref=rb2, now=0.5)
    print("      远端静音时：mic %.1f dB → out %.1f dB（应几乎不变）"
          % (_db(sig), _db(out)))
    assert abs(_db(out) - _db(sig)) < 0.5, "远端静音时不该动任何东西"
    assert small.n_adapt == 0, "远端静音时不该更新滤波器"
    print("      ✓ 远端静音 → 不更新、不衰减（不会把安静时的人声吃掉）")
    # (c) 全是静音
    out, info = small.process(np.zeros(256, np.float32), t_end=0.6, ref=rb2, now=0.6)
    assert float(np.max(np.abs(out))) < 1e-6
    print("      ✓ 全静音输入 → 全静音输出")
    # (d) 残余抑制真的生效（远端在响、没人插话）
    rows, aec3 = _run_scene(seconds=5.0, interrupt_at=None, resid_db=10.0)
    assert aec3.n_resid > 0, "残余抑制一次都没触发"
    print("      ✓ 残余抑制触发 %d 块（远端在响且没人在说话时才压）" % aec3.n_resid)

    print("\n" + "=" * 62)
    print("[OK] aec.py 自检全部通过（7/7）")
    print("     " + aec3.stats_line())
    print("=" * 62)
    return 0 if ok else 1


def _bench():
    """看看每块音频的处理开销 —— AEC 是要跑在实时循环里的。"""
    sr, block = 16000, 1024
    ref = ReferenceBuffer(sr)
    ref.start(0.0, speechlike(sr, 6.0, seed=1), sr)
    for sub in (32, 64, 128):
        for taps in (150.0, 250.0):
            aec = EchoCanceller(sr, taps_ms=taps, subblock=sub)
            t0 = time.time()
            n = 200
            for i in range(n):
                t_end = (i + 1) * block / float(sr)
                aec.process(np.random.randn(block).astype(np.float32) * 0.05,
                            t_end=t_end, ref=ref, now=t_end)
            dt = time.time() - t0
            audio = n * block / float(sr)
            print("  抽头 %4d (%3.0fms)  子块 %3d  →  %6.2f ms/块  "
                  "占实时 %5.1f%%  (RTF %.4f)"
                  % (aec.L, aec.taps_ms, sub, dt / n * 1000,
                     100.0 * dt / audio, dt / audio))
    return 0


def _main():
    p = argparse.ArgumentParser(description="MOSS 回声消除（AEC）")
    p.add_argument("--selftest", action="store_true", help="全套自检（合成房间）")
    p.add_argument("--bench", action="store_true", help="测处理开销")
    p.add_argument("--decode", default=None, help="解码一个音频文件，看参考 PCM")
    a = p.parse_args()

    if a.bench:
        return _bench()
    if a.decode:
        pcm = decode_reference(a.decode, 16000)
        if pcm is None:
            print("解码失败（没装 PyAV？）")
            return 2
        print("解码成功：%d 样本 / %.2fs / 峰值 %.3f"
              % (pcm.size, pcm.size / 16000.0, float(np.max(np.abs(pcm)))))
        return 0
    return _selftest()


if __name__ == "__main__":
    sys.exit(_main())
