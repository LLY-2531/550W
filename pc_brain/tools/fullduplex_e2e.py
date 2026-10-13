#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 17 端到端探针：MOSS 正在说话，你插一句 —— 它听得见吗？
============================================================
这个脚本要回答的问题只有一个，而且必须是**行为**层面的回答：

    它在念念有词的时候，你说"等一下"，它到底有没有听进去？

为什么 `aec.py --selftest` 不够
------------------------------------------------------------
`aec.py` 的自检证明的是「回声能被减掉 22dB」。但那只是链条中的一环。
真正决定"你插嘴它听不听得见"的，是**三个部件一起**的动作：

    麦克风 → [AEC 减回声] → [分段器切句] → [识别] → 你看到一句话

只要其中任何一道闸没对齐，ERLE 再高也没用。最典型的三种错法：
  1. AEC 减干净了，但 `echo_gate()` 还在按"它在说话"屏蔽分段器 → 你还是白说
  2. `echo_gate()` 放行了，但融合层的 `gated` 还按静音算 → 屏幕上永远挂着 SPEAK
  3. 参考信号没接上（比如 SAPI 合成到文件那步失败），却仍然开着麦克风 →
     MOSS 听见自己 → 自己回答自己 → 无限自我对话

这个脚本就是**照着重现**上面三条，用真的分段器、真的识别线程、
真的 `AudioWorker`、真的 `echo_blind()`，只把两样东西换成仿真：
  - 麦克风：用 FakeRoom 把"扬声器的声音"按真实声学路径灌回来
  - 嗓子：不驱动声卡，只在 ReferenceBuffer 上按真实时长登记参考

于是它验的是「判定 → 闸门 → 切句 → 识别」整条链，而不是滤波器系数。

四个场景（2×2：AEC 开/关 × 有没有人插话）
------------------------------------------------------------
  S1  AEC 开 · 有人插话   → 期待：**切出一句，且判定是"用户"**（真全双工）
  S2  AEC 关 · 有人插话   → 期待：**一句都切不出来**（旧半双工，插话被吞）
  S3  AEC 关 · 没人插话   → 期待：**切出句子，且判定是"自己"**
                            （这就是为什么必须有那道闸 —— 不设闸它就自问自答）
  S4  AEC 开 · 没人插话   → 期待：**一句都切不出来**（回声清干净了，不自说自听）

S1 和 S2 的对比 = 阶段 17 到底解决了什么。
S3 和 S4 的对比 = 为什么不能"干脆别屏蔽麦克风"。

怎么区分"这句话是用户的还是自己的"
------------------------------------------------------------
让"用户的嗓音"和"MOSS 的嗓音"基频明显不同（205Hz vs 118Hz），
识别引擎就用一个**频带能量比**判：
    r = E(180~260Hz) / (E(180~260Hz) + E(90~150Hz))
MOSS 的嗓音基频 118Hz 落在低频带 → r 低；用户 205Hz 落在高频带 → r 高。
脚本开跑前会先在**纯信号**上标定这个判据（见 calibrate），数字打出来，
分离不好就当场看得见 —— 不靠"我觉得应该能分"。

用法：
    python tools/fullduplex_e2e.py                # 跑四个场景
    python tools/fullduplex_e2e.py --only s1      # 只跑一个（调试用）
    python tools/fullduplex_e2e.py --fast         # 时间轴压缩（不推荐，只在赶时间时用）
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

from aec import EchoCanceller, ReferenceBuffer, FakeRoom, room_impulse, speechlike  # noqa: E402
from asr import AsrEngine, AsrWorker, UtteranceSegmenter                          # noqa: E402
from moss_brain import AudioWorker, echo_blind                                    # noqa: E402
from voice import SpeakWorker                                                     # noqa: E402


# ================================================================
#  常量：时间轴与信号
# ================================================================
SR = 16000
BLOCK = 1024
BLOCK_S = BLOCK / float(SR)          # 64ms

# MOSS 这一句的时长（它一直在说，用户才有得插）
FAR_S = 3.0
# 用户插话的起点与时长（落在 MOSS 说话的中间，这是最关键的一刀：
# 插话完全发生在"它正在播"的时间窗内，半双工的闸一定会拦住它）
NEAR_AT = 1.4
NEAR_S = 1.1

# 两个嗓音的基频。差得越开，"是谁在说"越好判 —— 这一点直接决定
# 探针的判据可不可靠，所以宁可取得极端一点。
FAR_F0 = 118.0
NEAR_F0 = 205.0

# 声学耦合：回声相对远端信号的电平。0.6 意味着"麦克风收到的自己的声音"
# 只比"扬声器放出来的"低 4.4dB —— 真实免提里这是个偏恶劣的值，
# 故意取恶劣的，好让"没 AEC"的后果暴露得清楚。
COUPLING = 0.6
# 近端（用户）在麦克风处的电平。比回声高 6dB —— 真人凑近笔记本说话的样子。
NEAR_DBFS = -20.0


def db(x):
    return 20.0 * math.log10(max(float(x), 1e-12))


def set_rms(sig, target_dbfs):
    """把一段信号的 RMS 缩放到指定 dBFS。用于把"用户嗓音"摆到合适音量。"""
    rms = float(np.sqrt(np.mean(sig.astype(np.float64) ** 2))) + 1e-12
    want = 10.0 ** (target_dbfs / 20.0)
    return (sig * (want / rms)).astype(np.float32)


def rms_dbfs(sig):
    if len(sig) == 0:
        return -99.0
    return db(np.sqrt(np.mean(np.asarray(sig, dtype=np.float64) ** 2)))


def band_ratio(pcm, sr=SR):
    """判据：高频带(180~260Hz) 占 (高频带 + 低频带(90~150Hz)) 的比例。

    MOSS 的嗓音 f0=118Hz，能量几乎全压在 90~150 → 比值低；
    用户的 f0=205Hz 在 180~260 → 比值高。**标定过了再用**，见 calibrate()。
    """
    x = np.asarray(pcm, dtype=np.float64).reshape(-1)
    if x.size < 128:
        return 0.0
    x = x - np.mean(x)
    w = x * np.hanning(x.size)
    sp = np.abs(np.fft.rfft(w)) ** 2
    fr = np.fft.rfftfreq(w.size, 1.0 / sr)
    low = float(sp[(fr >= 90.0) & (fr < 150.0)].sum())
    high = float(sp[(fr >= 180.0) & (fr < 260.0)].sum())
    return high / (low + high + 1e-12)


RATIO_TH = 0.55          # 高于它判"用户"，低于它判"自己"（由 calibrate 校验）


# ================================================================
#  1. 时间轴：把"扬声器在放什么 / 麦克风该收到什么"摆到一条钟上
# ================================================================
class Timeline:
    """仿真麦克风的时间轴。

    所有东西都挂在**墙上时钟**上（不是块号），因为 AEC 的 ReferenceBuffer
    和 AudioWorker 的 last_time 都是墙上时钟。要验真实链路，时间基准就得统一。

    远端（MOSS 的嗓子）：
        t_play 之前没声音；t_play 之后第 m 个样本在 t_play + m/sr 播出。
        t_play 由 _FarVoice 真正开始"播"的那一刻写进来 —— 不是脚本预先假设的，
        而是**跟着 SpeakWorker 实际调度走**，这样连线程延迟都一起仿真了。

    近端（用户）：
        起点**锚在 t_play 上**（t_play + near_at），而不是墙上某个绝对时刻。
        这是故意的：SpeakWorker 是个线程，"台词提交"到"真的开口"之间
        有不确定的调度延迟。如果近端按绝对时刻排，那点抖动就可能把
        "插话"挪到 MOSS 还没开口的时候 —— 那样测的就不是全双工了。
        锚在 t_play 上，插话就永远落在它说话的中间，每一次都一样。
    """

    def __init__(self, far, near, near_at, room):
        self.far = far
        self.near = near
        self.near_at = near_at
        self.room = room
        self.t_play = None

    @staticmethod
    def _slice(buf, i0, i1):
        """取 buf[i0:i1]，越界补零。i0/i1 可以是负数。"""
        n = i1 - i0
        out = np.zeros(n, dtype=np.float32)
        a = max(0, i0)
        b = min(len(buf), i1)
        if b > a:
            out[a - i0:b - i0] = buf[a:b]
        return out

    def mic_end_at(self, t_end, n):
        """返回"结束于 t_end 的 n 个采样"这一刻，麦克风收到的东西。"""
        # ---- 远端回声 ----
        if self.t_play is None:
            echo = self.room.echo_of(np.zeros(n, dtype=np.float32))
        else:
            m_end = int(round((t_end - self.t_play) * SR))
            far_blk = self._slice(self.far, m_end - n, m_end)
            echo = self.room.echo_of(far_blk)

        # ---- 近端（用户），锚在 t_play 上 ----
        near_blk = np.zeros(n, dtype=np.float32)
        if self.t_play is not None:
            i_end = int(round((t_end - self.t_play - self.near_at) * SR))
            near_blk = self._slice(self.near, i_end - n, i_end)

        # ---- 麦克风本底噪声 ----
        # ⚠️ 一定要加。不加的话"没声音"就是**精确的零**（-180dBFS），
        # 噪声底会被压到 -180，于是任何一丁点残差都判成"有人在说话"，
        # 分段器会在纯静音上疯狂切句。真麦克风永远有本底噪声，
        # 仿真里就必须有，否则 VAD 的 SNR 判据形同虚设。
        noise = self.room.noise_like(n)
        return (echo + near_blk + noise).astype(np.float32)


class SimMic:
    """按真实时间吐块的假麦克风。

    它自己**负责限速**（睡到该出这一块的时刻），所以 AudioWorker 里那段
    "demo/wav 按真实时间播放"的限速会自然退化成空转 —— 这正是我们想要的：
    整条链跑在真实节奏上，跟接真声卡时一致。
    """

    def __init__(self, timeline, block=BLOCK, sr=SR):
        self.tl = timeline
        self.sr = sr                    # AudioWorker 会读它算块时长
        self.block = block
        self.t_start = time.time()
        self.n = 0
        self.last_time = None

    def read(self):
        self.n += 1
        t_end = self.t_start + self.n * self.block / float(self.sr)
        left = t_end - time.time()
        if left > 0:
            time.sleep(left)
        self.last_time = t_end
        return self.tl.mic_end_at(t_end, self.block)

    def release(self):
        pass


# ================================================================
#  2. 假嗓子：占住"说话中"这段时间，并按约定登记参考
# ================================================================
class _FarVoice:
    """只做两件事：占时间（让 speak_worker.busy() 为真）、登记参考信号。

    它**不出声**。回声由 FakeRoom 在麦克风侧合成 —— 因为这里要的是
    可控、可复现的声学路径，而不是真去驱动一块声卡。
    """

    name = "simfar"
    supports_reference = True

    def __init__(self, far, timeline, sr=SR):
        self.far = far
        self.tl = timeline
        self.sr = sr
        self.n_spoken = 0
        self.interrupted = 0

    def speak(self, text, should_stop=None, ref=None, ref_pcm=None, ref_sr=None):
        t0 = time.time()
        # 关键：把"远端从这一刻开始播"写进时间轴。之后麦克风侧照它算回声。
        self.tl.t_play = t0
        if ref is not None:
            ref.start(t0, ref_pcm if ref_pcm is not None else self.far,
                      ref_sr or self.sr)
        dur = len(self.far) / float(self.sr)
        stopped = False
        try:
            while True:
                el = time.time() - t0
                if el >= dur:
                    break
                if should_stop and should_stop():
                    stopped = True
                    break
                time.sleep(0.01)
            return int((time.time() - t0) * 1000), stopped
        finally:
            if ref is not None:
                ref.finish(time.time())
            self.n_spoken += 1
            if stopped:
                self.interrupted += 1

    def stop(self):
        pass

    def close(self):
        pass


# ================================================================
#  3. 判"是谁在说"的识别引擎
# ================================================================
class JudgeEngine(AsrEngine):
    """假识别引擎：不认字，只判"这句话是用户说的还是 MOSS 自己的回声"。

    为什么这样就够：探针验的不是识别准确率（那是阶段 12 的事），
    而是"该不该有一句话被送进来识别"。所以引擎只要回答
    「送来的这段音频长得像谁的嗓子」，就能证明闸门放行的是不是对的东西。
    """

    name = "judge"

    def __init__(self, log):
        self.log = log

    def transcribe(self, pcm, samplerate=SR):
        r = band_ratio(pcm, samplerate)
        who = "用户" if r > RATIO_TH else "自己"
        self.log.append({"who": who, "ratio": round(r, 3),
                         "dur": round(len(pcm) / float(samplerate), 2)})
        return "%s(%.2f)" % (who, r)


# ================================================================
#  4. 一个场景
# ================================================================
class Scenario:
    def __init__(self, name, title, aec_on, has_near, guard=True):
        self.name = name
        self.title = title
        self.aec_on = aec_on
        self.has_near = has_near
        self.guard = guard            # False = 模拟 --no-echo-guard（只给 S3 用）


def run_scenario(sc, far, near, room_ir, verbose=False):
    """跑一个场景，返回 (切出的句子数, 每次识别的判定列表, 诊断字符串)。"""
    # 没人插话的场景，就把近端置空 —— 于是麦克风里只剩"回声 + 本底噪声"
    near_sig = near if sc.has_near else np.zeros(0, dtype=np.float32)

    # ---- 每次都用全新的房间实例：FakeRoom 内部有卷积尾巴状态，
    #      跨场景复用会让上一个场景的回声漏到下一个里 ----
    room = FakeRoom(SR, room_ir, coupling=COUPLING, noise_db=-62.0, seed=3)
    tl = Timeline(far, near_sig, NEAR_AT, room)

    refbuf = ReferenceBuffer(SR, seconds=6.0) if sc.aec_on else None
    aec = None
    if sc.aec_on:
        aec = EchoCanceller(SR, taps_ms=250.0, subblock=64)

    voice = _FarVoice(far, tl)
    speak_worker = SpeakWorker(voice, guard_ms=200, dedup_s=0.0,
                               reference=refbuf)
    speak_worker.start()

    seg = UtteranceSegmenter(SR, BLOCK)
    asr_log = []
    engine = JudgeEngine(asr_log)
    asr_worker = AsrWorker(engine, SR, queue_size=4)
    asr_worker.start()

    # 闸门：就用 moss_brain 里那个**真身**，不另写一份
    if sc.guard:
        gate = lambda: echo_blind(speak_worker, aec, refbuf)      # noqa: E731
    else:
        gate = lambda: False                                      # noqa: E731

    class _Args:
        samplerate = SR
        blocksize = BLOCK
        no_clap = True
        clap_th = -20.0
        db_th = -50.0
        snr_th = 8.0

    src = SimMic(tl)
    aw = AudioWorker(src, _Args(), BLOCK_S, segmenter=seg,
                     echo_gate=gate, on_utterance=asr_worker.submit,
                     aec=aec, reference=refbuf)

    n_utt = [0]
    orig_feed = seg.feed

    def counting_feed(blk, voiced=None):
        out = orig_feed(blk, voiced)
        n_utt[0] += len(out)
        return out

    seg.feed = counting_feed

    # ---- 起跑 ----
    aw.start()
    time.sleep(0.35)                       # 让采集线程先热起来
    speak_worker.submit("系统自检完成，各单元上线，正在核对各项参数。")   # 开口

    # 跑到"远端播完 + 足够长的静默"为止。
    #
    # ⚠️ 这里是探针最容易写错的地方：**分段器不是一有声音就吐句子的**，
    # 它要等 end_ms（700ms）的连续静音，确认"话说完了"才把这一整句吐出来。
    # 收得太早 → 永远看不到那句话 → 误以为是 AEC 没生效。
    # 所以判据挂在 t_play 上（而不是墙上某个固定秒数），
    # 给 1.8s 的余量 = 700ms 静音 + 识别 + 线程抖动。
    t_cap = time.time() + FAR_S + NEAR_AT + 4.0
    while time.time() < t_cap:
        if tl.t_play is not None and (time.time() - tl.t_play) >= FAR_S + 1.8:
            break
        time.sleep(0.05)
    # 场景里若压根没等到"开口"（比如嗓子没起来），别死等
    time.sleep(0.3)

    aw.shutdown()
    aw.join(timeout=2.0)
    speak_worker.shutdown()
    asr_worker.shutdown()
    time.sleep(0.3)
    asr_worker.join(timeout=2.0)

    diag = "ERLE %.1f dB · 更新 %d · 双讲冻结 %d · 残余抑制 %d" % (
        aec.erle_db, aec.n_adapt, aec.n_dtd, aec.n_resid) if aec else "（AEC 关）"
    if verbose:
        print(f"      [diag] {diag}")
    return n_utt[0], asr_log, diag


# ================================================================
#  5. 标定 + 主流程
# ================================================================
def calibrate(far, near):
    r_far = band_ratio(far)
    r_near = band_ratio(near)
    print("  判据标定（纯信号，不含回声/噪声）：")
    print(f"    MOSS 的嗓子（f0={FAR_F0:.0f}Hz） 比值 = {r_far:.3f}"
          f"   → 判「自己」{'✓' if r_far <= RATIO_TH else '✗'}")
    print(f"    用户的嗓子（f0={NEAR_F0:.0f}Hz） 比值 = {r_near:.3f}"
          f"   → 判「用户」{'✓' if r_near > RATIO_TH else '✗'}")
    ok = (r_far <= RATIO_TH) and (r_near > RATIO_TH)
    gap = r_near - r_far
    print(f"    分离度 = {gap:.3f}（阈值 {RATIO_TH}，越大越稳）")
    return ok


def main():
    ap = argparse.ArgumentParser(description="阶段17 全双工端到端探针")
    ap.add_argument("--only", default=None, help="只跑某个场景，如 s1/s2/s3/s4")
    ap.add_argument("--verbose", action="store_true", help="打印每个场景的内部统计")
    a = ap.parse_args()

    print("=" * 74)
    print("  阶段 17 端到端探针：MOSS 正说着话，你插一句，它听得见吗？")
    print("=" * 74)
    print(f"  采样 {SR}Hz · 块 {BLOCK}（{BLOCK_S * 1000:.0f}ms）· "
          f"MOSS 说 {FAR_S}s · 你插话 @{NEAR_AT}s 说 {NEAR_S}s")
    print(f"  声学耦合 {COUPLING} · 近端电平 {NEAR_DBFS:.0f}dBFS\n")

    # ---- 造信号 ----
    far = speechlike(SR, FAR_S, seed=11, f0=FAR_F0)
    near = set_rms(speechlike(SR, NEAR_S, seed=23, f0=NEAR_F0), NEAR_DBFS)
    ir = room_impulse(SR, delay_ms=55.0, tail_ms=120.0)
    # 房间冲激响应归一化到单位能量（和 aec.py 自检一致），
    # 这样 coupling 的物理意义才干净：回声 ≈ 远端 × coupling。
    ir = (ir / math.sqrt(float(np.sum(ir ** 2)))).astype(np.float64)

    print(f"  远端电平 {rms_dbfs(far):.1f} dBFS · 近端电平 {rms_dbfs(near):.1f} dBFS")
    print(f"  单程回声估计 {rms_dbfs(far) + db(COUPLING):.1f} dBFS"
          f"（近端比它高 {NEAR_DBFS - (rms_dbfs(far) + db(COUPLING)):.1f}dB）\n")

    if not calibrate(far, near):
        print("\n[失败] 「是谁在说」这个判据本身都分不开，先修信号再谈 AEC。")
        return 2
    print()

    scenarios = [
        Scenario("s1", "AEC 开 · 有人插话 → 应该听见", aec_on=True, has_near=True),
        Scenario("s2", "AEC 关 · 有人插话 → 旧行为，插话被吞", aec_on=False, has_near=True),
        Scenario("s3", "AEC 关 · 没人插话 → 不自设闸就会自问自答",
                 aec_on=False, has_near=False, guard=False),
        Scenario("s4", "AEC 开 · 没人插话 → 回声清干净，不自说自听",
                 aec_on=True, has_near=False),
    ]
    if a.only:
        scenarios = [s for s in scenarios if s.name == a.only]
        if not scenarios:
            print(f"[失败] 没有叫 {a.only} 的场景")
            return 2

    results = {}
    print("  " + "-" * 70)
    for sc in scenarios:
        print(f"  [{sc.name.upper()}] {sc.title}")
        n_utt, log, diag = run_scenario(sc, far, near, ir, verbose=a.verbose)
        results[sc.name] = (n_utt, log)
        if log:
            for i, rec in enumerate(log):
                print(f"        识别 #{i + 1}: {rec['who']}  "
                      f"(比值 {rec['ratio']} · 时长 {rec['dur']}s)")
        else:
            print("        （没有任何音频被送去识别）")
        print(f"        切句 {n_utt} 段 · {diag}")
        print()

    # ---------------- 判定 ----------------
    print("  " + "=" * 70)
    print("  结论")
    print("  " + "=" * 70)
    fails = []

    def check(name, cond, msg):
        mark = "✓" if cond else "✗"
        print(f"    {mark} {msg}")
        if not cond:
            fails.append(name)

    # S1：全双工的核心断言 —— 用户插话必须被听见，而且必须判成"用户"
    n1, l1 = results.get("s1", (0, []))
    if "s1" in results:
        check("s1-cut", n1 >= 1,
              f"S1 全双工：插话被切出来了（{n1} 段 ≥ 1）")
        check("s1-who", any(r["who"] == "用户" for r in l1),
              "S1 全双工：切出来的确实是**用户的**声音（不是回声漏网）")

    # S2：旧行为必须被复现 —— 半双工下插话被整个吞掉
    n2, _ = results.get("s2", (0, []))
    if "s2" in results:
        check("s2-swallowed", n2 == 0,
              f"S2 半双工：插话被闸门吞掉了（{n2} 段，应为 0）"
              " —— 这就是阶段 17 要修的毛病")

    # S3：没人却切出句子，而且判成"自己" → 证明闸不能省
    n3, l3 = results.get("s3", (0, []))
    if "s3" in results:
        check("s3-self", n3 >= 1 and all(r["who"] == "自己" for r in l3),
              f"S3 反面教材：不设闸就听见自己（{n3} 段，全判成「自己」）"
              " —— 自说自听就是这么来的")

    # S4：AEC 开且没人在时，一段都不该切出来
    n4, _ = results.get("s4", (0, []))
    if "s4" in results:
        check("s4-silent", n4 == 0,
              f"S4 全双工：没人说话时不自说自听（{n4} 段，应为 0）")

    print()
    if fails:
        print(f"  [失败] {len(fails)} 项没过：{', '.join(fails)}")
        return 1
    print("  [OK] 全双工端到端通过。")
    print("       它说话时你插一句，句子会被切出来、送去识别；")
    print("       而没人说话时，它不会把自己的回声当成你。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
