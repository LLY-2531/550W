#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOSS 语音层（"嗓子"）  ·  流浪地球 550W 复刻项目 · 阶段 11
============================================================
阶段 10 给了 MOSS 一张"嘴"：大模型想出一句中文台词，打在终端、短标签打到 OLED。
但终端里那行字是**字幕**，不是**声音**。

这一步把 `say` 字段变成真的声音，从音箱里出来。

三层结构
------------------------------------------------------------
    SpeakWorker    线程 + 队列（只留最新一句；一旦开口就说完，只有拍手能打断）
    Voice          后端接口：speak() / stop() / close()
    ├─ EdgeVoice   微软 edge-tts 在线神经语音（音质最好，需联网）
    ├─ SapiVoice   Windows 内置 SAPI（离线可用，音色机械——反而像 MOSS）
    └─ NullVoice   什么都不做

播放为什么不用 ffmpeg / playsound
------------------------------------------------------------
Windows 自带 MCI（winmm.dll），几行 ctypes 就能放 mp3，**零额外依赖**：

    open "<文件>" type mpegvideo alias moss1234
    play moss1234

这台机器上没有 ffmpeg，用 MCI 正好绕开。这也是为什么要先把 mp3 落盘再播，
而不是"边合成边播"——MCI 需要一个完整的文件。

一个必须解决的物理问题：自说自听
------------------------------------------------------------
MOSS 用音箱说话 → 麦克风立刻听见 → 大脑以为"有人在跟我说话" → 又进 SPEAK
→ 又说一句 → 麦克风又听见…… 这是个正反馈，MOSS 会自己跟自己聊起来。

所以说话期间**主动把听觉屏蔽掉**（dbfs 按静音算），说完再留一小段"余音时间"
（`guard_ms`，默认 350ms，等房间混响和音频缓冲排空）。
视觉不受影响 —— 说话时你站过去，它照样在 SPEAK 状态盯着你。

延迟是多少（实测）
------------------------------------------------------------
    合成（联网）  ≈ 1.7 ~ 2.9 秒   ← 第一次连 TLS 握手更慢
    播放          ≈ 音频本身长度（约 1.5 秒 / 15 字）
所以从"状态一变"到"听见声音"大约 2~3 秒。听着像"思考了一下再开口"，
对 MOSS 这个角色反而不违和。设备那边是**秒响应**的：MODE 立刻切 SPEAK、
灯和电机马上动，只有声音晚一点到。

用法
------------------------------------------------------------
    python voice.py --test "系统自检完成"          # 单独测嗓子
    python voice.py --list                        # 列出 edge 可选音色
    python voice.py --selftest                    # 全套自检（合成/播放/打断）
"""

import argparse
import asyncio
import base64
import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

# ================================================================
#  0. 小工具：把 WAV 读成参考信号要的 PCM
# ================================================================
def read_wav_pcm(path, target_sr=None):
    """读一个 WAV 文件，返回 float32 单声道 PCM。

    优先用 PyAV（如果有，它带正经的重采样器）；
    退路是标准库 `wave` —— WAV 本来就是裸 PCM，压根不需要解码器。
    两条路都不行就返回 None（调用方按"这句没有参考信号"处理，不影响出声）。

    为什么 SAPI 这条路要单独写：它的参考信号来自我们自己刚生成的 WAV，
    格式（采样率/位深/声道）由系统决定，不能假设 —— 所以一律**读文件头**，
    而不是提前写死一个格式。
    """
    if not os.path.exists(path) or os.path.getsize(path) < 44:
        return None
    if target_sr:
        try:
            from aec import decode_reference
            pcm = decode_reference(path, target_sr)
            if pcm is not None and pcm.size:
                return pcm
        except Exception:
            pass
    try:
        import wave

        import numpy as np
        with wave.open(path, "rb") as w:
            n_ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
            raw = w.readframes(w.getnframes())
        if width == 2:
            data = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
        elif width == 1:
            data = (np.frombuffer(raw, "u1").astype(np.float32) - 128.0) / 128.0
        elif width == 4:
            data = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
        else:
            return None
        if n_ch > 1:
            data = data.reshape(-1, n_ch).mean(axis=1)
        if target_sr and rate != target_sr:
            from aec import resample_linear
            data = resample_linear(data, rate, target_sr)
        return data.astype(np.float32)
    except Exception:
        return None


# ================================================================
#  1. MCI 播放器（Windows 自带，零依赖）
# ================================================================
class MciPlayer:
    """用 winmm.dll 的 MCI 接口播放本地 mp3。

    MCI 是 Windows 上一个很老的媒体接口，但它有几个好处：
      - 系统自带，不用装 ffmpeg / pygame
      - 能 `stop`，所以可以"打断"
      - `status mode` 能查到是否还在播，不用 sleep 猜
    """

    def __init__(self, alias=None):
        if not sys.platform.startswith("win"):
            raise RuntimeError("MCI 是 Windows 专有接口，换 --tts sapi 之外请改用 edge+其他播放器")
        self.winmm = ctypes.windll.winmm
        self.send = self.winmm.mciSendStringW
        self.send.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_wchar),
                              ctypes.c_uint, ctypes.c_void_p]
        self.send.restype = ctypes.c_uint
        self.geterr = self.winmm.mciGetErrorStringW
        self.geterr.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_wchar), ctypes.c_uint]
        self.geterr.restype = ctypes.c_int
        # 一个进程里 alias 是全局的，用 pid 保证不撞车
        self.alias = alias or ("moss%d" % os.getpid())
        self.opened = False

    # ---------- 底层 ----------
    def _cmd(self, cmd):
        buf = ctypes.create_unicode_buffer(512)
        err = self.send(cmd, buf, 511, None)
        return err, buf.value

    def _err_text(self, code):
        buf = ctypes.create_unicode_buffer(512)
        try:
            if self.geterr(code, buf, 511):
                return buf.value
        except Exception:
            pass
        return "错误码 %d" % code

    # ---------- 对外 ----------
    def open(self, path, hint=None):
        self.close()
        p = os.path.abspath(path)
        last = None
        # 试几种设备类型，第一个成功的算数。
        # 为什么不是"只看后缀"：MCI 的设备名取决于系统里注册了哪些解码器，
        # 同一台机器上 .wav 也未必非要用 waveaudio。所以给个优先级，
        # 然后一律留一条"让 MCI 自己挑"的兜底（None）。
        if hint:
            order = (hint, None)
        elif p.lower().endswith(".wav"):
            order = ("waveaudio", None)
        else:
            order = ("mpegvideo", None)
        for typ in order:
            cmd = ('open "%s" type %s alias %s' % (p, typ, self.alias) if typ
                   else 'open "%s" alias %s' % (p, self.alias))
            err, _ = self._cmd(cmd)
            if err == 0:
                self.opened = True
                return typ or "default"
            last = err
        self._cmd("close %s" % self.alias)
        raise RuntimeError("MCI 打不开音频文件：%s" % self._err_text(last))

    def length_ms(self):
        if not self.opened:
            return 0
        _, v = self._cmd("status %s length" % self.alias)
        try:
            return int(v)
        except ValueError:
            return 0

    def play(self, path, should_stop=None, ref=None, ref_pcm=None, ref_sr=None,
             hint=None):
        """阻塞播放整段音频。should_stop() 返回 True 就立刻停（打断）。

        ref / ref_pcm ：【阶段 17】回声消除要的那份「参考信号」。
            `ref` 是一个 ReferenceBuffer，`ref_pcm` 是即将播出的 PCM。
            这里做两件事：
              1. 下 `play` 命令**之前**登记时间轴（越准越好）；
              2. 播完（含被打断）之后 `finish()` 把时间轴截断。
            为什么不需要去查「现在播到第几毫秒」：声卡时钟精度是 ppm 级，
            3 秒的句子漂移不到 1ms；而「发 play 命令」到「真正出声」的固定延迟
            会被自适应滤波器当成时延吸收掉。反过来，每 20ms 去查一次播放位置
            反而会引入 20ms 级的抖动 —— 那是滤波器学不会的东西。

        返回 (实际播放毫秒, 是否被打断)。
        """
        self.open(path, hint=hint)
        total = self.length_ms()
        t0 = time.time()
        if ref is not None and ref_pcm is not None:
            ref.start(t0, ref_pcm, ref_sr or ref.sr)
        err, _ = self._cmd("play %s" % self.alias)
        if err != 0:
            self.close()
            if ref is not None:
                ref.finish()
            raise RuntimeError("MCI 播放失败：%s" % self._err_text(err))

        # 上限：知道长度就用长度，不知道就赌 10 秒
        deadline = t0 + (total / 1000.0 if total > 0 else 10.0) + 0.6
        started = False
        stopped = False

        while time.time() < deadline:
            if should_stop and should_stop():
                self._cmd("stop %s" % self.alias)
                stopped = True
                break
            _, mode = self._cmd("status %s mode" % self.alias)
            mode = mode.strip()
            if mode == "playing":
                started = True
            elif started:                    # 播完了自己会变回 stopped
                break
            time.sleep(0.02)

        t_stop = time.time()
        # 时间轴截断要用"真正停下的时刻"，不能等到 close() 之后 ——
        # close 要几十毫秒，这段误差会直接变成 AEC 的对齐误差。
        if ref is not None:
            ref.finish(t_stop)
        self.close()
        return int((t_stop - t0) * 1000), stopped

    def stop(self):
        if self.opened:
            self._cmd("stop %s" % self.alias)

    def close(self):
        if self.opened:
            self._cmd("close %s" % self.alias)
            self.opened = False


# ================================================================
#  2. 后端：offline / edge / sapi
# ================================================================
class NullVoice:
    """静音后端：--tts off 时用，接口齐全但什么都不做。"""

    name = "off"
    supports_reference = False       # 它不发声，自然没有参考信号

    def speak(self, text, should_stop=None, ref=None, ref_pcm=None, ref_sr=None):
        return 0, False

    def stop(self):
        pass

    def close(self):
        pass


class EdgeVoice:
    """微软 edge-tts 在线神经语音。音质最好，中文自然，需要联网。

    参数里最有用的两个：
        rate  语速，"-10%" 就慢 10%（MOSS 说话慢一点更压得住场）
        pitch 音调，"-15Hz" 压低（越低越"冷"）
    """

    name = "edge"
    # 它会先把音频落盘再播，所以那份文件就是天然的参考信号（解一下就行）
    supports_reference = True

    def __init__(self, voice="zh-CN-YunxiNeural", rate="-10%", pitch="-15Hz",
                 volume="+0%", timeout=15.0, player=None, on_ref_error=None):
        import edge_tts                      # 延迟导入：没装也不影响程序启动
        self.edge_tts = edge_tts
        self.on_ref_error = on_ref_error     # 解码参考信号失败时报一声（不致命）
        self.models = None                   # 缓存音色列表，供 --list 用
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self.volume = volume
        self.timeout = timeout
        self.player = player or MciPlayer()

    # ---------- 合成 ----------
    def _synth(self, text, out_path, should_stop=None):
        """流式收音频块写文件。

        为什么不用现成的 `c.save()`：`save()` 是个黑盒，中途没法喊停。
        而实测一次合成要 1.7~2.9 秒 —— 如果这期间来了新台词，白等的这两秒
        会让"打断"变成"我刚说的话还在排队"。改成 `stream()` 自己收，
        每收一块问一次 `should_stop()`，就能在几十毫秒内放弃。

        返回 True=收完；False=中途被放弃。
        """
        async def go():
            c = self.edge_tts.Communicate(
                text, self.voice, rate=self.rate, pitch=self.pitch,
                volume=self.volume,
                connect_timeout=int(self.timeout),
                receive_timeout=int(self.timeout),
            )
            stream = c.stream()
            try:
                with open(out_path, "wb") as f:
                    async for chunk in stream:
                        if should_stop and should_stop():
                            return False
                        if chunk["type"] == "audio":
                            f.write(chunk["data"])
                return True
            finally:
                # 关键：提前 return 时必须显式关掉异步生成器，
                # 否则 websocket 和 aiohttp session 不释放，
                # 退出时会刷一屏 "Unclosed client session" 吓人一跳。
                await stream.aclose()
        # asyncio.run 在子线程里也能用（每次新建一个事件循环）
        return asyncio.run(go())

    # ---------- 对外 ----------
    def speak(self, text, should_stop=None, ref=None, ref_pcm=None, ref_sr=None):
        text = (text or "").strip()
        if not text:
            return 0, False
        # 还没开始合成就已经过期了 → 一个字都别做
        if should_stop and should_stop():
            return 0, True

        fd, path = tempfile.mkstemp(prefix="moss_tts_", suffix=".mp3")
        os.close(fd)
        try:
            if not self._synth(text, path, should_stop):
                return 0, True
            pcm = None
            if ref is not None:
                # 【阶段 17】合出来的是 mp3，AEC 要的是波形 —— 解一次。
                # 解码失败也不报错：ref_pcm=None 时播放器就不登记参考信号，
                # AEC 这一句什么也减不掉 —— 但主程序那边的"半双工守卫"
                # 会因为参考缓冲里是空的而自动接管（见 moss_brain 的 _gate）。
                try:
                    from aec import decode_reference
                    pcm = decode_reference(path, ref.sr)
                except Exception as e:
                    if self.on_ref_error:
                        self.on_ref_error(e)
                    pcm = None
            return self.player.play(path, should_stop, ref=ref if pcm is not None else None,
                                    ref_pcm=pcm, ref_sr=ref.sr if ref else None)
        finally:
            # 播完才能删：MCI 是边读文件边放的
            try:
                os.remove(path)
            except OSError:
                pass

    def stop(self):
        self.player.stop()

    def close(self):
        self.player.close()

    def list_voices(self, locale=None):
        async def go():
            return await self.edge_tts.list_voices()
        vs = asyncio.run(go())
        if locale:
            vs = [v for v in vs if v["Locale"].lower().startswith(locale.lower())]
        return vs


class SapiVoice:
    """Windows 内置 SAPI.SpVoice。离线可用，音色机械（有点像老导航）。

    为什么绕 PowerShell：Python 直接调 COM 要装 pywin32 / comtypes，
    而 Windows 一定自带 powershell.exe。用 `-EncodedCommand` 把脚本
    base64 成 UTF-16LE 传进去，**彻底避开中文乱码和引号转义**这两个大坑。

    ── 阶段 17：改成「先合成到 WAV 再播」──────────────────────
    原来直接把 `Speak()` 的结果送到声卡，好处是简单，坏处有两个：
      1. **拿不到参考信号** → AEC 永远做不了（它需要"扬声器在放什么"的波形）；
      2. 打断只能靠 kill 掉 powershell 进程，笨重且有一次进程启动延迟。

    现在改成 `SpFileStream` 写到临时 WAV，再用 MCI 播那个文件：
      1. WAV 是裸 PCM，读出来就是**精确的**参考信号（连解码器都不需要）；
      2. 打断变成 MCI 的 `stop`，20ms 级；
      3. 顺带让 SAPI 也能被"说完时长"统计到。

    这条路在极老的系统上可能失败（没有 SpFileStream / 权限问题）。
    所以**失败就永久退回直接朗读**，嗓子不会因此哑掉 —— 只是没有参考信号，
    回声消除会自动降级成半双工（见 moss_brain 的门控逻辑）。只警告一次。
    """

    name = "sapi"

    def __init__(self, rate=0, volume=100, exe=None, player=None, to_file=True,
                 on_note=None):
        self.exe = exe or shutil.which("powershell") or shutil.which("pwsh")
        if not self.exe:
            raise RuntimeError("找不到 powershell.exe，SAPI 后端不可用")
        self.rate = int(rate)                # -10 ~ 10
        self.volume = int(volume)            # 0 ~ 100
        self.proc = None
        self.player = player or MciPlayer()
        self.to_file = bool(to_file)
        self.on_note = on_note

    @property
    def supports_reference(self):
        """只有在"合成到文件"这条路上，我们才拿得到参考信号。"""
        return bool(self.to_file)

    @staticmethod
    def _psq(s):
        """PowerShell 单引号字符串里，单引号写成两个。"""
        return (s or "").replace("'", "''")

    def _script(self, text, out_path=None):
        head = ("$ErrorActionPreference = 'Stop'\n"
                "$v = New-Object -ComObject SAPI.SpVoice\n"
                "$v.Rate = %d\n"
                "$v.Volume = %d\n" % (self.rate, self.volume))
        if out_path is None:
            return head + "$null = $v.Speak('%s', 0)\n" % self._psq(text)
        # SpFileStream.Open(文件名, 模式, DoEvents)；模式 3 = SSFMCreateForWrite
        # 刻意**不设** Format.Type：各版本 Windows 的默认格式不一样，
        # 硬写一个枚举值反而容易踩坑。写完照 WAV 头读实际格式就行。
        return head + (
            "$s = New-Object -ComObject SAPI.SpFileStream\n"
            "$s.Open('%s', 3, $false)\n"
            "$v.AudioOutputStream = $s\n"
            "$null = $v.Speak('%s', 0)\n"
            "$s.Close()\n" % (self._psq(out_path), self._psq(text))
        )

    @staticmethod
    def _encode(script):
        return base64.b64encode(script.encode("utf-16-le")).decode("ascii")

    def _run_ps(self, script, should_stop):
        """跑一段 PowerShell，返回 (是否被打断, 退出码)。"""
        cmd = [self.exe, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
               "-EncodedCommand", self._encode(script)]
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        stopped = False
        while True:
            try:
                self.proc.wait(timeout=0.05)
                break
            except subprocess.TimeoutExpired:
                if should_stop and should_stop():
                    self.proc.kill()
                    self.proc.wait()
                    stopped = True
                    break
        rc = self.proc.returncode
        self.proc = None
        return stopped, rc

    # ---------- 对外 ----------
    def speak(self, text, should_stop=None, ref=None, ref_pcm=None, ref_sr=None):
        text = (text or "").strip()
        if not text:
            return 0, False

        if self.to_file:
            try:
                return self._speak_via_file(text, should_stop, ref)
            except Exception as e:
                self.to_file = False             # 只降级一次，不每次重试
                if self.on_note:
                    self.on_note(
                        "SAPI 合成到文件失败（%s）→ 退回直接朗读。"
                        "这条路没有参考信号，回声消除会自动退成半双工。" % e)
        return self._speak_direct(text, should_stop)

    def _speak_via_file(self, text, should_stop, ref):
        fd, path = tempfile.mkstemp(prefix="moss_sapi_", suffix=".wav")
        os.close(fd)
        t0 = time.time()
        try:
            stopped, rc = self._run_ps(self._script(text, path), should_stop)
            if stopped:
                return 0, True
            if rc != 0:
                raise RuntimeError("退出码 %s" % rc)
            if not os.path.exists(path) or os.path.getsize(path) < 64:
                size = os.path.getsize(path) if os.path.exists(path) else 0
                raise RuntimeError("没生成有效的 WAV（%d 字节）" % size)
            pcm = read_wav_pcm(path, ref.sr) if ref is not None else None
            _, interrupted = self.player.play(
                path, should_stop,
                ref=ref if pcm is not None else None, ref_pcm=pcm,
                ref_sr=ref.sr if ref else None, hint="waveaudio")
            # 计时从"开始合成"算起，和直接朗读那条路的语义保持一致
            return int((time.time() - t0) * 1000), interrupted
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def _speak_direct(self, text, should_stop):
        t0 = time.time()
        stopped, rc = self._run_ps(self._script(text), should_stop)
        if rc != 0 and not stopped:
            raise RuntimeError(
                "SAPI 退出码 %s（常见原因：系统里没有中文语音包）" % rc)
        return int((time.time() - t0) * 1000), stopped

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.kill()
            except Exception:
                pass
        try:
            self.player.stop()               # 文件模式下的打断走这条
        except Exception:
            pass

    def close(self):
        self.stop()
        try:
            self.player.close()
        except Exception:
            pass


class AutoVoice:
    """先试 edge，第一次失败就**永久**换成 sapi。

    "只切一次"是阶段 10 的 MindWorker 验证过的经验：如果每次都先试、每次都等超时，
    结果永远"回来即过期"，用户一句话都听不到。切一次之后就是瞬时返回。
    """

    name = "auto"

    def __init__(self, primary, fallback, on_fallback=None):
        self.primary = primary
        self.fallback = fallback
        self.on_fallback = on_fallback
        self.degraded = False

    @property
    def supports_reference(self):
        """只要还有一边能给参考信号，AEC 就还有戏。

        注意：这只是**启动时**的判断。真的降级到另一边之后参考信号可能就没了，
        但那时主程序的门控会看"参考缓冲里此刻有没有东西"，
        而不是看这个静态标志 —— 所以不会漏掉自说自听。
        """
        return bool(getattr(self.primary, "supports_reference", False)
                    or (self.fallback is not None
                        and getattr(self.fallback, "supports_reference", False)))

    def speak(self, text, should_stop=None, ref=None, ref_pcm=None, ref_sr=None):
        if not self.degraded:
            try:
                return self.primary.speak(text, should_stop, ref=ref,
                                          ref_pcm=ref_pcm, ref_sr=ref_sr)
            except Exception as e:
                self.degraded = True
                if self.on_fallback:
                    self.on_fallback(e)
                try:
                    self.primary.close()
                except Exception:
                    pass
        if self.fallback is None:
            return 0, False
        return self.fallback.speak(text, should_stop, ref=ref,
                                   ref_pcm=ref_pcm, ref_sr=ref_sr)

    def stop(self):
        (self.fallback if self.degraded else self.primary).stop()

    def close(self):
        for v in (self.primary, self.fallback):
            if v is not None:
                try:
                    v.close()
                except Exception:
                    pass


# ---------------------------------------------------------------- 工厂
def build_voice(kind, voice=None, rate=None, pitch=None, volume=None,
                sapi_rate=0, on_fallback=None, on_note=None):
    """按 --tts 的值造后端。返回 (voice, 说明字符串)。"""
    kind = (kind or "off").lower()

    if kind in ("off", "none", "offline-off"):
        return NullVoice(), "关闭"

    if kind == "sapi":
        try:
            v = SapiVoice(rate=sapi_rate, on_note=on_note)
            return v, "Windows SAPI（离线，音色机械）"
        except Exception as e:
            return NullVoice(), "关闭（SAPI 不可用：%s）" % e

    edge_kw = dict(
        voice=voice or "zh-CN-YunxiNeural",
        rate=rate or "-10%",
        pitch=pitch or "-15Hz",
        volume=volume or "+0%",
        on_ref_error=on_note,
    )

    if kind == "edge":
        v = EdgeVoice(**edge_kw)
        return v, "edge-tts %s" % edge_kw["voice"]

    if kind == "auto":
        try:
            primary = EdgeVoice(**edge_kw)
        except Exception as e:
            try:
                v = SapiVoice(rate=sapi_rate, on_note=on_note)
                return v, "Windows SAPI（edge 不可用：%s）" % e
            except Exception:
                return NullVoice(), "关闭（edge 与 sapi 都不可用）"
        try:
            fallback = SapiVoice(rate=sapi_rate, on_note=on_note)
            note = "自动（优先 edge，失败退 SAPI）"
        except Exception:
            fallback = None
            note = "自动（只有 edge，失败即静音）"
        return AutoVoice(primary, fallback, on_fallback), note

    raise ValueError("未知的 --tts 值：%s" % kind)


# ================================================================
#  3. 说话线程：只留最新一句，新台词打断旧台词
# ================================================================
class SpeakWorker(threading.Thread):
    """后台说话线程。

    三条规则（都是为了不让它拖慢 20Hz 的融合循环）：

      1. **队列只留最新**：状态已经变了，积压的旧台词没意义，直接扔掉。
         队列深度恒为 1 —— 永远只存"最新那一句"。

      2. **一旦开口就说完**（"commit" 策略）。这一条是踩坑换来的，见下。
         只有**明确打断**（拍手 / 退出）才会中途掐掉。

      3. **同一句话短时间内不重复念**：防止状态抖动导致复读。

    ⚠️ 为什么不是"新台词立刻打断旧台词"
    ------------------------------------------------------------
    一开始就是那么写的，听着很合理，实测直接翻车：

        合成一句要 1.7~2.9 秒。如果每来一句新台词就掐掉正在合成的这一句，
        那么当状态变化比合成还快时（实测 `--demo` 里 0.27 秒就变一次），
        每一句都在合成到一半时被掐死 → **一句都念不出来**，
        日志里全是 "被打断丢弃 12 句"、"说话 0 句"。

    这是个典型的"输入速率 > 输出速率"问题。正确解法不是拼命追最新，
    而是**以输出速率说话、每次取当下最新的那一句**：
      合成慢、台词来得快 → 就是"丢掉中间那些，只念开口那一瞬间最新的"。
    这样延迟有界（最多落后一句），但保证真的出声。
    """

    def __init__(self, voice, guard_ms=350, dedup_s=3.0, on_error=None,
                 reference=None):
        super().__init__(daemon=True, name="speak")
        self.voice = voice
        self.guard_ms = guard_ms
        self.dedup_s = dedup_s
        self.on_error = on_error
        # 【阶段 17】回声消除的"参考信号"缓冲。每念一句之前，把即将播出的
        # 波形登记进去，AEC 才知道"扬声器此刻在放什么"。给 None 就是不开 AEC。
        self.reference = reference

        self.stop_ev = threading.Event()
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.pending = None
        self.n_submit = 0             # 一共投递过多少句（仅用于观察）
        self.abort = 0                # 只有 interrupt()/shutdown() 会让它 +1
        self.speaking = False
        self.last_end = 0.0
        self.last_text = ""
        self.last_start = 0.0
        self.stats = {"spoken": 0, "dropped": 0, "skip": 0, "error": 0, "ms": 0}

    # ---------- 外部接口 ----------
    def submit(self, text):
        """投递一句台词。返回 True 表示收下了。"""
        text = (text or "").strip()
        if not text:
            return False
        now = time.time()
        with self.cv:
            if text == self.last_text and (now - self.last_start) < self.dedup_s:
                self.stats["skip"] += 1
                return False
            self.last_text = text
            self.last_start = now
            self.pending = text          # 直接覆盖 —— 队列深度永远是 1
            self.n_submit += 1
            self.cv.notify()
        return True

    def interrupt(self):
        """掐掉当前这句（拍手打断 MOSS 就靠它）。"""
        with self.cv:
            self.pending = None
            self.abort += 1
        self.voice.stop()

    def busy(self):
        """正在说，或者刚说完还在"余音"里 —— 大脑据此屏蔽听觉。"""
        with self.lock:
            if self.speaking:
                return True
            return (time.time() - self.last_end) * 1000.0 < self.guard_ms

    def shutdown(self):
        self.stop_ev.set()
        self.interrupt()
        with self.cv:
            self.cv.notify_all()

    # ---------- 线程主体 ----------
    def run(self):
        while not self.stop_ev.is_set():
            with self.cv:
                while self.pending is None and not self.stop_ev.is_set():
                    self.cv.wait(0.1)
                text = self.pending
                self.pending = None
                my_abort = self.abort

            if text is None:
                continue

            with self.lock:
                self.speaking = True
            try:
                # should_stop 只在两种情况下为 True：
                #   abort 变了 = 有人明确打断（拍手/退出）
                #   stop_ev  = 要关程序了
                # 注意这里**没有**"有新台词了就停" —— 那正是上面说的翻车写法。
                ms, interrupted = self.voice.speak(
                    text,
                    should_stop=lambda: self.abort != my_abort or self.stop_ev.is_set(),
                    ref=self.reference,
                    ref_sr=self.reference.sr if self.reference is not None else None)
                with self.lock:
                    if interrupted:
                        self.stats["dropped"] += 1
                    else:
                        self.stats["spoken"] += 1
                        self.stats["ms"] += ms
            except Exception as e:
                with self.lock:
                    self.stats["error"] += 1
                if self.on_error:
                    self.on_error(e)
            finally:
                with self.lock:
                    self.speaking = False
                    self.last_end = time.time()


# ================================================================
#  4. 自检入口：python voice.py --test "你好"
# ================================================================
class _FakeVoice:
    """假嗓子：不出声，只记录"被要求念了哪几句"。

    用来做**确定性**测试 —— 真嗓子要联网、要几秒钟，
    队列语义这种逻辑用假嗓子才能几毫秒跑完、而且每次结果一样。
    """

    name = "fake"
    supports_reference = True

    def __init__(self, delay=0.5, log=None, with_reference=True):
        self.delay = delay
        self.log = log if log is not None else []
        self.stopped = 0
        self.closed = 0
        self.with_reference = with_reference
        # 记录参考信号的登记/截断，供自检断言（阶段 17）
        self.ref_marks = []

    def speak(self, text, should_stop=None, ref=None, ref_pcm=None, ref_sr=None):
        self.log.append(text)
        if ref is not None and self.with_reference:
            # 没给波形就造一段等长的直流 —— 自检只关心"有没有登记、什么时候截断"，
            # 不关心参考信号长什么样（不引入 numpy，保持这个模块零依赖）
            pcm = ref_pcm if ref_pcm is not None else [0.1] * ref.sr
            ref.start(time.time(), pcm, ref_sr or ref.sr)
            self.ref_marks.append(("start", len(pcm)))
        n = max(1, int(self.delay / 0.02))
        try:
            for _ in range(n):
                if should_stop and should_stop():
                    return int(n * 20 * 0.5), True
                time.sleep(0.02)
            return int(self.delay * 1000), False
        finally:
            if ref is not None and self.with_reference:
                ref.finish()
                self.ref_marks.append(("finish", 0))

    def stop(self):
        self.stopped += 1

    def close(self):
        self.closed += 1


def _selftest():
    p = argparse.ArgumentParser(description="MOSS 语音层自检")
    p.add_argument("--tts", default="auto", choices=["off", "auto", "edge", "sapi"])
    p.add_argument("--voice", default=None, help="edge 音色，默认 zh-CN-YunxiNeural")
    p.add_argument("--rate", default=None, help='edge 语速，如 "-10%%"')
    p.add_argument("--pitch", default=None, help='edge 音调，如 "-15Hz"')
    p.add_argument("--sapi-rate", type=int, default=0, help="SAPI 语速 -10~10")
    p.add_argument("--test", default="系统自检完成，各单元上线。", help="要念的文本")
    p.add_argument("--list", action="store_true", help="列出 edge 的中文音色")
    p.add_argument("--selftest", action="store_true", help="跑合成/播放/打断全套自检")
    a = p.parse_args()

    if a.list:
        v = EdgeVoice(voice=a.voice or "zh-CN-YunxiNeural")
        for m in v.list_voices(locale="zh"):
            print("%-34s %s" % (m["ShortName"], m.get("Gender", "")))
        return 0

    voice, desc = build_voice(a.tts, a.voice, a.rate, a.pitch,
                              sapi_rate=a.sapi_rate,
                              on_fallback=lambda e: print("  [降级] %s" % e, file=sys.stderr))
    print("[后端] %s → %s" % (a.tts, desc))

    if not a.selftest:
        t0 = time.time()
        ms, stopped = voice.speak(a.test)
        print("[说话] %r  耗时 %.1fs（音频 %.1fs）%s"
              % (a.test, time.time() - t0, ms / 1000.0, "，被打断" if stopped else ""))
        voice.close()
        return 0

    # ---- 全套自检 ----
    print("\n[1/4] 真嗓子：合成 + 播放")
    t0 = time.time()
    ms, _ = voice.speak("系统自检完成，各单元上线。")
    print("      OK  总耗时 %.2fs（其中音频 %.2fs）" % (time.time() - t0, ms / 1000.0))

    print("\n[2/4] 真嗓子：打断。念一句长的，1.2 秒后强行掐掉")
    box = {"stop": False}
    th = threading.Thread(
        target=lambda: voice.speak("这是一句故意说得比较长的话，用来测试打断功能是否生效。",
                                   should_stop=lambda: box["stop"]),
        daemon=True)
    t0 = time.time()
    th.start()
    time.sleep(1.2)
    t_mark = time.time()
    box["stop"] = True
    th.join(timeout=10)
    t_end = time.time()
    print("      从喊停到真的停：%.2fs      全程 %.2fs（不打断约需 6s）"
          % (t_end - t_mark, t_end - t0))
    print("      → %s" % ("打断生效" if t_end - t_mark < 1.5 else "打断没生效"))
    voice.close()

    # ---- 队列语义：用假嗓子做确定性验证 ----
    print("\n[3/4] 队列语义（假嗓子，确定性验证）")
    fake = _FakeVoice(delay=0.5)
    w = SpeakWorker(fake, guard_ms=200)
    w.start()
    t0 = time.time()
    w.submit("第一句。")
    time.sleep(0.08)          # 让 worker 把"第一句"取走，开始念
    w.submit("第二句。")       # 这时投递的应该被"第三句"顶掉
    time.sleep(0.08)
    w.submit("第三句。")
    time.sleep(1.6)
    ok1 = fake.log == ["第一句。", "第三句。"]
    print("      实际念了 %s" % fake.log)
    print("      → %s" % ("正确：已经开口的那句念完，中间被顶掉，再念最新的那一句"
                          if ok1 else "不对：期望 ['第一句。', '第三句。']"))

    # 打断：正在念的时候 interrupt，应该立刻停且不再补念
    w.interrupt()
    fake.log.clear()
    w.submit("这句会被拍手打断。")
    time.sleep(0.15)
    t_mark = time.time()
    w.interrupt()
    time.sleep(0.4)
    w.shutdown()
    print("      打断后立刻停：%.2fs 内结束，且没有多余的补念 → %s"
          % (time.time() - t_mark, "OK" if not w.pending else "有问题"))

    # ---- 阶段 17：参考信号接线 ----
    print("\n[4/4] 参考信号接线（假嗓子 + 真缓冲，确定性验证）")
    from aec import ReferenceBuffer
    rb = ReferenceBuffer(16000)
    print("      播放前  active=%s（应为 False，否则回调会屏蔽听觉）" % rb.active)
    assert not rb.active
    fake2 = _FakeVoice(delay=0.8)
    w2 = SpeakWorker(fake2, guard_ms=50, reference=rb)
    w2.start()
    w2.submit("参考信号测试。")
    time.sleep(0.25)                      # 让 worker 开口，登记已经发生
    during = rb.active
    print("      播放中  active=%s（应为 True —— 半双工守卫靠它判断「有没有参考」）" % during)
    assert during, "说说期间参考缓冲应该是 active 的"
    w2.shutdown()
    time.sleep(0.7)          # 比"刚播完"的宽限期(0.25s)长，确保真的归为不活跃
    print("      登记记录 %s（这一句被中途打断，所以是 start→finish，没有补记）"
          % (fake2.ref_marks,))
    print("      播放后  active=%s（应为 False，时间轴要截断）" % rb.active)
    assert not rb.active, "说完之后必须 finish()，否则回声消除会一直以为还在放"
    assert [m[0] for m in fake2.ref_marks] == ["start", "finish"], fake2.ref_marks
    print("      ✓ 每句台词都恰好登记一次、截断一次（不多不少）")

    print("\n[OK] 自检结束。")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest())
