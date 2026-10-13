#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOSS 多模态融合大脑  ·  流浪地球 550W 复刻项目 · 阶段 9
============================================================
阶段 7 给了 MOSS「眼睛」（moss_vision.py），阶段 8 给了它「耳朵」
（moss_audio.py）。但它们各跑各的，而且都抢同一个串口 —— 同一时刻只能开一个。

这个程序把两者合成一个大脑：
    眼睛线程（摄像头）──┐
                        ├──► 融合判断 ──► 一条串口 ──► ESP32
    耳朵线程（麦克风）──┘

融合之后 MOSS 才有真正的"多模态判断"：
    看见人 + 听见说话   → SPEAK   （有人在跟我讲话）
    只看见人            → VISION  （有人在，我盯着看）
    只听见声音          → LISTEN  （有动静，我在听）
    只看见画面在动      → MOTION  （有东西在动）
    什么都没有          → IDLE    （待机）

关键点：**视觉和听觉必须并发**。摄像头 read() 会阻塞（约 30fps），麦克风
read() 也会阻塞（约 15 块/秒）。串行做的话，等画面的 33 毫秒里就漏掉了声音。
所以这里用两个线程各管一路，主线程只做"读快照 → 融合 → 输出"。

典型用法：
    python moss_brain.py                      # 真摄像头 + 真麦克风 + 预览窗
    python moss_brain.py --demo               # 全合成，没摄像头没麦克风也能跑
    python moss_brain.py --auto               # 自动挑一个有信号的麦克风
    python moss_brain.py --port rfc2217://localhost:4000    # 接 Wokwi 仿真
    python moss_brain.py --no-window --frames 400           # 无窗口自测

输出示例：
    {"t": 3.21, "faces": 1, "motion": 0.004, "dbfs": -28.4, "floor": -59.1,
     "clap": false, "mode": "SPEAK", "text": "face+voice", "sent": true}
"""

import argparse
import collections
import json
import math
import os
import queue
import sys
import threading
import time
import wave

# ---------------------------------------------------------------- 依赖检查
try:
    import numpy as np
except ImportError:
    sys.exit("[缺少依赖] 没装 NumPy。请在终端执行：pip install numpy")

try:
    import cv2
except ImportError:
    sys.exit("[缺少依赖] 没装 OpenCV。请在终端执行：pip install opencv-python")

try:
    import sounddevice as sd
except ImportError:
    sys.exit("[缺少依赖] 没装 sounddevice。请在终端执行：pip install sounddevice")

try:
    import serial  # 来自 pyserial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# 语言层（阶段 10）。找不到也不影响感知功能，只是 --llm 会被忽略。
try:
    from mind import build_mind, MindWorker, DEFAULT_PERSONA
    HAS_MIND = True
except ImportError:
    HAS_MIND = False

# 语音层（阶段 11）。同上：没有它只是不出声，感知照常。
try:
    from voice import build_voice, SpeakWorker
    HAS_VOICE = True
except ImportError:
    HAS_VOICE = False

# 听觉理解层（阶段 12）。同上：没有它只是"听不懂你说什么"，其余照常。
try:
    from asr import UtteranceSegmenter, AsrWorker, build_engine, DEFAULT_ROOT as ASR_ROOT
    HAS_ASR = True
except ImportError:
    HAS_ASR = False

# 长期记忆（阶段 13）。没有它只是"每次启动都重新认识你"，其余照常。
try:
    from memory import MemoryStore
    HAS_MEMORY = True
except ImportError:
    HAS_MEMORY = False

# 工具层（阶段 13）。没有它只是"只会聊、不会办事"，其余照常。
try:
    from moss_tools import ToolBox
    HAS_TOOLS = True
except ImportError:
    HAS_TOOLS = False

# 执行器层（阶段 14）。没有它只是"读得懂世界、但动不了世界"，其余照常。
try:
    from moss_actuators import ActuatorLink, install as install_actuators
    HAS_ACTUATORS = True
except ImportError:
    HAS_ACTUATORS = False

# 唤醒门（阶段 15）。没有它只是"听见什么都要答一句"，其余照常。
try:
    from wake import WakeGate
    HAS_WAKE = True
except ImportError:
    HAS_WAKE = False

# 主动行为层（阶段 16）。没有它只是"你不开口它就不动"，其余照常。
try:
    from behave import BehaviorEngine
    HAS_BEHAVE = True
except ImportError:
    HAS_BEHAVE = False

# 回声消除（阶段 17）。没有它只是"它一开口就听不见你"（半双工），其余照常。
try:
    from aec import EchoCanceller, ReferenceBuffer
    HAS_AEC = True
except ImportError:
    HAS_AEC = False

# 视觉层（阶段 18）。没有它只是退回"只认人脸"，其余照常。
# ⚠️ 注意这里**只导入不构造** —— 真正的检测器在 main() 里按 --vision-model 造，
# 因为"下没下模型 / 装没装 onnxruntime"要等运行时才知道，
# 而且构造失败**不能**让整个 MOSS 起不来（视觉降级，听觉和对话还得能用）。
try:
    from vision import (make_detector, summarize as vision_summarize,
                        labels_line, count_by_name as vision_count_by_name,
                        zh_name, PERSON_LIKE as VISION_PERSON_LIKE,
                        MODELS as VISION_MODELS,
                        list_models as vision_list_models,
                        download_model as vision_download_model,
                        model_dir as vision_model_dir)
    HAS_VISION = True

    # 哪些"类"算作"有人"。以 vision.py 为准，不在主程序里再抄一份 ——
    # 抄两份迟早会漂移（这个项目已经因为"判据抄两份"吃过一次亏，见阶段 17）。
    PERSON_LIKE_NAMES = tuple(VISION_PERSON_LIKE)
except ImportError:
    HAS_VISION = False
    # 万一 vision.py 整个 import 不进来（比如连 numpy 都没装），
    # "有没有人"这条**最关键**的判断不能跟着一起消失 —— 给一份最小兜底。
    PERSON_LIKE_NAMES = ("person", "face")

# 变化感知层（阶段 19）。没有它只是"东西变了它不吭声"，人/拍手照常。
# ⚠️ 和 behavior 一样，这一层的产物是"往设备发命令 / 说一句话"，
# 所以只有真接上设备时才值得建（见 main() 里的判断）。
try:
    from presence import PresenceTracker
    HAS_PRESENCE = True
except ImportError:
    HAS_PRESENCE = False

# 串口是"一票资源"：主循环发 MODE/TEXT、执行器工具发 LED/SPIN/BEEP，
# 两条线程写同一个口。pyserial 的 write 不保证原子，所以必须共用一把锁。
_IO_LOCK = threading.RLock()


# ================================================================
#  1. 命令行参数
# ================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="MOSS 多模态大脑 —— 摄像头 + 麦克风同时跑，融合成一个状态，推送串口"
    )
    # ---------- 总开关 ----------
    p.add_argument("--demo", action="store_true",
                   help="全合成模式：画面和声音都用假数据（没有摄像头/麦克风时用它验证链路）")

    # ---------- 视觉 ----------
    p.add_argument("--vision-source", default="0",
                   help="画面来源：0=摄像头 / demo=合成画面 / 视频或图片文件路径")
    p.add_argument("--width", type=int, default=640, help="采集宽度，默认 640")
    p.add_argument("--height", type=int, default=480, help="采集高度，默认 480")
    p.add_argument("--motion-th", type=float, default=0.008,
                   help="运动判定阈值 0~1，默认 0.008（画面变化面积占比）")
    # ---- 【阶段 18】强视觉：认出画面里的东西 ----
    p.add_argument("--vision-model", default="auto",
                   choices=["auto", "yolo", "haar", "null", "fake"],
                   help="检测后端：auto=有模型就 yolo / yolo / haar（只认人脸）/ "
                        "null（只测运动）/ fake（脚本，测试用）。默认 auto")
    p.add_argument("--vision-model-path", default=None,
                   help="直接指定 .onnx 文件（绕过模型目录）")
    p.add_argument("--vision-dir", default=None,
                   help=f"模型目录，默认 {vision_model_dir() if HAS_VISION else '~/.cache/moss_vision'}")
    p.add_argument("--vision-mirror", default="hf-mirror.com",
                   help="模型镜像站，默认 hf-mirror.com（国内直连 huggingface 不通）")
    p.add_argument("--vision-download", action="store_true",
                   help="下模型然后退出（auto 不会替你偷偷下 12MB）")
    p.add_argument("--vision-list", action="store_true",
                   help="列出可用模型然后退出")
    p.add_argument("--vision-conf", type=float, default=0.35,
                   help="目标检测置信度阈值，默认 0.35")
    p.add_argument("--vision-iou", type=float, default=0.45,
                   help="NMS 的 IoU 阈值，默认 0.45")
    p.add_argument("--vision-imgsz", type=int, default=640, help="模型输入边长，默认 640")
    p.add_argument("--vision-threads", type=int, default=2,
                   help="ONNX 推理线程数，默认 2。别贪多 —— 音频要实时，"
                        "视觉掉到 15fps 没人觉得卡，音频掉一块就是漏听半句话")
    p.add_argument("--vision-detect-every", type=int, default=3,
                   help="每 N 帧跑一次检测，默认 3（运动检测仍然每帧都跑）。"
                        "640×480 下 YOLOv8n 约 60ms 一帧，全跑会把视觉线程压到 15fps")
    p.add_argument("--vision-max-obj", type=int, default=12,
                   help="一帧最多报几个目标，默认 12")

    # ---------- 阶段 19：画面里的东西变了要吭声 ----------
    # 为什么这几个门槛做成参数：它们**没法离线定死**。
    # 检测器有多抖、光照多差、桌上东西多密 —— 只有真机现场才知道。
    # 默认值是按"能忍的噪声下限"给的，宁可钝一点。
    p.add_argument("--no-objects", action="store_true",
                   help="【阶段19】关掉「东西变了」的感知（默认开）。"
                        "关掉只是不再跟踪变化，识别本身和 look 工具都不受影响")
    p.add_argument("--objects-enter", type=int, default=None,
                   help="【阶段19】连续几次**检测**看到才算「在场」，默认 3。"
                        "单位是检测次数不是帧 —— 否则一改 --vision-detect-every 就偷改了语义")
    p.add_argument("--objects-leave", type=int, default=None,
                   help="【阶段19】连续几次漏检才算「不在场」，默认 7（刻意比进入钝）。"
                        "晚说一句只是迟钝，把「手挡了一下」说成「东西没了」是编造事实")
    p.add_argument("--objects-count", type=int, default=None,
                   help="【阶段19】个数变化要连续确认几次，默认 3（个数是二阶信息，更抖）")
    p.add_argument("--objects-warmup", type=int, default=None,
                   help="【阶段19】开机后先看几次只建基线、不报变化，默认 10。"
                        "没它的话它一启动就报一屋子本来就在那儿的东西")

    # ---------- 听觉 ----------
    p.add_argument("--audio-source", default="mic",
                   help="声音来源：mic=麦克风 / demo=合成音频 / 路径.wav=录音文件")
    p.add_argument("--device", default=None,
                   help="麦克风：留空=系统默认；也可填编号或名字的一部分")
    p.add_argument("--auto", action="store_true",
                   help="自动试听并挑一个真正有信号的麦克风（见 pc_audio/README 第三节）")
    p.add_argument("--probe-seconds", type=float, default=0.4,
                   help="自动挑设备时每个试听多久，默认 0.4 秒")
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

    # ---------- 串口 ----------
    p.add_argument("--port", default=None,
                   help="串口。COM3=真机；rfc2217://localhost:4000=Wokwi 仿真；不给就不发")
    p.add_argument("--baud", type=int, default=115200, help="串口波特率，默认 115200")

    # ---------- 主循环 ----------
    p.add_argument("--rate", type=float, default=20.0,
                   help="融合频率(Hz)，默认 20（每秒做 20 次融合判断）")
    p.add_argument("--stable", type=int, default=4,
                   help="连续 N 次融合结果一致才切状态，默认 4（防抖动）")
    p.add_argument("--heartbeat", type=float, default=2.0,
                   help="状态没变时，每隔几秒也输出一行 JSON，默认 2 秒")
    p.add_argument("--no-window", action="store_true", help="不弹预览窗口（纯输出 JSON）")
    p.add_argument("--frames", type=int, default=0,
                   help="主循环跑多少次后退出，0=一直跑（自测用）")
    p.add_argument("--verbose", action="store_true", help="每次循环都输出 JSON")

    # ---------- 语言层（阶段 10）：让 MOSS 说人话 ----------
    p.add_argument("--llm", default="off", choices=["off", "offline", "api", "auto"],
                   help="台词后端：off=关 / offline=内置模板(零配置) / "
                        "api=OpenAI兼容接口 / auto=先试api失败退回模板")
    p.add_argument("--api-base",
                   default=os.environ.get("MOSS_API_BASE", "https://api.deepseek.com/v1"),
                   help="OpenAI 兼容接口地址，默认 DeepSeek（也可指向通义/智谱/本地 ollama）")
    p.add_argument("--api-key", default=None,
                   help="API Key；也可设环境变量 MOSS_API_KEY / OPENAI_API_KEY")
    p.add_argument("--model", default=os.environ.get("MOSS_MODEL", "deepseek-chat"),
                   help="模型名，默认 deepseek-chat")
    p.add_argument("--llm-timeout", type=float, default=8.0,
                   help="单次大模型请求超时(秒)，默认 8")
    p.add_argument("--persona", default=None, help="角色设定，默认就是 MOSS 本人")
    p.add_argument("--llm-test", action="store_true",
                   help="只测语言层（不打开摄像头/麦克风），打印一句台词后退出")

    # ---------- 语音层（阶段 11）：让 MOSS 真的出声 ----------
    p.add_argument("--tts", default="sapi", choices=["off", "auto", "edge", "sapi"],
                   help="嗓子后端：off=不出声 / auto=先试edge失败退sapi / "
                        "edge=edge-tts在线神经语音(音质最好) / sapi=Windows内置(离线)")
    p.add_argument("--tts-voice", default=None,
                   help="edge 音色，默认 zh-CN-YunxiNeural（--tts-list 可看全部）")
    p.add_argument("--tts-rate", default=None, help='edge 语速，默认 "-10%%"（慢一点更像MOSS）')
    p.add_argument("--tts-pitch", default=None, help='edge 音调，默认 "-15Hz"（压低更冷）')
    p.add_argument("--tts-sapi-rate", type=int, default=0,
                   help="SAPI 语速 -10~10，默认 0")
    p.add_argument("--tts-guard", type=float, default=350.0,
                   help="说完后继续屏蔽听觉的毫秒数，默认 350（防风箱混响自激）")
    p.add_argument("--tts-dedup", type=float, default=3.0,
                   help="同一句话多少秒内不重复念，默认 3 秒（防状态抖动复读）")
    p.add_argument("--barge-in", action="store_true",
                   help="允许拍手打断 MOSS 说话。默认关闭 —— 因为麦克风分不清"
                        "「你拍的」和「MOSS 自己说话的开头」，容易自己打断自己")
    p.add_argument("--no-echo-guard", action="store_true",
                   help="关掉「说话时屏蔽听觉」的自说自听抑制。一般不用关，"
                        "只有戴耳机、确定音箱声音不会被麦克风收进去时才关")
    p.add_argument("--tts-test", action="store_true",
                   help="只测嗓子（不打开摄像头/麦克风），说一句后退出")
    p.add_argument("--tts-list", action="store_true",
                   help="列出 edge 可用的中文音色后退出")

    # ---------- 听觉理解层（阶段 12）：让 MOSS 听懂你说什么 ----------
    p.add_argument("--asr", default="whisper", choices=["off", "auto", "whisper", "api"],
                   help="语音识别：off=只听音量 / whisper=本地faster-whisper / "
                        "api=OpenAI兼容接口 / auto=先试whisper失败退api")
    p.add_argument("--asr-model", default="small",
                   help="whisper 模型 tiny/base/small/medium，默认 small（首次要下 484MB）")
    p.add_argument("--asr-dir", default=None, help="模型目录，默认 ~/.cache/moss_asr")
    p.add_argument("--asr-mirror", default=None,
                   help="模型下载镜像，默认 https://hf-mirror.com（huggingface.co 国内不通）")
    p.add_argument("--asr-device", default="cpu", help="cpu 或 cuda")
    p.add_argument("--asr-compute-type", default="int8", help="int8 / int8_float32 / float32")
    p.add_argument("--asr-lang", default="zh", help="识别语言，默认 zh（留空=自动）")
    p.add_argument("--asr-beam", type=int, default=5, help="beam size，默认 5")
    p.add_argument("--asr-api-base", default=os.environ.get("MOSS_ASR_BASE",
                                                            "https://api.openai.com/v1"),
                   help="ASR 接口地址（OpenAI 兼容 /audio/transcriptions）")
    p.add_argument("--asr-api-key", default=None,
                   help="ASR 的 Key；也可设环境变量 MOSS_ASR_KEY")
    p.add_argument("--asr-api-model", default="whisper-1", help="ASR 接口的模型名")
    p.add_argument("--asr-end-ms", type=float, default=700.0,
                   help="静音多少毫秒算一句话说完，默认 700（太小会把停顿切成两句）")
    p.add_argument("--asr-min-ms", type=float, default=250.0,
                   help="有效说话时长不足这么多毫秒就丢弃，默认 250（挡咳嗽/关门声）")
    p.add_argument("--asr-queue", type=int, default=2, help="等待识别的句子最多排几段，默认 2")
    p.add_argument("--asr-history", type=int, default=3,
                   help="记住最近几轮对话喂给大模型，默认 3（0=不记）")
    p.add_argument("--asr-hold", type=float, default=1.2,
                   help="问完话后多少秒内不再自言自语，默认 1.2（别把你的回答顶掉）")
    p.add_argument("--asr-test", action="store_true",
                   help="只测识别（不打开摄像头），把一句话识别出来就退出")

    # ---------- 有了 ASR 之后，MOSS 该少说多听 ----------
    p.add_argument("--narrate", default="quiet", choices=["auto", "all", "quiet"],
                   help="状态变化时要不要自言自语：all=每次都说（阶段10/11行为）/ "
                        "quiet=只在有人来·有人说话·拍手·出错时说，且限速 / "
                        "auto=有ASR时用quiet，没ASR时用all（默认）")
    p.add_argument("--narrate-gap", type=float, default=9999.0,
                   help="quiet 模式下两次开口至少隔多少秒，默认 9999（=开机说一次后基本闭嘴）")

    # ---------- 阶段 13：长期记忆 + 工具 ----------
    p.add_argument("--no-memory", action="store_true",
                   help="关掉长期记忆（回到阶段 12 行为：不落盘、关掉再开就忘了）")
    p.add_argument("--memory-file", default=None,
                   help="记忆文件路径，默认 pc_brain/data/moss_memory.json")
    p.add_argument("--memory-turns", type=int, default=None,
                   help="喂给大模型的对话轮数，默认跟 --asr-history 一致")
    p.add_argument("--no-tools", action="store_true",
                   help="关掉工具调用（模型不能查时间/天气、不能主动记东西）")
    p.add_argument("--no-actuators", action="store_true",
                   help="【阶段14】关掉执行器（模型不能动灯环/电机/蜂鸣器/屏幕）")
    p.add_argument("--tool-max-rounds", type=int, default=1,
                   help="一个回合里最多连续调几次工具，默认 1（防死循环）")

    # ---------- 阶段 15：唤醒词（叫名字才答应）----------
    # 默认跟着 ASR 走：开了 --asr 就默认开唤醒门。
    # 因为"麦克风常开、谁说话都答"正是这一步要解决的问题，
    # 要退回旧行为就显式加 --no-wake。
    p.add_argument("--no-wake", action="store_true",
                   help="【阶段15】关掉唤醒门：任何人说话都回应（回到阶段 14 的行为）")
    p.add_argument("--wake-word", default=None,
                   help="唤醒词，逗号分隔可给多个；默认 moss（自带莫斯/摩斯等同音别名）")
    p.add_argument("--wake-window", type=float, default=20.0,
                   help="被叫醒后多少秒内不用再喊名字，默认 20（每说一句会续期）")
    p.add_argument("--wake-strict", action="store_true",
                   help="只认句首（「MOSS，几点了」）。关掉「句中提到名字也算唤醒」——"
                        "适合不希望闲聊时被误唤醒的场景")
    p.add_argument("--wake-ack", default="voice", choices=["off", "beep", "voice"],
                   help="被叫醒时怎么应一声：off=只打日志（默认）/ beep=设备蜂鸣一下 / "
                        "voice=说一句'在'（注意：会让麦克风屏蔽一小会儿）")

    # ---------- 阶段 16：主动行为（它自己也会动）----------
    p.add_argument("--autonomy", default=None, choices=["off", "l0", "l1"],
                   help="【阶段16】它自己能做多少事：off=完全被动 / "
                        "l0=只转头+改灯色（默认，不发出声音）/ l1=还允许蜂鸣")
    p.add_argument("--no-proactive", action="store_true",
                   help="等价 --autonomy off")
    p.add_argument("--proactive-gap", type=float, default=6.0,
                   help="两次主动动作之间至少隔多少秒，默认 6（防抽风）")

    # ---------- 阶段 17：回声消除（它说话时你也能插嘴）----------
    p.add_argument("--aec", default="off", choices=["off", "on", "auto"],
                   help="【阶段17】回声消除：auto=有嗓子且拿得到参考信号就开（默认）/ "
                        "on=强制开（拿不到参考会退化回半双工）/ off=关")
    p.add_argument("--aec-taps-ms", type=float, default=250.0,
                   help="自适应滤波器长度(毫秒)，默认 250。要同时覆盖"
                        "「声卡往返时延 + 房间混响」，无线耳机/蓝牙音箱这类"
                        "高时延设备可以加到 300~400")
    p.add_argument("--aec-subblock", type=int, default=64,
                   help="内部更新块长(样本)，默认 64（16kHz 下 4ms）。"
                        "调大略省 CPU，但收敛变慢")
    p.add_argument("--aec-mu", type=float, default=None,
                   help="步长，默认 0.5。调大收敛快但容易抖；"
                        "--selftest 里 0.5~0.8 是最好的区间")
    p.add_argument("--aec-dtd", type=float, default=2.0,
                   help="双讲判据倍数，默认 2.0。它误判成「你在说话」→ 噪声变大；"
                        "漏判 → 滤波器被你的声音带坏。听到自己回声时先调小")
    p.add_argument("--aec-resid", type=float, default=10.0,
                   help="残余回声抑制(dB)，默认 10。确认「只有自己在说」时才压，"
                        "所以调大不会吃掉你插话的声音；调到 0 就关掉这一级")
    return p.parse_args()


# ================================================================
#  2. 串口（和 moss_vision / moss_audio 用同一套协议）
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
        print("       → 继续只输出 JSON，识别不受影响。", file=sys.stderr)
        return None


def send_raw(ser, line):
    """发一行串口命令。非 ASCII 字符会被丢弃（固件端只认 ASCII）。

    注意 _IO_LOCK：阶段 14 起执行器工具也会往同一个口写字，
    不加锁两条线程的 write 会交错成半截命令。
    """
    if ser is None:
        return False
    try:
        with _IO_LOCK:
            ser.write((line + "\n").encode("ascii", "ignore"))
            ser.flush()
        return True
    except Exception as e:
        print(f"[提示] 串口写入失败：{e}", file=sys.stderr)
        return False


def send_state(ser, mode, text):
    """把状态推给 ESP32。协议就是固件里那几个串口命令。"""
    if ser is None:
        return False
    ok = send_raw(ser, f"MODE {mode}")
    send_raw(ser, f"TEXT {text}")
    return ok


# ================================================================
#  3. 视觉一路（画面来源 + 检测 + 工作线程）
# ================================================================
class DemoScene:
    """没有摄像头时的替身：画一个会动的小球。

    它会「动 2 秒、停 1 秒」地循环，这样你能看到状态在 MOTION / IDLE 之间跳。

    ⚠️ 合成画面里是**画出来的圆**，不是真实物体：
      · 人脸检测（Haar）认不出它 → `faces` 永远是 0
      · 目标检测（YOLO）也认不出它 → `objects` 永远是空
    想在 `--demo` 下把"认出东西"这条链也走通，加 `--vision-model fake`：
    那是脚本化的假检测器，框是**写死的**（见 vision.py 的 FakeDetector），
    专门用来离线验证"检测结果 → 融合 → look 工具 → 主动行为"这一段。
    """

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.i = 0
        self.x, self.y = w // 2, h // 2

    def read(self):
        self.i += 1
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        img[:] = (28, 18, 10)                                  # 深蓝底（BGR）

        phase = self.i % 160
        if phase < 100:                                        # 前 100 帧：动
            self.x = int(self.w / 2 + (self.w / 3.0) * np.sin(self.i / 12.0))
            self.y = int(self.h / 2 + (self.h / 4.0) * np.sin(self.i / 9.0))
            tag = "MOVING"
        else:                                                  # 后 60 帧：停
            tag = "STILL"

        cv2.putText(img, "MOSS BRAIN  [DEMO]", (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (210, 230, 255), 2)
        cv2.putText(img, tag, (20, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
        cv2.circle(img, (self.x, self.y), 70, (255, 160, 60), -1)
        return True, img

    def release(self):
        pass


class FileScene:
    """视频文件 / 图片文件。图片会被反复复制成帧流。"""

    def __init__(self, path, w, h):
        self.is_image = path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        self.cap = None
        self.img = None
        if self.is_image:
            self.img = cv2.imread(path)
            if self.img is None:
                raise FileNotFoundError(path)
            self.img = cv2.resize(self.img, (w, h))
        else:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise FileNotFoundError(path)

    def read(self):
        if self.is_image:
            return True, self.img.copy()
        ok, frame = self.cap.read()
        if not ok:                       # 视频放完了 → 倒回开头，循环播
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return ok, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()


class CameraScene:
    """真实摄像头。"""

    def __init__(self, index, w, h):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == "nt" else 0)
        if not self.cap.isOpened():      # 有些机器用 CAP_DSHOW 打不开，退回默认后端
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"打不开摄像头 {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 只留 1 帧，避免画面延迟

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


def make_vision_source(spec, w, h):
    if spec == "demo":
        print("[OK] 视觉：合成画面（demo）", file=sys.stderr)
        return DemoScene(w, h)
    if str(spec).isdigit():
        return CameraScene(int(spec), w, h)
    return FileScene(spec, w, h)


# 阶段 9 的 load_face_cascade() 已经搬进 vision.py 的 HaarDetector ——
# 那不是简单的搬家：以前"人脸检测"是主程序里的一段 if，现在它是
# "四个可替换后端之一"，主程序不再知道框是谁给的。
# 留着两份实现只会漂移（这个项目已经因为"判据抄两份"吃过亏，见阶段 17）。


class VisionWorker(threading.Thread):
    """视觉线程：不断读画面、检测**物体**和运动，把最新结果放进快照。

    为什么单独开一个线程：摄像头 read() 会阻塞。如果放在主循环里，
    等画面的这段时间就没法去读麦克风了。两路必须并发。

    【阶段 18】检测这件事从"内置的 Haar 级联"换成了**一个可替换的检测器对象**
    （`vision.py` 里的 Detector）。线程本身不再关心"框是谁给的"，
    只负责"读帧 → 交出帧让检测器看 → 收结果"。这样：
      · 换模型不改这里
      · 测试可以塞一个脚本化的假检测器（`tools/vision_e2e.py` 就是这么干的）

    ⚠️ **检测和运动分开跑**：运动检测每帧都做（几毫秒，而且它必须每帧做才准），
    物体检测每 `detect_every` 帧才做一次（YOLOv8n 在 640×480 上约 60ms）。
    全跑的话视觉线程会被压到 15fps，而且没有任何收益 ——
    "桌上有几个杯子"这件事 0.2 秒看一次和 0.03 秒看一次，人感觉不出区别；
    但"画面在不在动"必须每帧看，否则抖一下就漏了。

    【阶段 19】线程里多挂了一个 `PresenceTracker`：检测结果喂给它，
    它把**抖动的逐帧框**滤成**稳定的在场计数**，放进快照的 `obj_counts`。
    为什么把滤波放在这个线程里、而不是主循环里：因为它的门槛单位是
    「检测次数」，而"检测了几次"只有这里知道（主循环看到的是帧）。
    放主循环就得把 `detect_every` 的除法抄过去一份，迟早对不上。
    """

    def __init__(self, source, detector, fps, motion_th, detect_every=3,
                 presence=None):
        super().__init__(daemon=True, name="vision")
        self.source = source
        self.detector = detector
        self.fps = fps
        self.motion_th = motion_th
        self.detect_every = max(1, int(detect_every))
        self.presence = presence
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self._snap = {"faces": 0, "boxes": [], "objects": [], "labels": "",
                      "obj_counts": {}, "objects_ready": False,
                      "motion": 0.0, "frame": None, "ok": True, "t": 0.0}
        self.n_frames = 0
        self.n_error = 0
        self.n_detect = 0

    def run(self):
        prev_gray = None
        objs = []                    # 上一次检测的结果，跨帧保留
        while not self.stop.is_set():
            t0 = time.time()
            try:
                ok, frame = self.source.read()
            except Exception:
                ok, frame = False, None
            if not ok or frame is None:
                self.n_error += 1
                if self.n_error > 20:                 # 连续读不到 → 放弃视觉这一路
                    with self.lock:
                        self._snap["ok"] = False
                    break
                time.sleep(0.2)
                continue
            self.n_error = 0
            self.n_frames += 1

            # ---------- 灰度 + 降噪（运动在灰度图上做）----------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            # ---------- 运动：和上一帧做差，统计"变了多少像素" ----------
            if prev_gray is None:
                motion = 0.0
            else:
                diff = cv2.absdiff(prev_gray, gray)
                _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                motion = float(np.count_nonzero(th)) / float(th.size)
            prev_gray = gray

            # ---------- 物体检测：每 detect_every 帧一次 ----------
            if self.n_frames % self.detect_every == 1 or self.detect_every == 1:
                try:
                    objs = self.detector.detect(frame) or []
                except Exception:
                    # 检测器出问题不能把整条视觉线程带走 ——
                    # 运动检测还活着，MOSS 至少还知道"有东西在动"。
                    objs = []
                self.n_detect += 1
                # 【阶段 19】喂给滤波层。⚠️ 必须放在**检测那一次**里，
                # 不能挪到 if 外面 —— 它内部的门槛单位是"检测次数"，
                # 每帧喂一次等于把"连续 3 次检出"偷换成"连续 1 次"。
                # 另外 `objs` 是跨帧保留的旧值，不在检测帧上它就是**陈的**，
                # 拿陈数据喂等于把同一个数字数两遍。
                if self.presence is not None:
                    try:
                        self.presence.update(objs)
                    except Exception:
                        pass

            # ---------- "有人"这件事仍然走老朋友 faces ----------
            # fuse() 的优先级表是按"有没有人"写的（阶段 9 定的），
            # 所以这里把"人"从物体列表里挑出来，接口保持不变。
            # ⚠️ 别的物体**不进** faces —— 一个杯子不该让 MOSS 切到"有人在"。
            persons = [o for o in objs if o.get("name") in PERSON_LIKE_NAMES]
            boxes = [tuple(o["box"]) for o in persons]

            with self.lock:
                self._snap = {"faces": len(persons), "boxes": boxes,
                              "objects": objs,
                              "labels": labels_line(objs) if HAS_VISION else "",
                              # 【阶段 19】稳定计数（已滤掉"人不报/家具不报"）。
                              # 注意快照里放的是**副本** —— 主循环拿去算 diff 时
                              # 不能改到跟踪器的内部状态。
                              "obj_counts": (self.presence.visible()
                                             if self.presence is not None else {}),
                              "objects_ready": (self.presence.ready
                                                if self.presence is not None else False),
                              "motion": motion, "frame": frame, "ok": True,
                              "t": time.time()}

            if self.fps > 0:                          # 限速，别把 CPU 跑满
                dt = time.time() - t0
                if dt < 1.0 / self.fps:
                    time.sleep(1.0 / self.fps - dt)

    def snapshot(self):
        with self.lock:
            s = dict(self._snap)
        return s

    def shutdown(self):
        self.stop.set()
        try:
            self.source.release()
        except Exception:
            pass


# ================================================================
#  4. 听觉一路（声音来源 + 分析 + 工作线程）
# ================================================================
def probe_inputs(seconds=0.4):
    """逐个试听所有输入设备，按实际音量从大到小返回。

    Windows 上一个麦克风常常同时挂在 MME / DirectSound / WASAPI / WDM-KS
    四套接口下，而其中有些接口会给出「数字静音」（一直 -96 dBFS，读到的全是零）。
    这时你会以为"麦克风坏了"，其实只是选错了接口。
    """
    results = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] <= 0:
            continue
        rate = int(d["default_samplerate"])
        ch = min(2, d["max_input_channels"])
        try:
            rec = sd.rec(int(seconds * rate), samplerate=rate, channels=ch,
                         device=i, dtype="float32", blocking=True)
            mono = rec.mean(axis=1)
        except Exception:
            continue
        if mono.size == 0:
            continue
        rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
        peak = float(np.max(np.abs(mono)))
        results.append({
            "index": i, "name": d["name"],
            "host": sd.query_hostapis(d["hostapi"])["name"], "rate": rate,
            "rms": 20.0 * math.log10(rms + 1e-9),
            "peak": 20.0 * math.log10(peak + 1e-9),
        })
    results.sort(key=lambda r: r["rms"], reverse=True)
    return results


def sane(entry):
    """读数是否可信。数字静音约 -96 dBFS，无数据约 -180，偶尔驱动 bug 报正几百 dB。"""
    return -150.0 < entry["rms"] < 5.0


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


class MicSource:
    """真实麦克风。用回调把数据丢进队列，工作线程慢慢算 —— 这样不会丢样本。"""

    def __init__(self, device, samplerate, blocksize):
        self.q = queue.Queue(maxsize=64)
        self.samplerate = samplerate
        self.last_time = None       # 最近一块的**采集时刻**（AEC 对齐用）
        self.stream = sd.InputStream(
            device=device, channels=1, samplerate=samplerate,
            blocksize=blocksize, dtype="float32", callback=self._callback,
        )
        self.stream.start()

    def _callback(self, indata, frames, time_info, status):
        try:
            # 时刻和音频**一起**入队。分开存的话，队列一积压，
            # 读到的那块音频配的却是"最新一块的时刻"，AEC 会拿错位的声音去减。
            self.q.put_nowait((indata[:, 0].copy(), time.time()))
        except queue.Full:
            pass                                        # 算不过来就丢最老的，不让延迟堆积

    def read(self):
        try:
            block, t = self.q.get(timeout=2.0)
            self.last_time = t
            return block
        except queue.Empty:
            self.last_time = time.time()
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
    """没有麦克风时的替身：一段「安静 → 拍手 → 说话」的循环（3.2 秒一轮）。"""

    def __init__(self, samplerate, blocksize):
        self.sr, self.n = samplerate, blocksize
        self.i = 0
        self.last_time = None
        self.cycle_blocks = max(6, int(3.2 * samplerate / blocksize))
        self.quiet_blocks = self.cycle_blocks * 19 // 50
        self.talk_blocks = self.quiet_blocks + 1

    def read(self):
        self.last_time = time.time()
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

        if n_ch > 1:
            data = data.reshape(-1, n_ch).mean(axis=1)

        self.data = data
        self.n = blocksize
        self.pos = 0
        self.last_time = None

    def read(self):
        self.last_time = time.time()
        if len(self.data) == 0:
            return None
        end = self.pos + self.n
        if end <= len(self.data):
            blk = self.data[self.pos:end]
            self.pos = end
        else:
            blk = np.concatenate([self.data[self.pos:], self.data[: end - len(self.data)]])
            self.pos = end - len(self.data)
        return blk.astype(np.float32)

    def release(self):
        pass


def make_audio_source(args):
    if args.audio_source == "demo":
        print("[OK] 听觉：合成音频（demo）", file=sys.stderr)
        return DemoSource(args.samplerate, args.blocksize)
    if args.audio_source.lower().endswith(".wav"):
        if not os.path.exists(args.audio_source):
            raise FileNotFoundError(args.audio_source)
        src = WavSource(args.audio_source, args.blocksize)
        print(f"[OK] 听觉：录音文件 {args.audio_source}（{src.sr}Hz）", file=sys.stderr)
        return src
    return open_mic(args.device, args.samplerate, args.blocksize)


class NoiseFloor:
    """自适应噪声底：取「最近几秒音量」的 10% 分位数。

    不能用平均值或指数平滑 —— 那样人一说话、或啪一下拍手，噪声底就被抬上去，
    接下来反而听不见安静的差别。取低分位数，值大概率落在"没人说话的瞬间"。
    trusted=False 的块（比如判定为拍手的瞬态）不进窗口。
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
    return 20.0 * math.log10(rms + 1e-9), 20.0 * math.log10(peak + 1e-9)


class AudioWorker(threading.Thread):
    """听觉线程：不断读声音块、判断有没有人在说话/拍手，把最新结果放进快照。

    阶段 12 起还多干一件事：把音频喂给**分段器**，切出"一整句"交给识别。
    segmenter / echo_gate / on_utterance 三个参数都是可选的 ——
    不给就退化成阶段 9 的行为（只听音量，不听内容）。
    """

    def __init__(self, source, args, block_seconds, segmenter=None,
                 echo_gate=None, on_utterance=None, aec=None, reference=None):
        super().__init__(daemon=True, name="audio")
        self.source = source
        self.args = args
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.floor = NoiseFloor(window_seconds=4.0, block_seconds=block_seconds)
        self._snap = {"dbfs": -99.0, "floor": -99.0, "snr": 0.0,
                      "clap": False, "listening": False, "text": "quiet",
                      "clap_seq": 0, "ok": True, "t": 0.0}
        self.prev_peak_db = -99.0
        self.n_blocks = 0
        self.n_error = 0
        self._is_loop_source = not isinstance(source, MicSource)
        self.segmenter = segmenter
        self.echo_gate = echo_gate
        self.on_utterance = on_utterance
        self.n_utterances = 0
        # 【阶段 17】回声消除。放在"读进来"和"判音量"之间 ——
        # 顺序很关键：先减掉自己的声音，再看电平/拍手/切句，
        # 否则融合逻辑仍然会把自己说的话当成"有人在讲话"。
        self.aec = aec
        self.reference = reference
        self.n_aec = 0
        self.aec_info = None

    def run(self):
        while not self.stop.is_set():
            t0 = time.time()
            block = self.source.read()
            if block is None:
                self.n_error += 1
                if self.n_error > 20:
                    with self.lock:
                        self._snap["ok"] = False
                    break
                time.sleep(0.2)
                continue
            self.n_error = 0
            self.n_blocks += 1

            # ---------- 阶段 17：先把"自己的声音"从这一块里减掉 ----------
            if self.aec is not None and self.reference is not None:
                # 时间戳用"这块音频**被采集到**的时刻"，不是"现在"。
                # MicSource 的回调会把时刻和音频一起入队 —— 队列积压时
                # 这两者能差好几秒，而 AEC 对时间对齐是最敏感的（差 1ms 都减不干净）。
                t_block = getattr(self.source, "last_time", None) or time.time()
                block, info = self.aec.process(block, t_end=t_block, ref=self.reference)
                self.aec_info = info
                self.n_aec += 1

            level, peak_db = block_level(block)

            # ---------- 拍手：峰值很高 + 前一块很安静（短促爆音的特征）----------
            is_clap = False
            if not self.args.no_clap:
                is_clap = (peak_db >= self.args.clap_th) and \
                          (self.prev_peak_db < self.args.clap_th - 12.0)
            self.prev_peak_db = peak_db

            # ---------- 噪声底：拍手这种瞬态不进窗口，免得把本底抬上去 ----------
            floor = self.floor.update(level, trusted=not is_clap)
            snr = level - floor

            # ---------- 判定"有没有声音" ----------
            hearing = (level >= self.args.db_th) and (snr >= self.args.snr_th)

            # ---------- 阶段 17：AEC 的"这段只有回声"判决 ----------
            # 为什么不能只看电平：滤波器刚起步那几百毫秒还不准，
            # 残留回声会高过 db_th —— 于是分段器在"纯回声"上切出一句，
            # MOSS 又把自己说的话当成用户提问，自问自答。
            # 实测（tools/fullduplex_e2e.py 的 S4）就是这么翻的车。
            #
            # AEC 手里有个比电平可靠得多的判据：它拿麦克风和**回声估计**比
            # （比值，跟房间耦合无关）。它说"远端在响、但麦克风没超出去"
            # 就是"这块里没有人的声音"。
            #
            # ⚠️ 注意这一条同时罩住了两种情况，都是有意的：
            #   1. 滤波器已收敛、麦克风确实没超出回声 → 真的没人
            #   2. 滤波器还在预热（warm），残差大得不可信 → **当成没人**
            #      这是保守假设：预热只有几百毫秒，代价是"这一段如果正好有人
            #      开口，开头会漏掉"；收益是不会拿自己的残差当用户。
            #      拿不准的时候，宁可不说话，也不能自问自答。
            #
            # ⚠️ 这和旧的半双工**不是一回事**：旧的是干脆不给分段器喂音频
            # （reset，前置缓冲被清空）；这里只是把这一块标成"没人说话"，
            # 音频照喂、前置缓冲照转 —— 所以用户真开口时，
            # 句子开头那 300ms 一个样本都不会丢。
            if self.aec_info is not None and self.aec_info.get("far_active") \
                    and not self.aec_info.get("dtd"):
                hearing = False

            if is_clap:
                text = "clap!"
            elif hearing:
                text = "hearing"
            else:
                text = "quiet"

            # ---------- 阶段 12：切句，交给识别 ----------
            # 直接复用上面算好的 hearing 当 VAD 结论，避免"两个噪声底各判各的"
            # 导致两边打架。分段器只在没有 hearing 参数时才自己算。
            if self.segmenter is not None:
                if self.echo_gate is not None and self.echo_gate():
                    # MOSS 正在说话：麦克风里全是它自己的声音。
                    # 不拦住的话，它会把自个儿说的话录下来、识别、再当"用户提问"回给自己
                    # —— 无限自我对话。半双工就是这个意思（见 asr.py 开头）。
                    self.segmenter.reset()
                else:
                    for utt in self.segmenter.feed(block, voiced=hearing):
                        self.n_utterances += 1
                        if self.on_utterance is not None:
                            self.on_utterance(utt)

            with self.lock:
                prev_seq = self._snap["clap_seq"]
                self._snap.update({
                    "dbfs": level, "floor": floor, "snr": snr,
                    "clap": is_clap, "listening": hearing or is_clap,
                    "text": text,
                    "clap_seq": prev_seq + (1 if is_clap else 0),
                    "ok": True, "t": time.time(),
                })

            # ---------- 限速：demo / wav 按真实时间播放，不空转 ----------
            if self._is_loop_source:
                bs = self.args.blocksize / float(getattr(self.source, "sr", self.args.samplerate))
                left = bs - (time.time() - t0)
                if left > 0:
                    time.sleep(left)

    def snapshot(self):
        with self.lock:
            return dict(self._snap)

    def shutdown(self):
        self.stop.set()
        try:
            self.source.release()
        except Exception:
            pass


def echo_blind(speak_worker, aec, refbuf):
    """现在是不是「MOSS 在说话，而我不知道它在放什么」。

    只有这种时候才需要退回半双工（把它自己的声音按静音算）。
    阶段 17 之前这里恒等于 `speak_worker.busy()` —— 于是它一开口就聋。

    现在多问一句："我手上有它正在放的波形吗？"
      有  → 回声会由 AEC 减掉，麦克风**保持打开** → 真·全双工
      没有 → 退回老办法，说话期间屏蔽听觉

    为什么不是简单地看 "--aec 开没开"：开了也可能拿不到参考
    （SAPI 合成到文件失败、mp3 解码失败、临时文件写不进去）。
    那种情况下如果还傻开着麦克风，MOSS 就会听见自己 → 自己回答自己 →
    无限自我对话 —— 正是阶段 11/12 花大力气堵掉的那个坑。
    所以判据落在**"此刻缓冲里到底有没有东西"**上，而不是配置。

    ⚠️ 它是模块级函数（不是 main 里的闭包）是有意的：
    分段器那道闸（AudioWorker.echo_gate）和融合层那道闸
    （主循环里的 `gated`）必须用**同一个判据**，而 tools/fullduplex_e2e.py
    这个探针也要拿它来验。三处共用一份实现，才不会各说各话。
    """
    if speak_worker is None or not speak_worker.busy():
        return False
    if aec is not None and refbuf is not None and refbuf.active:
        return False                      # 有参考 → 交给 AEC，全双工
    return True                           # 没参考 → 半双工兜底


# ================================================================
#  5. 融合：把「看见的」和「听见的」拼成一个判断
# ================================================================
def _object_tag(vision, max_n=2):
    """【阶段 18】把"除了人以外还看见什么"压成一段很短的标签，给 OLED 第三行用。

    为什么只带**非人**的东西：人在 mode 和 `face xN` 里已经说过了，
    再带一个 "person" 是废话，而 OLED 第三行一共只有 23 个字符
    （固件的 `gLine1[24]`）。

    为什么只塞进 VISION / MOTION 两个分支：`text` 在这个程序里是
    "**这一路**为什么切到这个状态"的占位标签。视觉切的状态才该带物体的名字；
    LISTEN 是听出来的、IDLE 是没东西，往里塞物体名只会让人看不懂。
    """
    objs = vision.get("objects") or []
    if not objs:
        return ""
    rest = [o for o in objs if o.get("name") not in PERSON_LIKE_NAMES]
    if not rest:
        return ""
    rows = count_objs(rest)[:max_n]
    return ", ".join(n if c == 1 else "%s x%d" % (n, c) for n, c in rows)


def count_objs(objs):
    """按类名归并计数，返回 `[(name, n), ...]`，多的在前（英文类名）。

    这里**刻意**委托给 `vision.count_by_name`，只把三元组削成二元组 ——
    因为日志里的 `objs` 字段（走 `labels_line`）和 OLED 上的短标签（走这里）
    是**同一批物体**的两个说法，排序规则一旦各写一份就会漂。

    第一版就是各写一份：`count_by_name` 同数量时按**中文名**排，
    这里按**英文名**排。平时看不出差别（`cup`/`laptop` 两种排法恰好同序），
    直到 `keyboard`(键盘) 和 `laptop`(笔记本电脑) 撞上 ——
    日志写 `laptop` 在前、OLED 写 `keyboard` 在前，同一条记录自相矛盾。
    这种 bug 不报错、不影响功能，只是让人对着两份输出反复怀疑自己看错了。
    """
    if HAS_VISION:
        return [(n, c) for n, _zh, c in vision_count_by_name(objs)]
    # vision.py 没进来时的兜底：不排序也要能数出个数。
    tally = {}
    for o in objs:
        n = o.get("name") or "?"
        tally[n] = tally.get(n, 0) + 1
    return sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))


def fuse(vision, audio, motion_th):
    """融合规则（从上往下，第一条命中的胜出）。

    | 条件                    | 状态    | OLED 文字      | 含义           |
    |-------------------------|---------|----------------|----------------|
    | 看见脸 + 听见声音        | SPEAK   | face+voice     | 有人跟我讲话   |
    | 只看见脸                | VISION  | face xN        | 有人在，盯着看 |
    | 只听见声音              | LISTEN  | hearing        | 有动静，在听   |
    | 只看见画面在动          | MOTION  | moving         | 有东西在动     |
    | 什么都没有              | IDLE    | idle           | 待机           |

    为什么"脸 + 声"要优先于"只有脸"：因为"有人且正在说话"是最强的交互信号，
    这时设备应该表现为"在回答/在说话"（SPEAK），而不是傻看着（VISION）。

    【阶段 18】VISION / MOTION 的文字后面会挂上"还看见了什么"（`+ cup`）。
    这是**纯粹的信息增益**，不动优先级表 ——
    "有人在"和"有人在，桌上还有个杯子"是两个不同的认知，后者更有用。
    ⚠️ 优先级表本身一个字都没改：老的行为、老的测试全部保持原样。
    """
    n_faces = vision["faces"]
    hearing = audio["listening"]
    motion = vision["motion"]
    tag = _object_tag(vision)

    if n_faces > 0 and hearing:
        return "SPEAK", "face+voice"
    if n_faces > 0:
        return "VISION", (f"face x{n_faces} + {tag}" if tag else f"face x{n_faces}")
    if hearing:
        return "LISTEN", audio["text"]
    if motion > motion_th:
        return "MOTION", (f"moving + {tag}" if tag else "moving")
    return "IDLE", "idle"


def _worth_narrating(mode, text):
    """这一刻值不值得让 MOSS 主动开口？

    为什么需要这个判断（阶段 12 踩出来的）：
      开了 TTS 之后，MOSS 每次状态变化都要说一句，一句 4 秒；而状态每 2~3 秒
      就变一次 —— 结果它**几乎一直在说话**。而半双工下"它说话时麦克风是屏蔽的"，
      于是它永远听不见你。实测：`听懂 0 句 · 切段 0`。

      这不是 bug，是必然的物理结果：**一个嘴不停的东西，耳朵一定是聋的。**
      所以有 ASR 时它必须"少说多听"，只在这几种真值得开口的时候说。
    """
    return mode in ("VISION", "SPEAK", "ERROR") or text == "clap!"


# 说明：过去这里有个 _dialog_text()，把对话历史压成纯文本。
# 阶段 13 之后这件事由 memory.MemoryStore.dialog_text() 统一负责 ——
# 因为历史现在存在磁盘上、还要跟长期事实一起管，不适合再在两边各写一份。


# ================================================================
#  6. 预览窗上叠加显示
# ================================================================
def draw_overlay(frame, vision, audio, mode, text, fps):
    """把融合结果画到画面上：物体框 + 状态 + 两路信号读数 + 音量条。

    【阶段 18】以前只画人脸框（一个绿框），因为那时也只认脸。
    现在每个框都带**名字**：人和别的东西用两种颜色区分 ——
    这样"框对了没有"一眼就能看出来，不用去翻日志。
    """
    objs = vision.get("objects") or []
    if objs:
        for o in objs:
            x, y, w, h = o["box"]
            is_person = o.get("name") in PERSON_LIKE_NAMES
            color = (0, 220, 170) if is_person else (60, 180, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = (zh_name(o["name"]) if HAS_VISION else o["name"])
            cap = "%s %.2f" % (label, o.get("conf", 0.0))
            # 标签画在框上方；框贴着画面顶时改画在框内，免得被裁掉
            ty = y - 6 if y > 18 else y + 18
            cv2.putText(frame, cap, (x, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
    else:
        # 没有检测器时退回老的"只画人脸框"（boxes 仍然有效）
        for (x, y, w, h) in vision["boxes"]:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 170), 2)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 96), (20, 14, 8), -1)
    cv2.putText(frame, f"MODE  {mode}", (14, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(frame, f"TEXT  {text}", (14, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    cv2.putText(frame,
                f"face {vision['faces']}  obj {len(objs)}  motion {vision['motion'] * 100:4.1f}%  "
                f"dBFS {audio['dbfs']:6.1f}   floor {audio['floor']:6.1f}   {fps:4.1f}fps",
                (14, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 210, 255), 1)

    # 底部画一条音量条（把 dBFS -70..0 映射到宽度）
    W = frame.shape[1]
    y0 = frame.shape[0] - 26
    lvl = max(0.0, min(1.0, (audio["dbfs"] + 70.0) / 70.0))
    cv2.rectangle(frame, (14, y0), (W - 14, y0 + 12), (60, 60, 60), 1)
    cv2.rectangle(frame, (14, y0), (14 + int((W - 28) * lvl), y0 + 12),
                  (0, 220, 170) if audio["listening"] else (120, 120, 120), -1)
    return frame


# ================================================================
#  7. 主流程
# ================================================================
def print_banner(args, mind_desc="关闭", tts_desc="关闭", asr_desc="关闭",
                 mem_desc="关闭", tool_desc="关闭", act_desc="关闭",
                 wake_desc="关闭", beh_desc="关闭", aec_desc="关闭",
                 vision_desc=None, objects_desc=None):
    print("=" * 74)
    print("  MOSS 多模态大脑   ·   眼睛（摄像头） + 耳朵（麦克风） → 一个判断")
    print("=" * 74)
    print(f"  画面：{args.vision_source}   {args.width}x{args.height}   "
          f"运动阈值 {args.motion_th}")
    # 【阶段 18】检测器那一行单独打 —— "认不认得出东西"是这一阶段的主角，
    # 混在"画面来源"里没人看得见它到底起没起来。
    print(f"  识别：{vision_desc if vision_desc is not None else '关闭'}")
    # 【阶段 19】变化感知单独一行：它的门槛（连续几次检出/漏检）只有这里有，
    # 出了问题要能一眼看到自己实际用的是哪一组数，而不是去翻代码里的默认值。
    print(f"  变化：{objects_desc if objects_desc is not None else '关闭'}")
    print(f"  听觉：{args.audio_source}   采样率 {args.samplerate}Hz   "
          f"块 {args.blocksize}   门限 {args.db_th}dBFS / 信噪比 {args.snr_th}dB")
    print(f"  融合：{args.rate:.0f}Hz   防抖动 {args.stable} 次   "
          f"串口 {args.port or '（不发）'}")
    print(f"  听懂：{asr_desc}")
    print(f"  语言：{mind_desc}")
    print(f"  语音：{tts_desc}")
    print(f"  记忆：{mem_desc}")
    print(f"  工具：{tool_desc}")
    print(f"  手脚：{act_desc}")          # 【阶段 14】执行器（灯环/电机/蜂鸣/屏幕）
    print(f"  唤醒：{wake_desc}")         # 【阶段 15】叫名字才答应
    print(f"  自主：{beh_desc}")          # 【阶段 16】它自己也会动
    print(f"  全双工：{aec_desc}")        # 【阶段 17】它说话时你也能插嘴
    print("=" * 74)


def main():
    args = parse_args()

    # ---------- 【阶段 18】只列模型 / 只下模型：干完就退出 ----------
    if args.vision_list or args.vision_download:
        if not HAS_VISION:
            print("[错误] 找不到 vision.py。", file=sys.stderr)
            return 2
        if args.vision_list:
            print(vision_list_models(args.vision_dir))
            return 0
        try:
            p = vision_download_model("yolov8n", args.vision_dir, args.vision_mirror)
        except Exception as e:
            print(f"[失败] {e}", file=sys.stderr)
            return 1
        print(f"[OK] 模型就位：{p}")
        print("     现在可以：python moss_brain.py --vision-model auto")
        return 0


    # --demo 是总开关：两路都切成合成
    if args.demo:
        args.vision_source = "demo"
        args.audio_source = "demo"

    # 想出声就得有台词。没配台词后端时，自动挂上离线模板 ——
    # 这样单开 `--tts edge` 也能听到 MOSS 说话，而不是一片沉默让人以为坏了。
    if args.tts != "off" and args.llm == "off":
        args.llm = "offline"
        print("[提示] 开了语音但没选台词后端，自动使用离线模板台词（--llm offline）。",
              file=sys.stderr)

    # --asr-test：只想验证"听不听得懂"，把回答和出声都关掉，
    # 免得一边测试一边被 MOSS 的回答干扰（它说话时麦克风是屏蔽的）。
    if args.asr_test:
        args.llm = "off"
        args.tts = "off"
        print("[提示] --asr-test：只测识别，语言层和语音层都关掉；"
              "识别到一句话就退出。", file=sys.stderr)

    # ---------- 有 ASR 就"少说多听" ----------
    # 原因见 _worth_narrating()：嘴不停的东西，耳朵一定是聋的。
    if args.narrate == "auto":
        narrate_mode = "quiet" if args.asr != "off" else "all"
        if narrate_mode == "quiet":
            print("[提示] 识别开着 → MOSS 改成「少说多听」：只在有人来/有人说话/"
                  "拍手/出错时才主动开口（--narrate all 可改回絮叨模式）。",
                  file=sys.stderr)
    else:
        narrate_mode = args.narrate

    # ---------- 串口（阶段 14）----------
    # 为什么提到这里来：执行器工具（set_led/turn_motor…）要有设备才能挂载，
    # 而工具清单必须在**构造 ToolBox 之前**就定好。
    # 以前串口是最后才开的，那时工具箱已经建完了，执行器就挤不进清单。
    ser = open_serial(args.port, args.baud) if args.port else None

    # ---------- 执行器连接（阶段 14）：MOSS 的"手脚" ----------
    # 没串口 = 没手脚，但脑子照常转（跟"没麦克风也能跑"是一个道理）。
    link = None
    act_desc = "关闭"
    if not HAS_ACTUATORS:
        act_desc = "关闭（找不到 moss_actuators.py）"
    elif args.no_actuators:
        act_desc = "关闭（--no-actuators）"
    elif ser is None:
        act_desc = "关闭（没串口；加 --port 或 --port socket://localhost:4000）"
    else:
        try:
            link = ActuatorLink(ser, io_lock=_IO_LOCK, name=args.port,
                                log=lambda s: print(s, file=sys.stderr))
        except Exception as e:
            act_desc = "关闭（执行器没起来）"
            print(f"[提示] 执行器没起来：{e}", file=sys.stderr)

    # ---------- 长期记忆（阶段 13）：先准备好，语言层和工具都要用 ----------
    # 为什么放在语言层**之前**：工具（remember/recall）要往里写，
    # 而且提示词里要带上"你知道的关于用户的事"。
    memory = None
    mem_desc = "关闭"
    if args.no_memory:
        mem_desc = "关闭（--no-memory）"
    elif not HAS_MEMORY:
        mem_desc = "关闭（找不到 memory.py）"
        print("[提示] 找不到 memory.py，长期记忆被关掉。", file=sys.stderr)
    else:
        try:
            memory = MemoryStore(args.memory_file)
            mem_desc = memory.describe()
        except Exception as e:
            mem_desc = "关闭（记忆层没起来）"
            print(f"[提示] 记忆层没起来：{e}", file=sys.stderr)

    # ---------- 工具层（阶段 13）：模型可以"申请调用"的工具箱 ----------
    # 它只是一张"申请表"：模型说要调什么，我们执行、记账、把结果交回去。
    toolbox = None
    tool_desc = "关闭"
    if args.no_tools:
        tool_desc = "关闭（--no-tools）"
    elif not HAS_TOOLS:
        tool_desc = "关闭（找不到 moss_tools.py）"
    elif args.llm == "off":
        tool_desc = "关闭（没开语言层，没人会去调它）"
    else:
        try:
            # 【阶段 14】先装执行器工具，再建 ToolBox —— 顺序不能反：
            # ToolBox 是在构造那一刻快照 REGISTRY 的，装晚了清单里就没有。
            act_names = install_actuators(link, log=lambda s: print(s, file=sys.stderr))
            if act_names:
                act_desc = "%d 个：%s" % (len(act_names), " ".join(act_names))
            elif link is None:
                pass                      # 上面已经写好原因了
            else:
                act_desc = "关闭（这一轮没装）"

            toolbox = ToolBox(
                context={"memory": memory, "mode": "BOOT", "device": link},
                log=lambda s: print(s, file=sys.stderr))
            tool_desc = "%d 个：%s" % (len(toolbox.names()), " ".join(toolbox.names()))
        except Exception as e:
            tool_desc = "关闭（工具层没起来）"
            print(f"[提示] 工具层没起来：{e}", file=sys.stderr)

    # ---------- 语言层：先把"嘴"准备好，banner 里要显示它 ----------
    mind_worker = None
    backend = None
    mind_desc = "关闭"
    if args.llm != "off":
        if not HAS_MIND:
            mind_desc = "关闭（找不到 mind.py）"
            print("[提示] 找不到 mind.py，--llm 被忽略。", file=sys.stderr)
        else:
            key = (args.api_key or os.environ.get("MOSS_API_KEY")
                   or os.environ.get("OPENAI_API_KEY") or "")
            try:
                backend, mind_desc = build_mind(
                    args.llm, args.api_base, key, args.model, args.llm_timeout,
                    args.persona or DEFAULT_PERSONA,
                    tool_catalog=(toolbox.catalog() if toolbox else ""),
                    max_rounds=args.tool_max_rounds)
                mind_worker = MindWorker(backend, toolbox=toolbox,
                                         max_rounds=args.tool_max_rounds)
                mind_worker.start()
            except Exception as e:
                mind_desc = "关闭（语言层没起来）"
                print(f"[提示] 语言层没起来：{e}", file=sys.stderr)

    # ---------- 语音层：把"嗓子"准备好（阶段 11）----------
    speak_worker = None
    voice = None
    tts_desc = "关闭"
    if args.tts != "off":
        if not HAS_VOICE:
            tts_desc = "关闭（找不到 voice.py）"
            print("[提示] 找不到 voice.py，--tts 被忽略。", file=sys.stderr)
        else:
            try:
                voice, tts_desc = build_voice(
                    args.tts, args.tts_voice, args.tts_rate, args.tts_pitch,
                    sapi_rate=args.tts_sapi_rate,
                    on_fallback=lambda e: print(
                        f"[提示] edge 不可用（{e}），嗓子已切成 Windows SAPI。",
                        file=sys.stderr),
                    on_note=lambda s: print(f"[提示] {s}", file=sys.stderr))
            except Exception as e:
                tts_desc = "关闭（语音层没起来）"
                print(f"[提示] 语音层没起来：{e}", file=sys.stderr)

    # ---------- 回声消除（阶段 17）：必须赶在 SpeakWorker 之前 ----------
    # 因为它要把"参考缓冲"装到说话线程上（每念一句登记一次波形）。
    refbuf = None
    aec = None
    aec_desc = "关闭"
    if args.aec == "off":
        aec_desc = "关闭（--aec off）"
    elif voice is None:
        aec_desc = "关闭（没开 --tts，没有回声可消）"
    elif args.demo and args.aec != "on":
        # --demo 的声音是**合成**的，麦克风根本没有"听见"扬声器 ——
        # 没有真实的声学回路，AEC 就是拿着一段毫不相干的参考去减另一段音频，
        # 减不动 → 反复触发发散复位（实测 392 块里复位 20 次）。
        # 自说自听在 demo 里本来就不存在，所以默认关掉，别刷一堆没意义的统计。
        # 想看它的行为可以 --aec on 强制开。
        aec_desc = "关闭（--demo 是合成音频，没有真实回声回路；--aec on 可强制开）"
    elif not HAS_AEC:
        aec_desc = "关闭（找不到 aec.py）"
        print("[提示] 找不到 aec.py，--aec 被忽略。", file=sys.stderr)
    elif not getattr(voice, "supports_reference", False) and args.aec != "on":
        aec_desc = "关闭（这个嗓子给不出参考信号，用 --aec on 可强制开）"
    else:
        try:
            refbuf = ReferenceBuffer(args.samplerate, seconds=4.0)
            aec = EchoCanceller(
                args.samplerate, taps_ms=args.aec_taps_ms,
                subblock=args.aec_subblock, mu=args.aec_mu,
                dtd_ratio=args.aec_dtd, resid_db=args.aec_resid,
                log=lambda s: print(s, file=sys.stderr))
            aec_desc = aec.describe()
            if not getattr(voice, "supports_reference", False):
                aec_desc += "  ⚠ 这个嗓子没有参考信号 → 只能半双工"
        except Exception as e:
            refbuf = aec = None
            aec_desc = "关闭（没起来：%s）" % e
            print(f"[提示] 回声消除没起来：{e}", file=sys.stderr)

    if voice is not None:
        try:
            speak_worker = SpeakWorker(
                voice, guard_ms=args.tts_guard, dedup_s=args.tts_dedup,
                on_error=lambda e: print(f"[提示] 说话失败：{e}", file=sys.stderr),
                reference=refbuf)
            speak_worker.start()
        except Exception as e:
            tts_desc = "关闭（说话线程没起来）"
            print(f"[提示] 说话线程没起来：{e}", file=sys.stderr)

    # ---------- 听觉理解层：把"声音"变成"文字"（阶段 12）----------
    asr_worker = None
    segmenter = None
    asr_desc = "关闭"
    if args.asr != "off":
        if not HAS_ASR:
            asr_desc = "关闭（找不到 asr.py）"
            print("[提示] 找不到 asr.py，--asr 被忽略。", file=sys.stderr)
        else:
            opts = {
                "model": args.asr_model,
                "dir": args.asr_dir,
                "mirror": args.asr_mirror or "https://hf-mirror.com",
                "device": args.asr_device,
                "compute_type": args.asr_compute_type,
                "lang": args.asr_lang,
                "beam": args.asr_beam,
                "api_base": args.asr_api_base,
                "api_key": args.asr_api_key or os.environ.get("MOSS_ASR_KEY") or "",
                "api_model": args.asr_api_model,
            }
            try:
                engine, asr_desc = build_engine(args.asr, opts)
                asr_worker = AsrWorker(
                    engine, args.samplerate, queue_size=args.asr_queue,
                    on_error=lambda e: print(f"[提示] 识别失败：{e}", file=sys.stderr),
                    on_empty=lambda d: print(
                        f"[提示] 切出一段只有 {d * 1000:.0f}ms 的音频，太短，"
                        f"跳过识别（分段器在静音边界上的小毛刺）。", file=sys.stderr))
                asr_worker.start()
                segmenter = UtteranceSegmenter(
                    samplerate=args.samplerate, blocksize=args.blocksize,
                    end_ms=args.asr_end_ms, min_speech_ms=args.asr_min_ms)
            except Exception as e:
                asr_desc = "关闭（识别层没起来）"
                print(f"[提示] 识别层没起来：{e}", file=sys.stderr)

    # ---------- 唤醒门（阶段 15）：叫名字才答应 ----------
    # 它工作在"识别出来的文字"上，所以只有 ASR 起来了才有意义。
    # --asr-test 是识别层的自检，专门用来量"识别准不准"——
    # 被唤醒门拦掉会让它永远等不到结果，所以强制绕过。
    wake_gate = None
    wake_desc = "关闭"
    if args.no_wake:
        wake_desc = "关闭（--no-wake）"
    elif args.asr == "off":
        wake_desc = "关闭（没开 --asr）"
    elif args.asr_test:
        wake_desc = "关闭（--asr-test 专测识别，绕过）"
    elif not HAS_WAKE:
        wake_desc = "关闭（找不到 wake.py）"
        print("[提示] 找不到 wake.py，唤醒门被忽略。", file=sys.stderr)
    else:
        words = None
        if args.wake_word:
            raw_words = args.wake_word.replace("，", ",").replace("、", ",")
            words = [w.strip() for w in raw_words.split(",") if w.strip()]
            if not words:
                words = None
        wake_gate = WakeGate(
            words=words, window=args.wake_window, strict=args.wake_strict,
            log=lambda s: print(s, file=sys.stderr))
        wake_desc = wake_gate.describe()

    # ---------- 主动行为层（阶段 16）：它自己也会动 ----------
    # 必须有真串口才有意义 —— 主动行为的产物就是"往设备发命令"，
    # 没设备时它只是个空转的状态机，不如干脆关掉。
    behavior = None
    beh_desc = "关闭"
    if args.no_proactive or args.autonomy == "off":
        beh_desc = "关闭（完全被动）"
    elif not HAS_BEHAVE:
        beh_desc = "关闭（找不到 behave.py）"
        print("[提示] 找不到 behave.py，主动行为被忽略。", file=sys.stderr)
    elif link is None or not link.usable:
        beh_desc = "关闭（没接设备）"
    else:
        behavior = BehaviorEngine(
            args.autonomy or "l0", gap=args.proactive_gap,
            log=lambda s: print(s, file=sys.stderr))
        beh_desc = behavior.describe()

    # ---------- 视觉层（阶段 18）：造一个检测器 ----------
    # 这一段放在 banner 之前，因为 banner 要打"识别"那一行。
    # ⚠️ make_detector() 设计成**不抛异常**：模型没下 / onnxruntime 没装 /
    # 文件坏了 —— 都只降级（退 Haar、再退 null），不能让整个 MOSS 起不来。
    # 视觉瞎了，听觉和对话还得能用。
    detector = None
    vision_desc = "关闭"
    if not HAS_VISION:
        vision_desc = "关闭（找不到 vision.py）"
        print("[提示] 找不到 vision.py，视觉退回「只做运动检测」。", file=sys.stderr)
    else:
        try:
            detector, note = make_detector(
                args.vision_model, args.vision_dir, args.vision_mirror,
                model_path=args.vision_model_path, imgsz=args.vision_imgsz,
                conf_th=args.vision_conf, iou_th=args.vision_iou,
                max_obj=args.vision_max_obj, threads=args.vision_threads)
            vision_desc = detector.describe()
            if note:
                vision_desc += "  ← " + note
                print(f"[提示] 检测器降级：{note}", file=sys.stderr)
        except Exception as e:
            # 兜底中的兜底：连 make_detector 都炸了，也必须让程序起来
            print(f"[提示] 检测器构造失败（{e}），视觉退回只做运动检测。",
                  file=sys.stderr)
            detector, vision_desc = None, "关闭（构造失败：%s）" % e

    # ---------- 变化感知层（阶段 19）：把抖动的框滤成稳定计数 ----------
    # ⚠️ 建它的条件刻意**不包含"有没有接设备"** —— 和 behavior 不一样。
    # 理由：这一层本身是纯滤波，它产出的 `obj_counts` 还会被日志和
    # 快照用上（排错时"它到底看见几个杯子"比什么都重要）。
    # 没有设备时它照样跑，只是没人去发那条 LED 命令而已。
    presence = None
    objs_desc = "关闭"
    if args.no_objects:
        objs_desc = "关闭（--no-objects）"
    elif not HAS_PRESENCE:
        objs_desc = "关闭（找不到 presence.py）"
        print("[提示] 找不到 presence.py，「东西变了」的感知被忽略。", file=sys.stderr)
    elif detector is None or getattr(detector, "backend", "none") == "none":
        # 没有检测器时跟踪器只会看到空列表，"跟踪"就变成了纯开销。
        # 与其假装它在工作，不如照实说关着。
        objs_desc = "关闭（没有物体检测器）"
    else:
        # 参数默认值放在 presence.py 里（那里有配套的注释说明为什么是这几个数），
        # 这里只在用户显式传了才覆盖 —— 免得同一个数字在两处各有一份"真相"。
        kw = {}
        if args.objects_enter is not None:
            kw["enter_hits"] = args.objects_enter
        if args.objects_leave is not None:
            kw["leave_hits"] = args.objects_leave
        if args.objects_count is not None:
            kw["count_hits"] = args.objects_count
        if args.objects_warmup is not None:
            kw["warmup"] = args.objects_warmup
        try:
            presence = PresenceTracker(**kw)
            objs_desc = ("在场跟踪 · 连续 %d 次检出算来 / %d 次漏检算走 / "
                         "个数确认 %d 次 / 暖机 %d 次"
                         % (presence.enter_hits, presence.leave_hits,
                            presence.count_hits, presence.warmup))
        except Exception as e:
            print(f"[提示] 变化感知层构造失败（{e}），已忽略。", file=sys.stderr)
            presence, objs_desc = None, "关闭（构造失败：%s）" % e

    print_banner(args, mind_desc, tts_desc, asr_desc, mem_desc, tool_desc,
                 act_desc, wake_desc, beh_desc, aec_desc, vision_desc,
                 objs_desc)

    # ---------- 只列音色 ----------
    if args.tts_list:
        try:
            from voice import EdgeVoice
            for m in EdgeVoice(voice=args.tts_voice or "zh-CN-YunxiNeural").list_voices(locale="zh"):
                print(f"  {m['ShortName']:<34} {m.get('Gender', '')}")
        except Exception as e:
            print(f"[失败] 列音色要联网，且需要 edge-tts：{e}", file=sys.stderr)
            return 1
        return 0

    # ---------- 只测嗓子：不碰摄像头/麦克风 ----------
    if args.tts_test:
        if speak_worker is None:
            print("[错误] 语音层没起来，无法自检（看上面的提示）。", file=sys.stderr)
            return 2
        line = "系统自检完成，各单元上线。我是五百五十W。"
        print(f"[说话] {line}")
        t0 = time.time()
        speak_worker.submit(line)
        time.sleep(0.2)
        while speak_worker.busy():
            time.sleep(0.05)
        speak_worker.shutdown()
        print(f"[成功] 用了 {time.time() - t0:.1f} 秒")
        return 0

    # ---------- 只测语言层：不碰摄像头/麦克风 ----------
    if args.llm_test:
        if backend is None:
            print("[错误] 语言层没起来，无法自检（看上面的提示）。", file=sys.stderr)
            return 2
        ev = {"mode": "SPEAK", "faces": 1, "motion": 0.012,
              "dbfs": -28.0, "clap": False, "prev_say": ""}
        t0 = time.time()
        try:
            say, tag = backend.respond(ev)
        except Exception as e:
            print(f"[失败] {e}", file=sys.stderr)
            return 1
        print(f"[成功] {(time.time() - t0) * 1000:.0f} ms")
        print(f"  say = {say}     ← 打到终端（将来可接 TTS 出声）")
        print(f"  tag = {tag}     ← 发给 OLED（{len(tag)} 个 ASCII 字符）")
        return 0

    # ---------- 自动挑麦克风 ----------
    if args.auto and args.audio_source == "mic" and args.device is None:
        rows = probe_inputs(args.probe_seconds)
        best = next((r for r in rows if sane(r)), None)
        if best is None:
            print("[错误] 没找到有信号的输入设备（--probe 看详情）。改用 --demo 或 --device。",
                  file=sys.stderr)
            return 2
        args.device = best["index"]
        print(f"[自动选设备] --device {best['index']}  {best['name']}"
              f"（{best['host']}，当前 {best['rms']:.1f} dBFS）", file=sys.stderr)

    if args.audio_source == "mic":
        try:
            args.device = resolve_device(args.device)
        except RuntimeError as e:
            print(f"[错误] {e}", file=sys.stderr)
            return 2

    # ---------- 打开两路来源 ----------
    try:
        vision_src = make_vision_source(args.vision_source, args.width, args.height)
    except Exception as e:
        print(f"[错误] 打不开画面来源 '{args.vision_source}'：{e}", file=sys.stderr)
        print("       没有摄像头的话加 --demo，或看 pc_vision/README。", file=sys.stderr)
        return 2

    try:
        audio_src = make_audio_source(args)
    except Exception as e:
        print(f"[错误] 打不开声音来源 '{args.audio_source}'：{e}", file=sys.stderr)
        print("       排查顺序：--probe / --auto / --demo（详见 pc_audio/README）。",
              file=sys.stderr)
        try:
            vision_src.release()
        except Exception:
            pass
        return 2

    # 串口在更早的地方就开了（阶段 14 起，执行器工具需要它先就位），这里不再重复打开。
    # 视觉检测器（阶段 18）在 banner 之前就造好了 —— 见上面那一段。

    # ---------- 阶段 17：全双工的开关就是这个函数 ----------
    # 真正的判据在模块级 echo_blind() 里（探针要复用），这里只是把
    # 三个运行时对象绑上去，形成一个无参的闭包给 AudioWorker 用。
    def _echo_blind():
        return echo_blind(speak_worker, aec, refbuf)

    # ---------- 起两个线程 ----------
    block_seconds = args.blocksize / float(
        getattr(audio_src, "samplerate", getattr(audio_src, "sr", args.samplerate)))
    vw = VisionWorker(vision_src, detector, fps=30.0, motion_th=args.motion_th,
                      detect_every=args.vision_detect_every, presence=presence)
    if asr_worker is not None:
        aw = AudioWorker(audio_src, args, block_seconds, segmenter=segmenter,
                         echo_gate=_echo_blind, on_utterance=asr_worker.submit,
                         aec=aec, reference=refbuf)
    else:
        aw = AudioWorker(audio_src, args, block_seconds, aec=aec, reference=refbuf)
    vw.start()
    aw.start()

    print("[OK] MOSS 大脑启动。按 q 或 Ctrl+C 退出。", file=sys.stderr)
    print("-" * 74, file=sys.stderr)

    loop_period = 1.0 / args.rate if args.rate > 0 else 0.0
    cur_mode = None
    cur_text = ""
    pending = None
    stable_n = 0
    seen_clap_seq = 0
    cur_ev_id = 0          # 当前状态对应的"语言层事件号"
    applied_ev_id = 0      # 已经应用过台词的事件号
    last_say = ""          # 上一句中文台词（作为上下文喂给模型，避免重复）
    applied_asr_id = 0     # 已经处理过的识别结果序号
    n_heard = 0            # 真正问出去几句话（阶段 15 起不含"只喊了名字"）
    n_woken = 0            # 被唤醒几次（含只喊名字）—— 阶段 15
    n_ignored = 0          # 没喊名字、被安静放过的话 —— 阶段 15
    last_ask_at = 0.0      # 上次把"用户提问"交给语言层的时间
    last_narrate_at = 0.0  # 上次"自言自语"的时间（quiet 模式限速用）
    last_json = 0.0
    last_fps_at = time.time()
    last_v_frames = 0
    fps_now = 0.0
    iters = 0
    t0 = time.time()
    window_ok = not args.no_window
    # 最近几轮对话（用户说了什么 / MOSS 回了什么），作为上下文喂给大模型。
    # 没有它，MOSS 每句都是"失忆"的；有了它才能接上下文（"那它呢？"）。
    #
    # 阶段 13 起这份历史**落在磁盘上**（MemoryStore）：关掉再开还接着上一回聊，
    # 而且长期事实（"用户的名字"这类）也归它管，会随每次请求一起喂给模型。
    # 注意两个数字不是一回事：
    #   --memory-turns / --asr-history  决定"喂给模型几轮"（省 token）
    #   MemoryStore.max_dialog          决定"磁盘上留几轮"（默认 60）
    mem_turns = args.memory_turns if args.memory_turns is not None else args.asr_history
    mem_turns = max(0, mem_turns)

    def _dialog_snapshot():
        """这次请求要带上的对话历史（没开记忆就是空）。"""
        return memory.dialog_text(mem_turns) if memory is not None else ""

    def _facts_snapshot():
        """长期事实：让模型知道"对面是谁"。"""
        return memory.facts_text() if memory is not None else ""

    try:
        while True:
            iter_t0 = time.time()
            iters += 1

            vision = vw.snapshot()
            audio = aw.snapshot()

            # ---------- 有一个线程挂了就退出，别空转 ----------
            if not vision["ok"] and not audio["ok"]:
                print("[提示] 视觉和听觉都读不到了，结束。", file=sys.stderr)
                break

            # ---------- 自说自听抑制（阶段 11 的关键）----------
            # 音箱放出的声音会被自己的麦克风听见 → 融合逻辑以为"有人在讲话"
            # → 进 SPEAK → 又说一句 → 又听见…… 正反馈，MOSS 会自己跟自己聊起来。
            # 所以说话期间把听觉按静音算。
            #
            # 注意两点：
            #   1. **视觉不屏蔽** —— 说话时你走过去，它照样盯着你
            #   2. 默认连拍手也一起屏蔽。麦克风分不清"你拍的"和"MOSS 自己开口
            #      那一下的爆音"，放着不管它就会自己打断自己。
            #      真想用拍手打断，加 --barge-in。
            speaking = speak_worker is not None and speak_worker.busy()
            # --demo 的音频是合成出来的，根本不存在"自说自听"，
            # 这时候屏蔽听觉反而会把演示搞乱（LISTEN/SPEAK 全看不到了）。
            #
            # 【阶段 17】判据换成 `_echo_blind()` —— 它恰好就是
            # 「它在说话 且 我手上没有它正在放的波形」。AEC 一旦拿到参考
            # （refbuf.active 为真）它立刻返回 False，融合层照常进 LISTEN，
            # 于是你插嘴它听得见 —— 这就是真·全双工。
            #
            # 为什么不直接写 `speaking and not refbuf.active`：那样就有了两份
            # 判据（分段器那份 echo_gate、融合这份），迟早会漂移 ——
            # 最典型的漂移是"分段器放行了、融合层还按静音算"，症状是屏幕上
            # 一直挂着 SPEAK，你喊它它也纹丝不动。共用同一个函数最省心。
            gated = _echo_blind() and not args.demo and not args.no_echo_guard
            audio_fused = audio
            if gated:
                audio_fused = dict(audio)
                # ⚠️ 关键：融合逻辑读的是 `listening` 这个**已经算好的布尔值**，
                # 不是 dbfs。一开始只改了 dbfs，测试才发现完全没起作用
                # （日志里照样出现 "speak": true 却判成 LISTEN 的行）。
                # 所以必须把 listening 也按下去。
                audio_fused["listening"] = False
                audio_fused["dbfs"] = -99.0      # 留个"被屏蔽过"的痕迹，好排查
                audio_fused["clap"] = False
                audio_fused["text"] = "quiet"
                if not args.barge_in:
                    audio_fused["clap_seq"] = seen_clap_seq    # 冻住 = 当作没听见

            mode, text = fuse(vision, audio_fused, args.motion_th)

            # ---------- 阶段 15：睡着的时候给个提示 ----------
            # 不提示的话用户根本不知道它需不需要喊名字 —— 屏幕上永远写着 idle。
            # 只在 IDLE 时替换，免得把 VISION/listening 这些更有信息量的字盖掉。
            if (wake_gate is not None and mode == "IDLE"
                    and not wake_gate.awake):
                text = "SAY " + wake_gate.first_word.upper()

            # ---------- 防抖动 ----------
            if mode == pending:
                stable_n += 1
            else:
                pending, stable_n = mode, 1

            # ---------- 拍手是"事件"，绕过防抖动立刻响应 ----------
            clap_now = audio_fused["clap_seq"] > seen_clap_seq
            if clap_now:
                seen_clap_seq = audio_fused["clap_seq"]
                # --barge-in：拍手打断。人主动打招呼，MOSS 该立刻闭嘴听
                if speak_worker is not None and args.barge_in:
                    speak_worker.interrupt()

            changed = False
            if clap_now and mode in ("LISTEN", "IDLE"):
                # 拍手时如果只有声音、没有脸 → 立刻给一个明确的"听到了"回应
                if cur_mode != "LISTEN" or cur_text != "clap!":
                    mode, text, changed = "LISTEN", "clap!", True
                    cur_mode, cur_text = mode, text
                pending, stable_n = mode, 0
            elif stable_n >= args.stable and (mode != cur_mode or text != cur_text):
                cur_mode, cur_text, changed = mode, text, True

            # ---------- 输出 ----------
            # 注意：状态还没稳定（cur_mode 还是 None）时不做"心跳输出"，
            # 否则会先吐一行 mode:null，误导人以为程序错了。
            now = time.time()

            # ---------- 阶段 13：把最新快照交给工具层 ----------
            # 工具（look / stats 这类）要看"此刻"的画面和状态，所以每轮刷新上下文。
            # 注意：真正的工具执行发生在 MindWorker 线程里，读的就是这份快照。
            if toolbox is not None:
                toolbox.update(
                    vision={"faces": int(vision["faces"]),
                            "motion": float(vision["motion"]),
                            # 【阶段 18】物体清单是 look 工具"能不能答上来"的关键：
                            # 没有它，look 只能说"有 1 个人"（阶段 9 的水平）。
                            "objects": list(vision.get("objects") or []),
                            "labels": vision.get("labels") or "",
                            # look 工具要靠它分清"没认出东西"和"没开识别" ——
                            # 混在一起会让模型以为"画面里什么都没有"，然后开始编。
                            "detector": (detector.backend if detector is not None
                                         else "none")},
                    audio={"dbfs": float(audio["dbfs"]),
                           "listening": bool(audio["listening"])},
                    mode=cur_mode, uptime=now - t0, heard=n_heard)

            # ---------- 用户这一轮还没结束吗？----------
            # 正在说 / 还在识别 / 刚问完不久 —— 这三种情况都别让状态变化去打岔。
            # 否则会出现：你刚问完"几点了"，MOSS 先蹦一句"我看见你了"，
            # 把你的问题顶掉，然后那句"我看见你了"还要占掉 4 秒音箱时间。
            user_active = (
                (segmenter is not None and segmenter.in_speech)
                or (asr_worker is not None and asr_worker.busy())
                or (now - last_ask_at) < args.asr_hold
            )

            # ---------- 阶段 16：它自己想不想动一下 ----------
            # 位置很讲究：必须在 user_active 算完之后 ——
            # "你在说话时它别出声"这条礼仪就靠这个标志（见 behave.py）。
            # 整条链路不走网络、不走大模型，所以在 20Hz 的循环里直接判。
            if behavior is not None:
                act = behavior.tick({
                    "faces": int(vision["faces"]),
                    "motion": float(vision["motion"]),
                    "clap": bool(audio_fused["clap"]),
                    "mode": cur_mode,
                    "speaking": speaking,
                    "user_active": user_active,
                    # 【阶段 19】交给行为层的是**稳定计数**，不是这一帧的检测结果。
                    # 这一件事就是这个阶段全部的意义：抖动的原始框会在行为层被
                    # 当成"东西变了"，然后它就会像坏掉的报警器一样反复念叨。
                    # 滤波在 presence.py 里（视觉线程内），这里只是转交。
                    "obj_counts": vision.get("obj_counts") or {},
                    "objs_ready": bool(vision.get("objects_ready")),
                })
                if act is not None and link is not None and link.usable:
                    for cmd in act.cmds:
                        link.send(cmd)
                    # 【阶段 19】这句话要不要说出口 —— 能走到这里说明权限已经够了
                    # （L0 时 behave 根本不会给 say，见那边的 blocked_say 统计）。
                    #
                    # 这里**直接说**，不再过一遍大模型。三个理由：
                    #   ① 这句话本身就是完整的，而且措辞是被刻意约束过的
                    #      （presence.describe 只说"什么、几个"，绝不猜方位和原因）。
                    #      过一遍模型反而可能给它加上"在你桌上"这类编出来的东西。
                    #   ② 它是**反射**，和"有人来了转个头"同类 —— 不该等一秒多的网络往返。
                    #   ③ 只有一条代码路径：探针离线测的那条，就是真机上跑的那条。
                    if act.say:
                        last_say = act.say
                        if act.tag:
                            send_raw(ser, "TEXT " + act.tag)
                        if speak_worker is not None:
                            speak_worker.submit(act.say)
                        # 记进对话历史：这样之后你问"你刚才说什么了"它能答上来。
                        # user 留空 —— 这确实不是你说的话，是它自己看到的。
                        if memory is not None:
                            memory.add_turn("", act.say)
                        print(f"[主动] 说出口：{act.say}   [OLED] {act.tag}",
                              file=sys.stderr)

            heartbeat_due = (cur_mode is not None) and (now - last_json) >= args.heartbeat
            if changed or args.verbose or heartbeat_due:
                sent = send_state(ser, cur_mode, cur_text) if changed else False
                # 【阶段 19】滤波后的稳定计数。和下面 `objs`（**这一帧**
                # 检测器原始输出）的区别正是这一阶段要验证的东西 ——
                # 排错时两个都对出来看，一眼就知道"闪"是在哪一层被吃掉的。
                # ⚠️ 用 presence.brief() 而不是在这儿自己拼字符串：
                # 阶段 18 吃过一次亏（同一个排序写两份，日志和 OLED 各说各话）。
                stable = presence.brief() if presence is not None else ""
                rec = {
                    "t": round(now - t0, 2),
                    "faces": int(vision["faces"]),
                    "motion": round(vision["motion"], 4),
                    # 【阶段 18】认出来的东西。空的时候不打这个字段，
                    # 免得每行日志都挂一个 ""（噪声比信息多）。
                    **({"objs": vision["labels"]} if vision.get("labels") else {}),
                    **({"stable": stable} if stable else {}),
                    "dbfs": round(audio["dbfs"], 1),
                    "floor": round(audio["floor"], 1),
                    "clap": bool(audio["clap"]),
                    "mode": cur_mode,
                    "text": cur_text,
                    "sent": bool(sent),
                    # 注意 dbfs 是**真实测量值**没被屏蔽过。
                    # 说话时你会在这里看到 dbfs 明显抬起来 —— 那就是"自说自听"的
                    # 物理证据，也正是上面那段抑制逻辑存在的理由。
                    "speak": bool(speaking),
                }
                print(json.dumps(rec, ensure_ascii=False), flush=True)
                last_json = now
                # 该不该主动开口？三个条件：状态变了 / 没人打岔 / 这一刻值得说
                want_narrate = changed and mind_worker is not None
                if want_narrate and user_active:
                    want_narrate = False          # 你在说话或刚问完，别打岔
                if want_narrate and narrate_mode == "quiet":
                    if not _worth_narrating(cur_mode, cur_text):
                        want_narrate = False      # IDLE/MOTION/自己听着呢 → 闭嘴听
                    elif (now - last_narrate_at) < args.narrate_gap:
                        want_narrate = False      # 限速：别刚说完又开口

                if want_narrate:
                    last_narrate_at = now
                    # 把这次状态变化交给"嘴"，后台想台词，不阻塞这个 20Hz 循环
                    cur_ev_id = mind_worker.submit({
                        "mode": cur_mode,
                        "faces": int(vision["faces"]),
                        "motion": float(vision["motion"]),
                        "dbfs": float(audio["dbfs"]),
                        "clap": bool(audio["clap"]),
                        "prev_say": last_say,
                        "heard": "",              # 不是用户提问，是自言自语
                        "dialog": _dialog_snapshot(),
                        "facts": _facts_snapshot(),
                    })

            # ---------- 语言层结果：想好了就单独补一条 TEXT（不动 MODE）----------
            # 为什么不和 MODE 一起发：MODE 要立刻发（设备秒响应），
            # 而想台词要等网络，所以分成两步 —— 先切状态，再补台词。
            if mind_worker is not None:
                mr = mind_worker.latest()
                if mr and mr["seq"] == cur_ev_id and mr["seq"] != applied_ev_id:
                    applied_ev_id = mr["seq"]
                    last_say = mr["say"]
                    send_raw(ser, f"TEXT {mr['tag']}")
                    # 把 MOSS 的回答回填进对话历史（这一轮就完整了）
                    if memory is not None:
                        memory.fill_last_answer(mr["say"])
                    src = mr.get("backend", "mind")
                    used = ",".join(t["name"] for t in (mr.get("tools") or []))
                    note = ""
                    if speak_worker is not None:
                        note = ("  -> 出声" if speak_worker.submit(mr["say"])
                                else "  -> 重复，跳过")
                    print(f"[SAY] {mr['say']}   [OLED] {mr['tag']}   "
                          f"({src}, {mr['ms']}ms"
                          + (f", 工具 {used}" if used else "") + f"){note}",
                          file=sys.stderr)

            # ---------- 阶段 12：听懂用户说了什么 ----------
            # 走到这里说明分段器已经切出一整句、识别也已经出结果了。
            # 有了 heard，语言层的提示词就从"汇报当前状态"变成"回答这个人的问题"。
            if asr_worker is not None:
                ar = asr_worker.latest()
                if ar and ar["seq"] > applied_asr_id and ar["text"]:
                    applied_asr_id = ar["seq"]

                    # ---------- 阶段 15：先过唤醒门 ----------
                    # 门摆在"识别之后、送语言层之前"。为什么不在识别之前拦：
                    # 本地 whisper 跑一句几乎不花钱，真正花的是大模型调用；
                    # 而声学唤醒词要新依赖、没有麦克风就验证不了（见 wake.py 开头）。
                    wd = wake_gate.feed(ar["text"]) if wake_gate is not None else None

                    if wd is not None and not wd.hit:
                        # 没叫它。这不是"听不懂"，是"不该我管"——
                        # 所以既不进记忆、也不送模型，静静放着。
                        n_ignored += 1
                        print(f"[安静] 没喊名字，不回应：{ar['text']}", file=sys.stderr)
                    else:
                        # 被叫醒 / 本来就醒着：走正常对话
                        text_in = wd.text if wd is not None else ar["text"]
                        bare = bool(wd is not None and wd.bare)
                        reason = wd.reason if wd is not None else "off"
                        last_ask_at = time.time()

                        # 统计口径要分开，否则"唤醒次数"和"对话轮数"会混成一个数：
                        #   woken  —— 被叫醒几次（含只喊名字）
                        #   heard  —— 真正问出去几句话（不含只喊名字）
                        if reason.startswith("wake"):
                            n_woken += 1
                            if args.wake_ack == "beep" and link is not None and link.usable:
                                # 用蜂鸣而不是说话：蜂鸣不占语音通道，
                                # 也就不会在半双工下把用户的下一句挡掉。
                                link.send("BEEP 1200 80")

                        print(f"[HEARD] {text_in or '(只喊了名字)'}   "
                              f"(音频{ar['audio_s']}s, {ar['ms']}ms, "
                              f"RTF {ar['rtf']}, {ar['engine']}, {reason})",
                              file=sys.stderr)

                        if bare:
                            # 只喊了名字。把它当问题去问模型会很蠢
                            # （"你叫我干什么"）—— 应一声就够。
                            if speak_worker is not None:
                                speak_worker.submit("在。")
                        else:
                            n_heard += 1
                            if memory is not None:
                                memory.add_turn(text_in, "")
                            if mind_worker is not None:
                                cur_ev_id = mind_worker.submit({
                                    "mode": cur_mode or "LISTEN",
                                    "faces": int(vision["faces"]),
                                    "motion": float(vision["motion"]),
                                    "dbfs": float(audio["dbfs"]),
                                    "clap": bool(audio["clap"]),
                                    "prev_say": last_say,
                                    "heard": text_in,
                                    "dialog": _dialog_snapshot(),
                                    "facts": _facts_snapshot(),
                                })
                            elif speak_worker is not None:
                                # 没开语言层：至少复述一遍，好确认识别对了
                                speak_worker.submit(f"我听到你说：{text_in}")

            # ---------- 预览窗（必须在主线程里显示）----------
            if window_ok:
                frame = vision["frame"]
                if frame is not None:
                    if now - last_fps_at >= 0.5:
                        fps_now = (vw.n_frames - last_v_frames) / (now - last_fps_at)
                        last_fps_at, last_v_frames = now, vw.n_frames
                    canvas = draw_overlay(frame.copy(), vision, audio,
                                          cur_mode, cur_text, fps_now)
                    try:
                        cv2.imshow("MOSS Brain", canvas)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    except cv2.error as e:
                        print(f"[提示] 开不了预览窗，转为无窗口模式：{e}", file=sys.stderr)
                        window_ok = False

            # ---------- 限速到融合频率 ----------
            if loop_period > 0:
                dt = time.time() - iter_t0
                if dt < loop_period:
                    time.sleep(loop_period - dt)

            if args.asr_test and n_heard >= 1:
                break
            if args.frames and iters >= args.frames:
                break

    except KeyboardInterrupt:
        print("\n[提示] 收到 Ctrl+C，退出。", file=sys.stderr)
    finally:
        vw.shutdown()
        aw.shutdown()
        if mind_worker is not None:
            mind_worker.shutdown()
        if speak_worker is not None:
            speak_worker.shutdown()
        if asr_worker is not None:
            asr_worker.shutdown()
            asr_worker.join(timeout=10)      # 让它把队列里剩下的句子识别完
            try:
                asr_worker.engine.close()
            except Exception:
                pass
        if memory is not None:
            # 一定要落盘：不 close 的话，最后几轮对话会丢
            memory.close()
        if detector is not None:
            # ONNX 会话占着几百 MB 和一批线程，显式释放一下
            try:
                detector.close()
            except Exception:
                pass
        if link is not None:
            link.close()                     # 先停收设备的线程，再关串口
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    print(f"[OK] 结束。融合循环 {iters} 次 · 视觉 {vw.n_frames} 帧 · "
          f"听觉 {aw.n_blocks} 块。", file=sys.stderr)
    if detector is not None:
        # 【阶段 18】检测和取帧是两个频率，所以两个数都要打出来 ——
        # 只打"处理了多少帧"会让人以为检测器每帧都在跑（不是，默认 1/3）。
        print(f"     识别：跑检测 {vw.n_detect} 次（每 {args.vision_detect_every} 帧一次）"
              f" · {detector.stats_line() or detector.describe()}", file=sys.stderr)
    if speak_worker is not None:
        s = speak_worker.stats
        print(f"     说话 {s['spoken']} 句 · 被打断丢弃 {s['dropped']} 句 · "
              f"重复跳过 {s['skip']} 句 · 失败 {s['error']} 次", file=sys.stderr)
    if aec is not None:
        # 【阶段 17】AEC 只有在"它在说话"那几段才真的有事干，
        # 所以 aw.n_aec 远小于总块数很正常，不是没工作。
        print(f"     {aec.stats_line()}", file=sys.stderr)
        print(f"     回声消除实际处理 {aw.n_aec} 块"
              + ("（半双工兜底：这个嗓子从头到尾没给过参考信号）"
                 if not getattr(voice, "supports_reference", False) else ""),
              file=sys.stderr)
    if asr_worker is not None:
        s = asr_worker.stats
        seg = segmenter.stats if segmenter is not None else {}
        # 注意"识别成功"和"听懂"是两个数：前者是识别线程**产出**了几句，
        # 后者是主循环**消费**了几句。两者不相等，就说明有结果在
        # "线程产出 → 主循环读取"这一步丢了（单槽 result 会被后来的覆盖）。
        print(f"     听懂 {n_heard} 句 · 切段 {seg.get('utterances', 0)} · "
              f"识别成功 {s['ok']} · 识别为空 {s['empty']} · "
              f"太短丢弃 {seg.get('too_short', 0)} · "
              f"排队溢出丢弃 {s['dropped']} · 识别失败 {s['error']}", file=sys.stderr)
    if wake_gate is not None:
        print(f"     {wake_gate.stats_line()}", file=sys.stderr)
    if behavior is not None:
        print(f"     {behavior.stats_line()}", file=sys.stderr)
    if presence is not None:
        # 【阶段 19】这一行是"检测器到底抖不抖"的健康指标：
        # `闪一下没够着`很大就说明检测器在闪（该去调 --vision-conf、
        # 或者看看光照），而不是来改 presence 的门槛。
        print(f"     变化感知：{presence.stats_line()}", file=sys.stderr)
    if toolbox is not None:
        print(f"     工具：{toolbox.describe()}", file=sys.stderr)
    if link is not None:
        print(f"     执行器：{link.describe()}", file=sys.stderr)
    if memory is not None:
        m = memory.summary()
        print(f"     记忆：{m['path']}（对话 {m['turns']} 轮 · 长期事实 {m['facts']} 条 · "
              f"第 {m['sessions']} 次启动）", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
