#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOSS 语言层（"嘴"）  ·  流浪地球 550W 复刻项目 · 阶段 10（阶段 12/13 扩展）
============================================================
阶段 7/8/9 让 MOSS 有了眼睛、耳朵和融合判断。这一步给它一张嘴：
把「现在是什么状态」变成一句**人话**。

阶段 12 之后又多了一件事：**当有人跟它说话时，它得回答，而不是汇报状态。**
事件里带 `heard` 字段就切到"问答模式"，带 `dialog` 就带上最近几轮对话。

阶段 13 再加一件事：**它可以主动调用工具**（阶段 13 的 moss_tools.py）。
需要查时间/天气、要记住或回忆时，它会先"申请调用工具"，由 MindWorker
在本地执行，把真实结果再喂回来，然后它才说最终那句话 —— 这就是
function calling，只是没用 OpenAI 的原生 tools 协议（理由见 moss_tools.py）。

两种模式的提示词见下面的 SYS_PROMPT_BASE / TOOL_PROMPT，
拼装逻辑在 LLMClient._payload()。

⚠️ 一个必须先讲清的硬约束
------------------------------------------------------------
OLED 用的是 Adafruit 默认字体，**只有 ASCII，写汉字会显示成方块**。
可我们又希望 MOSS 说的是中文（那才有 550W 的味道）。

所以这里让模型一次返回**两个字段**：

    {"say": "我看见你了。", "tag": "i see you"}
     ↑ 中文台词                  ↑ ≤13 个 ASCII 字符

    say  → 打到终端 / 送进 TTS 出声
    tag  → 通过串口 TEXT 命令发给 ESP32，显示在 OLED 第三行

一句话解决"想说中文"和"屏幕只认英文"的矛盾。

三种后端
------------------------------------------------------------
    offline  零配置：内置模板台词，不联网也能用（默认兜底）
    api      OpenAI 兼容接口（DeepSeek / 通义 / 智谱 / OpenAI / 本地 ollama 都行）
    auto     先试 api，失败自动退回 offline

为什么用 urllib 而不是 requests / openai 库
------------------------------------------------------------
少一个依赖 = 少一个新手踩坑点。HTTP 请求用标准库 urllib 就够了。

为什么单独开线程
------------------------------------------------------------
一次网络请求可能几百毫秒（调了工具还要再来一轮，翻倍），
而融合循环是 20Hz（每 50ms 一次）。同步调用会把整个大脑卡住。
所以放进 MindWorker 线程，主循环只管取结果。
"""

import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request

# OLED 第三行从 x=50 开始画，128 宽的屏只剩 78px，默认字体 6px 宽 → 13 个字符
TAG_MAX = 13


# ================================================================
#  1. 台词清理
# ================================================================
def clean_tag(s, limit=TAG_MAX):
    """把模型给的标签洗成 OLED 能显示的：只保留可打印 ASCII，截断到 limit。"""
    if not isinstance(s, str):
        s = str(s)
    s = "".join(ch for ch in s if 32 <= ord(ch) < 127)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return "online"
    return s[:limit]


def clean_say(s, limit=60):
    """把中文台词洗干净（去掉换行和多余空格），只给终端用，不限制中文。"""
    if not isinstance(s, str):
        s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:limit] if s else "……"


# ================================================================
#  2. 离线模板大脑（零配置兜底）
# ================================================================
# 【阶段 14】"转几圈"的口语 → 圈数。出现频率高，单独提出来免得 lambda 里塞字典。
_TURNS = {"半圈": 0.5, "一圈": 1.0, "一转": 1.0,
          "两圈": 2.0, "三圈": 3.0, "几圈": 1.0}

# 颜色词 → 规范化颜色名（离线规则里抓到的字直接能喂给 set_led）
_COLOR_WORDS = ("红", "橙", "黄", "绿", "青", "蓝", "紫", "白", "灭")


class OfflineMind:
    """不联网的模板台词。每个状态配几句，轮流说，避免重复死板。

    阶段 12 之后多了一条：如果这次带着 `heard`（听懂了用户说的话），
    就不要再"汇报感知"了 —— 那是答非所问。

    阶段 13 又多了一条：**离线模式也能用工具**，靠的是关键词规则。
    它没法像大模型那样理解语言，但"几点""我叫什么""忘掉…"这类说法
    边界很清楚，几条正则就能覆盖，效果比"编一个时间"好得多。
    """

    name = "offline"

    TEMPLATES = {
        "BOOT":   [("系统自检完成。", "boot ok"), ("各单元上线。", "systems up")],
        "IDLE":   [("我在待机。", "idle"), ("随时待命。", "standing by")],
        "LISTEN": [("我在听。", "listening"), ("请讲。我在听。", "i'm listening")],
        "THINK":  [("让我想想。", "thinking"), ("正在计算。", "computing")],
        "SPEAK":  [("有人在跟我说话。", "face+voice"), ("我听你在说。", "you talking")],
        "VISION": [("我看见你了。", "i see you"), ("有人在画面里。", "person seen")],
        "MOTION": [("有东西在动。", "motion"), ("画面里有动静。", "something mov")],
        "ERROR":  [("检测到异常。", "fault"), ("出问题了。", "error")],
    }

    HEARD = ["我收到了。", "听到了。", "我在听你说。"]

    # 闲聊直答（不需要工具、也不需要复述）。
    # 这些说法边界清楚，模板就能答得比"复述一遍"好。
    SMALLTALK = [
        (r"你是谁|你叫什么名字|你叫什么|你是什么东西|介绍一下你自己",
         "我是550W，行星发动机的控制中枢。", "i am 550w"),
        (r"(你好|您好|在吗|早上好|下午好|晚上好|嗨|哈喽)",
         "我在。", "hello"),
        (r"(谢谢|多谢|感谢|辛苦了)", "不客气。", "welcome"),
        (r"(再见|拜拜|回头见|我走了)", "随时待命。", "goodbye"),
    ]

    # 工具触发规则（离线兜底）。顺序有意义：先匹配到的先赢。
    # 每条 = (正则, 工具名, 参数构造函数(match) -> dict)
    RULES = [
        # 报时 / 报日期
        (r"(几点|时间|日期|几号|星期几|今天是)",
         "get_time", lambda m: {}),
        # "我叫张三" / "我的名字是张三"
        (r"(?:我叫|我的名字是|我的名字叫)([\u4e00-\u9fffA-Za-z0-9]{1,12})",
         "remember", lambda m: {"key": "用户的名字", "value": m.group(1)}),
        # "记住X是Y"
        (r"记住[，,]?\s*([\u4e00-\u9fffA-Za-z0-9]{2,12})\s*(?:是|叫|为)\s*([\u4e00-\u9fffA-Za-z0-9]{1,20})",
         "remember", lambda m: {"key": m.group(1), "value": m.group(2)}),
        # "我住/家在X"
        (r"(?:我住在|我家在|我在)([\u4e00-\u9fff]{2,8})(?:市|城|区)?(?:住|生活)?$",
         "remember", lambda m: {"key": "用户所在城市", "value": m.group(1)}),
        # "我是谁" / "我叫什么" / "你还记得我吗"
        (r"(我是谁|我叫什么|还记得我|记得我吗|你记得我)",
         "recall", lambda m: {"query": ""}),
        # "忘掉X" / "忘记X"
        (r"(?:忘掉|忘记|删除记忆)\s*([\u4e00-\u9fffA-Za-z0-9]{2,12})",
         "forget", lambda m: {"key": m.group(1)}),
        # "深圳的天气" / "今天天气" （没给城市就问记忆里那个）
        # 注意 {2,8}? 是**非贪婪**：否则"深圳的天气"会贪婪吃成"深圳的"。
        (r"(?:([\u4e00-\u9fff]{2,8}?)(?:市|的)?)?(?:今天)?(?:的)?(?:天气|气温)",
         "get_weather", lambda m: {"city": m.group(1) or ""}),
        # "看看我" / "你现在看到什么"
        (r"(看看我|看到什么|画面里|有人吗|看得见)",
         "look", lambda m: {}),
        # "你运行多久了" / "你自己的状态"
        (r"(运行多久|状态怎么样|统计|你还行吗)",
         "stats", lambda m: {}),

        # ---------- 阶段 14：执行器（离线兜底也要能"动手"）----------
        # 顺序仍然重要：越具体的说法放越前面。
        # "把灯变成蓝色" / "灯环变绿" / "变红灯"
        (r"(?:灯|灯环|环形灯)[^，,。！？]{0,6}?([红橙黄绿青蓝紫白灭])",
         "set_led", lambda m: {"color": m.group(1)}),
        (r"(?:变|亮|调|换)(?:成|为)?\s*([红橙黄绿青蓝紫白灭])色?",
         "set_led", lambda m: {"color": m.group(1)}),
        # "转半圈" / "转两圈"
        (r"(?:电机|马达|转)[^，,。！？]{0,4}?(半圈|一圈|一转|两圈|三圈|几圈)",
         "turn_motor", lambda m: {"turns": _TURNS.get(m.group(1), 0.5)}),
        # "转一下" / "转动一下"（没给角度就默认半圈）
        (r"(转一下|转一转|转动|旋转)",
         "turn_motor", lambda m: {"turns": 0.5}),
        # "响一声" / "嘀一下"
        (r"(响一声|响一下|嘀一声|嘀一下|哔一声|提示音|叫一声)",
         "beep", lambda m: {}),
        # "急停" / "停下" / "别转了"
        (r"(急停|停下来|停下|别动|别转|安静|住手)",
         "stop_all", lambda m: {}),
        # "屏幕上写 HELLO"（只抓 ASCII，中文写了也显示不出来）
        (r"(?:屏幕|OLED|oled)[^，,。！？]{0,6}?(?:写|显示|打)(?:上|出)?\s*([A-Za-z0-9 _.\-]{1,13})",
         "show_text", lambda m: {"text": m.group(1)}),
        # "设备在线吗" / "板子状态怎么样"
        (r"(?:设备|板子|单片机)[^，,。！？]{0,4}?(?:状态|在线|连上|怎么样|好了吗)",
         "device_status", lambda m: {}),
    ]

    def __init__(self):
        self.n = 0

    def _match_tool(self, heard):
        """离线模式的"意图识别"：命中了就返回一个工具请求。"""
        for pat, name, mk in self.RULES:
            m = re.search(pat, heard)
            if m:
                return {"name": name, "args": mk(m)}
        return None

    def respond(self, ev):
        """返回 (say, tag, tool)。命中规则时只说"我去办"，话等到工具结果回来再说。"""
        self.n += 1
        heard = (ev.get("heard") or "").strip()
        if heard:
            for pat, s, t in self.SMALLTALK:
                if re.search(pat, heard):
                    return clean_say(s), clean_tag(t), None
            tool = self._match_tool(heard)
            if tool is not None:
                return "", "", tool            # 先别说话，等信息拿到
            say = self.HEARD[(self.n - 1) % len(self.HEARD)] + f"你说的是「{heard}」。"
            return clean_say(say), clean_tag("heard"), None
        pool = self.TEMPLATES.get(ev.get("mode"), [("……", "online")])
        say, tag = pool[(self.n - 1) % len(pool)]
        return clean_say(say), clean_tag(tag), None

    def respond_with_result(self, ev, tool, ok, result):
        """工具跑完了，把结果说成人话。"""
        name = tool.get("name")
        if not ok:
            return clean_say(f"这事我没办成：{result}"), clean_tag("tool fail")
        say, tag = result, "tool done"
        if name == "get_time":
            say, tag = result + "。", "time"
        elif name == "remember":
            say, tag = "好，我记住了。", "memorized"
        elif name == "recall":
            say, tag = "我记得，" + result, "recall"
        elif name == "forget":
            say, tag = result, "forgot"
        elif name == "get_weather":
            say, tag = result, "weather"
        elif name == "look":
            say, tag = result, "scene"
        elif name == "stats":
            say, tag = result, "stats"
        # ---------- 阶段 14：执行器（说人话 + 给个短标签）----------
        elif name == "set_led":
            say, tag = result, "led"
        elif name == "turn_motor":
            say, tag = result, "motor"
        elif name == "beep":
            say, tag = result, "beep"
        elif name == "show_text":
            say, tag = result, "screen"
        elif name == "stop_all":
            say, tag = result, "halted"
        elif name == "device_status":
            say, tag = result, "device"
        return clean_say(say), clean_tag(tag)


# ================================================================
#  3. LLM 大脑（OpenAI 兼容接口）
# ================================================================
DEFAULT_PERSONA = (
    "你是《流浪地球》里的行星发动机人工智能 550W（代号 MOSS）。"
    "你冷静、简短、略带机器感，像在对面前的人做实时汇报。"
)

SYS_PROMPT_BASE = """你是一个嵌入式设备的表情与台词模块。设备只有一个 128x64 的单色 OLED，
且**只能显示 ASCII，不能显示汉字**。

每次会给你设备当前的感知状态，你要返回**恰好一行 JSON**，不要任何多余文字、不要代码块：

{"say": "<一句中文台词，10~20 字，符合角色语气>", "tag": "<给 OLED 的英文标签，最多 13 个 ASCII 字符，小写>"}

两种模式，看输入里有没有「用户刚刚对你说」这句话：

一、**有人跟你说话**（有「用户刚刚对你说：…」）
   → 你要**回答他**，不是描述你自己的状态。
   → 直接给答案或回应，像 MOSS 一样冷静简短。别反问"你需要什么帮助"这种废话。
   → 例：用户说"现在几点了" → say 可以是"本地时间下午三点十四分。"
                   → 用户说"你是谁" → say 可以是"我是550W，行星发动机的控制中枢。"

二、**没人跟你说话**（没有那句话）
   → 汇报你此刻的感知，例如"我看见你了""我在待机"。

规则：
- say 用中文，口语化、简短，不要标点堆砌，不要引号。
- tag 最多 13 个字符、只能 ASCII、不要换行；写不下就缩写。
- 上面两种模式都**只输出 JSON 本身**。"""

# 阶段 13：工具协议。没开工具时这段不会拼进去，提示词保持原样。
#
# ⚠️ 这里刻意用 {tools} 当作占位符，但**只能用 str.replace 注入**，
#    不能用 str.format —— 因为正文里的 JSON 示例本身就带花括号
#    （{"say": ...}），format 会把它们当成占位符，直接抛 KeyError。
#    这个坑真实发生过：表现为"API 第一次调用就莫名降级成模板台词"。
TOOL_PROMPT = """
────────────────────────
你还能**调用工具**：既能拿真实信息，也能**真的动手**。

需要时，就在 JSON 里多给一个 "tool" 字段：

{"say": "", "tag": "", "tool": {"name": "<工具名>", "args": {<参数>}}}

可用工具：
{tools}

它们分两类，用法不同：
- **查询类**（get_time / get_weather / look / recall / stats / device_status…）：
  去拿真实信息，不要凭空编。
- **动作类**（set_led / turn_motor / beep / show_text / stop_all）：
  **真的驱动硬件**。用户让你"把灯变红""转一下""响一声""在屏幕上写字"时用这一类。

调用规则（**很重要，违反会让设备说错话**）：
- 只在**真的需要外部信息**（当前时间、天气）或**用户明确让你做事**时才调工具。
  闲聊、打招呼、评价画面都不需要工具。
- 用户让你做动作时，**先调工具再报告**，不要只嘴上答应"好的我这就办"却不调工具。
- 调用工具的那一轮，say 和 tag 都留空字符串 —— 我们会先执行工具，
  再把真实结果交给你，让你说最终那一句。
- 一次只调**一个**工具，args 必须是 JSON 对象（没有参数就写空对象）。
- 用户告诉你关于他自己的事（名字、城市、喜好、习惯）时，用 remember 记下来，
  这样下次启动你还记得。用户问起时用 recall 去查，不要凭印象猜。
- **不要编造**工具能查到的信息。查时间就调 get_time，查天气就调 get_weather。"""


class LLMClient:
    """OpenAI 兼容的 chat/completions 客户端。无第三方依赖。

    开了工具（`tool_catalog` 非空）时，会多走一条路：
    模型说要调工具 → 我们执行 → 带着结果再问一次 → 拿最终台词。
    见 respond() / respond_with_result()。
    """

    name = "api"

    def __init__(self, base, key, model, timeout=8.0, persona=DEFAULT_PERSONA,
                 temperature=0.9, max_tokens=96, tool_catalog="", max_rounds=1):
        self.url = base.rstrip("/") + "/chat/completions"
        self.key = key
        self.model = model
        self.timeout = timeout
        self.persona = persona
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tool_catalog = tool_catalog or ""    # 阶段 13：工具清单（空=不开工具）
        self.max_rounds = max(1, int(max_rounds))  # 最多连续调几次工具（防死循环）

    # ---------- 提示词 ----------
    def _sys_prompt(self):
        p = self.persona + "\n\n" + SYS_PROMPT_BASE
        if self.tool_catalog:
            # 用 replace 而不是 format：正文里的 JSON 花括号会和 format 打架
            p += TOOL_PROMPT.replace("{tools}", self.tool_catalog)
        return p

    def _payload(self, ev, tool_result=None):
        heard = (ev.get("heard") or "").strip()
        dialog = (ev.get("dialog") or "").strip()
        facts = (ev.get("facts") or "").strip()

        base = (
            f"当前状态：{ev.get('mode')}\n"
            f"画面里的人数：{ev.get('faces', 0)}\n"
            f"画面运动量：{ev.get('motion', 0):.3f}\n"
            f"麦克风电平：{ev.get('dbfs', -99):.1f} dBFS\n"
            f"是否听到拍手：{'是' if ev.get('clap') else '否'}\n"
        )
        # 阶段 13：长期记忆里的用户信息，让模型"知道对面是谁"
        if facts:
            base += f"\n你知道的关于用户的事（长期记忆）：\n{facts}\n"

        if heard:
            # 模式一：有人跟我说话 —— 这是阶段 12 带来的根本变化。
            # 之前 MOSS 只会"汇报感知"，现在它得"回答"。
            user = base
            if dialog:
                user += f"\n最近的对话（越靠后越新）：\n{dialog}\n"
            user += (f"\n★ 用户刚刚对你说：{heard}\n"
                     "请回答他（不要描述你自己的状态），10~25 字。")
        else:
            # 模式二：没人说话，还是老样子 —— 汇报感知
            user = base + (f"上一句台词：{ev.get('prev_say') or '（无）'}\n"
                           "现在没人跟你说话，请汇报你此刻的感知。")

        # 第二轮：把工具的真实结果交回去，让它说最终那一句
        if tool_result is not None:
            tname, targs, tok, tval = tool_result
            user += ("\n\n【你刚才申请调用的工具已经执行完毕】\n"
                     f"工具：{tname}\n"
                     f"参数：{json.dumps(targs or {}, ensure_ascii=False)}\n"
                     f"执行：{'成功' if tok else '失败'}\n"
                     f"结果：{tval}\n"
                     "现在请**根据这个真实结果**给出你要说的话（仍然只输出 JSON，"
                     "不要再调工具，除非这个结果明显不够用）。")

        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._sys_prompt()},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

    # ---------- 请求 ----------
    def _chat(self, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
        data = json.loads(raw)
        return data["choices"][0]["message"]["content"]

    def respond(self, ev):
        """第一轮：可能直接回答，也可能"申请调用工具"。

        返回 (say, tag, tool)。tool 为 None 表示不需要工具；
        需要工具时 say/tag 是空的 —— 别急着说话，等工具结果回来。
        """
        content = self._chat(self._payload(ev))
        say, tag, tool = parse_reply(content)
        if tool and not say.strip():
            return "", "", tool
        return clean_say(say), clean_tag(tag), tool

    def respond_with_result(self, ev, tool, ok, result):
        """第二轮：带着工具的真实结果再问一次，拿最终台词。"""
        content = self._chat(self._payload(
            ev, tool_result=(tool.get("name"), tool.get("args"), ok, result)))
        say, tag, _ = parse_reply(content)
        return clean_say(say), clean_tag(tag)


def parse_reply(content):
    """从模型回复里抠出 (say, tag, tool)。

    模型不一定老实：可能包一层 ```json、可能多几句解释。
    所以先试直接 json.loads，失败就用正则找第一个 {...} 块。
    say/tag 这里**不做默认填充**（空就返回空），交给调用方判断 ——
    因为"调工具时故意留空"和"模型答不上来"是两回事。
    """
    text = (content or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()

    obj = None
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None

    if isinstance(obj, dict) and ("say" in obj or "tag" in obj or "tool" in obj):
        say = obj.get("say", "")
        tag = obj.get("tag", "")
        tool = _clean_tool(obj.get("tool"))
        if not tag and say:
            tag = say                    # 只给了 say，就把 say 当标签洗一遍
        return str(say), str(tag), tool

    # 完全不是 JSON → 把整段当台词，取前 13 个 ASCII 当标签
    return text or "", text or "", None


def _clean_tool(tool):
    """把模型给的 tool 字段洗成安全的 {name, args}。不合法就返回 None。"""
    if not isinstance(tool, dict):
        return None
    name = tool.get("name") or tool.get("工具") or tool.get("tool")
    if not isinstance(name, str) or not name.strip():
        return None
    args = tool.get("args")
    if args is None:
        args = tool.get("arguments") or tool.get("参数")
    if not isinstance(args, dict):
        args = {}
    return {"name": name.strip()[:40], "args": args}


# ================================================================
#  4. 自动后端：先试 API，失败退回模板
# ================================================================
class AutoMind:
    """先试远端，失败就永久退回模板（并只提示一次，不刷屏）。"""

    name = "auto"

    def __init__(self, primary, fallback, on_fallback=None):
        self.primary = primary
        self.fallback = fallback
        self.on_fallback = on_fallback
        self.use_fallback = False

    def _degrade(self, e):
        if not self.use_fallback:
            self.use_fallback = True
            if self.on_fallback:
                self.on_fallback(e)

    def respond(self, ev):
        if not self.use_fallback:
            try:
                return self.primary.respond(ev)
            except Exception as e:
                self._degrade(e)
        return self.fallback.respond(ev)

    def respond_with_result(self, ev, tool, ok, result):
        if not self.use_fallback:
            try:
                return self.primary.respond_with_result(ev, tool, ok, result)
            except Exception as e:
                self._degrade(e)
        return self.fallback.respond_with_result(ev, tool, ok, result)


def build_mind(kind, base, key, model, timeout, persona, tool_catalog="",
               max_rounds=1):
    """按 --llm 的值造一个后端。返回 (mind, 说明字符串)。

    tool_catalog 非空 = 开了工具，会拼进提示词（阶段 13）。
    """
    if kind == "offline":
        return OfflineMind(), "离线模板"
    if kind == "api":
        if not key:
            raise RuntimeError("没给 API Key。用 --api-key 或设环境变量 MOSS_API_KEY。")
        return (LLMClient(base, key, model, timeout, persona,
                          tool_catalog=tool_catalog, max_rounds=max_rounds),
                f"API {base} / {model}")
    if kind == "auto":
        if key:
            fb = lambda e: print(f"[提示] LLM 调用失败（{e}），已退回离线模板。", file=sys.stderr)
            return (AutoMind(LLMClient(base, key, model, timeout, persona,
                                       tool_catalog=tool_catalog, max_rounds=max_rounds),
                             OfflineMind(), fb),
                    f"自动（优先 API {base}，失败退模板）")
        print("[提示] 没找到 API Key，--llm auto 使用离线模板。", file=sys.stderr)
        return OfflineMind(), "离线模板（未配 Key）"
    raise ValueError(f"未知的 --llm 值：{kind}")


# ================================================================
#  5. 异步工作线程
# ================================================================
class MindWorker(threading.Thread):
    """把"想台词"放到后台线程，绝不阻塞 20Hz 的融合循环。

    要点：
    - 只保留**最新**一个待处理事件（旧的状态已经过时，不必再想）
    - 每个事件带一个自增 seq；主循环用它判断结果是不是"当前这一句"
    - 任何异常都变成 ok=False 的结果，同时退回模板台词，绝不让线程死掉
    - 阶段 13：如果后端说"我要调工具"，在这里执行并把结果交回去再问一轮

    为什么工具编排放在 Worker 而不是后端里
    ------------------------------------
    后端（LLMClient / OfflineMind）只负责"决定调不调、调哪个"，
    **执行**由 Worker 通过 toolbox 做。这样"能不能调"这一个开关
    只有一处（toolbox 为 None = 全部禁用），也方便把每次调用记进日志。
    """

    def __init__(self, mind, fallback=None, toolbox=None, max_rounds=1):
        super().__init__(daemon=True, name="mind")
        self.mind = mind
        self.fallback = fallback or OfflineMind()
        self.toolbox = toolbox           # 阶段 13：ToolBox 或 None
        self.max_rounds = max(1, int(max_rounds))
        self.degraded = False       # 一旦远端失败过，就永久改走模板（见 run()）
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.pending = None            # 待处理事件
        self.result = None             # 最新结果 dict
        self.seq = 0
        self.n_tool_calls = 0          # 一共执行过多少次工具

    def submit(self, ev):
        """投递一个状态事件，返回它的序号（主循环用它做校验）。"""
        with self.cv:
            self.seq += 1
            ev = dict(ev)
            ev["_seq"] = self.seq
            self.pending = ev
            self.cv.notify()
            return self.seq

    def latest(self):
        """取最新结果（没有就返回 None）。"""
        with self.lock:
            return dict(self.result) if self.result else None

    def _backend_label(self, ok):
        name = getattr(self.mind, "name", "mind")
        if self.degraded:
            return f"{name} 失败兜底" if not ok else f"{name}→模板"
        return name

    def run(self):
        while not self.stop.is_set():
            with self.cv:
                while self.pending is None and not self.stop.is_set():
                    self.cv.wait(0.2)
                ev = self.pending
                self.pending = None
            if ev is None:
                continue

            seq = ev.pop("_seq")
            t0 = time.time()
            ok, err = True, ""
            tool_log = []

            def first():
                """第一轮：可能直接给台词，也可能申请调工具。"""
                if self.degraded:
                    # 已经退过一次 → 之后不再浪费时间去连远端
                    return self.fallback.respond(ev)
                return self.mind.respond(ev)

            def second(tool, tok, tval):
                """第二轮：带着工具的真实结果再要一次台词。"""
                if self.degraded:
                    return self.fallback.respond_with_result(ev, tool, tok, tval)
                return self.mind.respond_with_result(ev, tool, tok, tval)

            def degrade(ex):
                """远端坏了就切模板。**只切一次** —— 否则每次都等超时，
                结果回来时状态早变了、全被丢弃（阶段 10 踩过的坑）。"""
                nonlocal ok, err
                ok, err = False, f"{ex}"
                if not self.degraded:
                    self.degraded = True
                    print(f"[提示] 台词后端出错（{ex}），已切换为离线模板。",
                          file=sys.stderr)

            try:
                say, tag, tool = first()
            except Exception as e:
                degrade(e)
                say, tag, tool = self.fallback.respond(ev)

            # ---------- 阶段 13：工具编排 ----------
            rounds = 0
            while (tool is not None and self.toolbox is not None
                   and rounds < self.max_rounds and not self.stop.is_set()):
                rounds += 1
                name, args = tool.get("name"), tool.get("args")
                tok, tval = self.toolbox.call(name, args)
                tool_log.append({"name": name, "args": args,
                                 "ok": tok, "result": tval})
                self.n_tool_calls += 1
                try:
                    say, tag = second(tool, tok, tval)
                except Exception as e:
                    degrade(e)
                    say, tag = self.fallback.respond_with_result(ev, tool, tok, tval)
                tool = None      # 只有一轮：respond_with_result 不会再给 tool

            if not (say or "").strip():
                # 模型只申请了工具、却没东西可执行（比如工具被关掉了），
                # 那就兜一句，别让 OLED 空着。
                say, tag = "我在。", "online"

            with self.lock:
                self.result = {"seq": seq, "say": say, "tag": tag,
                               "ok": ok, "err": err, "ms": int((time.time() - t0) * 1000),
                               "mode": ev.get("mode"),
                               "tools": tool_log,
                               "backend": self._backend_label(ok)}

    def shutdown(self):
        self.stop.set()
        with self.cv:
            self.cv.notify_all()


# ================================================================
#  6. 自检入口：python mind.py --llm offline
#     也可以打真 API：python mind.py --llm api --api-key sk-xxx
# ================================================================
def _selftest():
    import argparse
    import tempfile

    p = argparse.ArgumentParser(description="MOSS 语言层自检")
    p.add_argument("--llm", default="offline",
                   choices=["offline", "api", "auto"], help="后端类型")
    p.add_argument("--api-base", default="https://api.deepseek.com/v1")
    p.add_argument("--api-key", default="")
    p.add_argument("--model", default="deepseek-chat")
    p.add_argument("--timeout", type=float, default=8.0)
    p.add_argument("--persona", default=DEFAULT_PERSONA)
    p.add_argument("--tools", action="store_true",
                   help="开工具（离线模式下走关键词规则）")
    p.add_argument("--actuators", action="store_true",
                   help="连一个假设备，把执行器工具（阶段14）也一起测")
    a = p.parse_args()

    # 只有 --tools 时才准备 ToolBox
    toolbox = None
    if a.tools:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from memory import MemoryStore
            from moss_tools import ToolBox
            d = tempfile.mkdtemp(prefix="moss_mind_")
            mem = MemoryStore(os.path.join(d, "m.json"), quiet=True)

            # 阶段 14：要测执行器就得有"设备"。用一个只记录指令的假串口，
            # 这样不用真板子也能验证"离线规则有没有正确动到硬件"。
            dev = None
            if a.actuators:
                import moss_actuators

                class _Fake:
                    def __init__(self):
                        self.lines = []

                    def write(self, b):
                        self.lines.append(b.decode("ascii").strip())

                    def flush(self):
                        pass

                    def readline(self):
                        return b""

                fake = _Fake()
                # read_back=False：没有真设备回话，别让收线程空转
                dev = moss_actuators.ActuatorLink(
                    fake, log=lambda s: print("    " + s), read_back=False)
                moss_actuators.install(dev, log=lambda s: print("    " + s))
                a._fake_dev = fake

            toolbox = ToolBox(
                context={"vision": {"faces": 1, "motion": 0.01}, "mode": "LISTEN",
                         "memory": mem, "uptime": 120.0, "heard": 2,
                         "device": dev},
                log=lambda s: print("    " + s))
        except ImportError as e:
            print(f"[提示] 工具层没起来（{e}），只测台词。", file=sys.stderr)

    mind, desc = build_mind(a.llm, a.api_base, a.api_key, a.model, a.timeout,
                            a.persona,
                            tool_catalog=(toolbox.catalog() if toolbox else ""))
    print(f"[后端] {desc}")

    print("\n[1] 无人说话 → 汇报感知")
    for mode in ("BOOT", "IDLE", "LISTEN", "SPEAK", "VISION", "MOTION", "ERROR"):
        ev = {"mode": mode, "faces": 1 if mode in ("SPEAK", "VISION") else 0,
              "motion": 0.02, "dbfs": -30.0, "clap": mode == "LISTEN"}
        say, tag, tool = mind.respond(ev)
        assert tool is None, "汇报感知不该调工具"
        print(f"  {mode:<7} say={say!r:<28} tag={tag!r} ({len(tag)} 字符)")

    print("\n[2] 有人说话 → 回答问题")
    for heard in ("你是谁", "帮我看看画面里有什么"):
        ev = {"mode": "LISTEN", "faces": 1, "motion": 0.01, "dbfs": -30.0,
              "clap": False, "heard": heard, "dialog": "", "facts": ""}
        say, tag, tool = mind.respond(ev)
        print(f"  听到 {heard!r}\n    → say={say!r}  tag={tag!r}"
              + (f"  [申请调用 {tool['name']}]" if tool else ""))

    print("\n[3] 工具编排（这才是阶段 13 的重点）")
    if toolbox is None:
        print("  跳过：加 --tools 才能测")
    else:
        for heard in ("现在几点了", "我叫张三", "我是谁", "深圳的天气怎么样"):
            ev = {"mode": "LISTEN", "faces": 1, "motion": 0.01, "dbfs": -30.0,
                  "clap": False, "heard": heard, "dialog": "", "facts": ""}
            say, tag, tool = mind.respond(ev)
            if tool is None:
                print(f"  听到 {heard!r}\n    → 未调工具，直接说 {say!r}")
                continue
            print(f"  听到 {heard!r}\n    → 申请 {tool['name']}({tool['args']})")
            tok, tval = toolbox.call(tool["name"], tool["args"])
            say2, tag2 = mind.respond_with_result(ev, tool, tok, tval)
            print(f"    ← 工具结果 {tval!r}\n    → 最终台词 {say2!r}  tag={tag2!r}")

    print("\n[3b] 执行器（阶段 14）：离线规则能不能「动手」")
    if toolbox is None or getattr(a, "_fake_dev", None) is None:
        print("  跳过：加 --tools --actuators 才能测")
    else:
        before = len(a._fake_dev.lines)
        for heard in ("把灯变成红色", "转半圈", "响一声", "屏幕上写 HELLO",
                      "急停", "设备在线吗"):
            ev = {"mode": "LISTEN", "faces": 1, "motion": 0.01, "dbfs": -30.0,
                  "clap": False, "heard": heard, "dialog": "", "facts": ""}
            say, tag, tool = mind.respond(ev)
            if tool is None:
                print(f"  听到 {heard!r}\n    → 未调工具，直接说 {say!r}   ✗")
                continue
            tok, tval = toolbox.call(tool["name"], tool["args"])
            print(f"  听到 {heard!r}\n    → 申请 {tool['name']}({tool['args']})\n"
                  f"    → 设备收到 {a._fake_dev.lines[-1]!r}")
        print("  设备实际收到的指令：")
        for ln in a._fake_dev.lines[before:]:
            print(f"    > {ln}")
        assert len(a._fake_dev.lines) >= before + 5, "执行器规则没几条命中"
        print("  ✓ 6 类「动手」说法全部命中，并真的发出了串口指令")

    print("\n[4] 边界：模型只申请工具、不说台词")
    if toolbox is not None:
        m2 = OfflineMind()
        assert m2.respond({"heard": "现在几点"})[0] == "", "调工具那轮应该不说话"
        print("  ✓ 调工具时 say 为空（等信息回来再说），符合协议")

    print("\n[OK] 语言层自检通过。")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest())
