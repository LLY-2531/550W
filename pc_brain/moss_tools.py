#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOSS 工具层（"手"）  ·  流浪地球 550W 复刻项目 · 阶段 13
============================================================
阶段 12 结束时 MOSS 能聊天了 —— 但它只会**聊**，什么也做不了。
你问它"几点了"，它只能编一个时间；你说"记住我叫张三"，它下一句就忘了。

这一步给模型一双手：**它可以主动要求"调用某个工具"，由我们在本地执行，
把真实结果再喂回去，然后它才开口。**

这就是 function calling（工具调用），只是没用 OpenAI 那套原生 tools 协议 ——
理由见下。

为什么不用 OpenAI 的原生 tools 字段
------------------------------------------------------------
1. **兼容性**：本项目的卖点之一是"任何 OpenAI 兼容接口都能用"。但
   DeepSeek / 通义 / 智谱 / 本地 ollama 对 function_calling 的支持程度
   差别很大，有的不支持、有的格式有微妙差异，踩坑成本高。
2. **我们本来就要自定义格式**：每轮都必须返回 `{"say","tag"}`（OLED 只认
   ASCII），这已经是一个自定义协议了。再叠一层原生 tools 反而更乱。
3. **可控**：工具由**我们**执行。模型只是"提出申请"，能不能调、参数怎么清、
   超时怎么办，全在我们手里 —— 这对一个要连硬件的东西很重要。

于是协议只加一个可选字段：

    {"say": "...", "tag": "...", "tool": {"name": "get_time", "args": {}}}

没有 tool 字段 = 普通回答。有 = 先执行再答。

工具的写法：一个装饰器就注册好了
------------------------------------------------------------
    @tool("get_time", "查当前日期和时间", args="无参数")
    def _t_time(args, ctx):
        return "现在是 ..."

`ctx` 是运行时上下文（当前画面快照、听没听到声音、记忆库……），
由 moss_brain 每轮塞进来。工具不直接 import 主程序，保持单向依赖。

自检：
    python moss_tools.py            # 不需要网络和硬件，联网工具允许失败
"""

import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request

# 单个工具的结果最长给模型看多少字（太长了浪费 token，也容易把模型带偏）
RESULT_MAX = 240
# 联网工具的默认超时
NET_TIMEOUT = 6.0

REGISTRY = {}
_LOCK = threading.RLock()


def tool(name, desc, args="无参数", net=False):
    """注册一个工具。

    name : 模型要填的标识（英文小写，见提示词）
    desc : 一句中文说明，会出现在提示词里
    args : 参数说明，同样会出现在提示词里
    net  : 是否要联网（自检时允许它失败）
    """
    def deco(fn):
        with _LOCK:
            REGISTRY[name] = {
                "name": name, "desc": desc, "args": args, "net": net,
                "fn": fn, "calls": 0, "errors": 0, "ms": 0,
            }
        return fn
    return deco


def _s(v, n=120):
    """把参数洗成安全、简短的单行字符串。"""
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        v = json.dumps(v, ensure_ascii=False)
    return " ".join(str(v).split())[:n]


# ================================================================
#  工具实现
# ================================================================
_WEEK = ["一", "二", "三", "四", "五", "六", "日"]


@tool("get_time", "查当前的日期和时间（本地时钟）", args="无参数")
def _t_time(args, ctx):
    lt = time.localtime()
    return ("现在是 %d年%d月%d日 星期%s %02d:%02d:%02d"
            % (lt.tm_year, lt.tm_mon, lt.tm_mday, _WEEK[lt.tm_wday],
               lt.tm_hour, lt.tm_min, lt.tm_sec))


@tool("get_weather", "查某个城市的实时天气", args='{"city": "城市名，如 深圳"}',
      net=True)
def _t_weather(args, ctx):
    city = _s((args or {}).get("city") or (args or {}).get("城市"), 30)
    if not city:
        return "没给城市名，查不了。"
    url = ("https://wttr.in/%s?format=%%l+%%C+%%t+%%h+%%w&lang=zh"
           % urllib.parse.quote(city))
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=NET_TIMEOUT) as r:
        txt = r.read().decode("utf-8", "replace").strip()
    txt = " ".join(txt.split())
    if not txt or "Unknown location" in txt or "Sorry" in txt:
        return f"查不到「{city}」的天气。"
    return txt[:RESULT_MAX]


@tool("look", "重新看一眼画面：认出了什么东西、有几个人、有没有在动",
      args="无参数")
def _t_look(args, ctx):
    """【阶段 18】"看"这件事的分界线。

    阶段 9 的 look 只能回答"有几个人、在不在动" —— 不是它偷懒，
    是那时候的视觉层**根本没看出来过别的东西**（Haar 级联只认正脸）。
    所以"帮我看看桌上有什么"这种问题，模型只能靠编。

    现在检测器会给一批带名字的框，这个工具第一次能真的答上来。
    它只说**它看到的**，看到什么说什么，没看到就说没看到 ——
    宁可答"没认出东西"，也不能让模型以为"没返回"等于"什么都没有"。
    """
    v = (ctx or {}).get("vision") or {}
    if not v:
        return "现在没有画面数据（摄像头可能没开）。"
    faces = int(v.get("faces", 0))
    motion = float(v.get("motion", 0.0))
    objs = list(v.get("objects") or [])
    backend = v.get("detector") or "none"

    parts = []
    # ---------- 1) 认出了什么 ----------
    if objs:
        try:
            from vision import summarize
            names = summarize(objs)           # 已经是"2 个杯子、1 个人"这种中文
        except ImportError:
            names = v.get("labels") or ""
        if names:
            parts.append("我看到 " + names)
    elif backend in ("yolo", "fake"):
        # 真在跑目标检测，但这一帧什么都没有 —— 这是**有信息量**的答案
        parts.append("没认出什么东西")
    elif backend == "haar":
        parts.append("只开了人脸检测（认不出杯子这类东西）")
    else:
        parts.append("现在没开物体识别")

    # ---------- 2) 有几个人 ----------
    # 判据是"**人是不是已经在上面那句话里说过了**"，而不是"用的哪个后端" ——
    # 第一版按后端判，结果 YOLO 模式下「1 个人」被说了两遍。
    # 只要 objects 里有人，就不再重复；objects 为空（老格式的 ctx、
    # 或者检测器关了）时，这一条仍然是唯一的信息来源，保持阶段 9 的老行为。
    person_said = any((o.get("name") in ("person", "face")) for o in objs)
    if not person_said:
        if faces <= 0:
            parts.append("画面里没有人")
        elif faces == 1:
            parts.append("画面里有 1 个人")
        else:
            parts.append(f"画面里有 {faces} 个人")

    # ---------- 3) 动没动 ----------
    parts.append("有人在动" if motion > 0.02 else "基本静止")
    return "，".join(parts) + f"（运动量 {motion * 100:.1f}%）"


@tool("remember", "把一件关于用户的事长期记住（跨会话不丢）",
      args='{"key": "简短标签，如 用户的名字", "value": "内容"}')
def _t_remember(args, ctx):
    mem = (ctx or {}).get("memory")
    if mem is None:
        return "现在没有开记忆功能，记不了。"
    a = args or {}
    key = _s(a.get("key") or a.get("标签") or a.get("名字"), 40)
    value = _s(a.get("value") or a.get("内容") or a.get("值"), 200)
    if not key or not value:
        return "记住失败：key 和 value 都得给。"
    if not mem.remember(key, value):
        return f"记住失败：{key} 或内容不合法。"
    return f"已记住：{key} = {value}"


@tool("recall", "回忆之前记住的事", args='{"query": "关键词，可留空"}')
def _t_recall(args, ctx):
    mem = (ctx or {}).get("memory")
    if mem is None:
        return "现在没有开记忆功能，想不起来。"
    q = _s((args or {}).get("query") or (args or {}).get("关键词"), 40)
    got = mem.recall(q, limit=6)
    if not got:
        return "什么都没记住，没有可以回忆的。"
    return "；".join(f"{k}：{v}" for k, v in got)


@tool("forget", "忘掉某条之前记住的事", args='{"key": "要忘掉的标签"}')
def _t_forget(args, ctx):
    mem = (ctx or {}).get("memory")
    if mem is None:
        return "现在没有开记忆功能。"
    key = _s((args or {}).get("key") or (args or {}).get("标签"), 40)
    if not key:
        return "忘掉失败：没给 key。"
    return f"已忘掉「{key}」。" if mem.forget(key) else f"没有「{key}」这条记录。"


@tool("stats", "查 MOSS 自己的运行统计（运行多久、听懂几句话、记忆里有多少东西）",
      args="无参数")
def _t_stats(args, ctx):
    c = ctx or {}
    up = float(c.get("uptime", 0.0))
    h, rem = divmod(int(up), 3600)
    m, sec = divmod(rem, 60)
    parts = [f"已连续运行 {h}小时{m}分{sec}秒" if h else f"已连续运行 {m}分{sec}秒"]
    parts.append(f"本次听懂 {int(c.get('heard', 0))} 句话")
    parts.append(f"当前状态 {c.get('mode') or '未知'}")
    mem = c.get("memory")
    if mem is not None:
        s = mem.summary()
        parts.append(f"记忆里存着 {s['facts']} 条事实、{s['turns']} 轮对话，"
                     f"这是第 {s['sessions']} 次启动")
    return "；".join(parts) + "。"


# ================================================================
#  执行器
# ================================================================
class ToolBox:
    """工具的"调度台"：给模型看清单，替模型执行调用。

    它本身不做任何判断 —— 调不调用、怎么调用，都是模型决定的。
    这里只负责：查名字、洗参数、兜异常、限长度、记统计。
    """

    def __init__(self, context=None, enabled=None, log=None):
        self.enabled = list(enabled) if enabled else list(REGISTRY.keys())
        self.ctx = dict(context or {})
        self.log = log
        self.stats = {"calls": 0, "ok": 0, "fail": 0}
        self.lock = threading.RLock()

    # ---------- 上下文 ----------
    def update(self, **kw):
        """每轮融合循环把最新快照塞进来（画面/听觉/状态/计数）。"""
        with self.lock:
            self.ctx.update(kw)

    def catalog(self):
        """生成给模型看的工具清单（中文，一行一个）。"""
        lines = []
        with _LOCK:
            for name in self.enabled:
                t = REGISTRY.get(name)
                if t:
                    lines.append(f"- {name}：{t['desc']}。参数：{t['args']}")
        return "\n".join(lines)

    def names(self):
        return list(self.enabled)

    # ---------- 执行 ----------
    def call(self, name, args=None):
        """执行一个工具。返回 (ok, 结果文本)。**绝不抛异常**。"""
        name = _s(name, 40).lower()
        with self.lock:
            self.stats["calls"] += 1

        with _LOCK:
            t = REGISTRY.get(name)
        if t is None:
            with self.lock:
                self.stats["fail"] += 1
            if self.log:
                self.log(f"[工具] 模型要调 {name!r}，没有这个工具；"
                         f"可用的是 {self.names()}")
            return False, f"没有 {name} 这个工具。"

        if name not in self.enabled:
            with self.lock:
                self.stats["fail"] += 1
            return False, f"工具 {name} 这次被关掉了。"

        if args is None:
            args = {}
        if not isinstance(args, dict):
            # 模型有时会传个字符串或数组进来，兜一下
            args = {"value": _s(args)}

        t0 = time.time()
        try:
            result = t["fn"](args, self.ctx)
            ok = True
            with _LOCK:
                t["calls"] += 1
                t["ms"] += int((time.time() - t0) * 1000)
        except Exception as e:
            ok, result = False, f"执行出错：{e}"
            with _LOCK:
                t["errors"] += 1

        result = _s(result, RESULT_MAX)
        dt = int((time.time() - t0) * 1000)
        with self.lock:
            self.stats["ok" if ok else "fail"] += 1
        if self.log:
            self.log(f"[工具] {name}({_s(args, 60)}) → {'ok' if ok else '失败'} "
                     f"{dt}ms  {result}")
        return ok, result

    def describe(self):
        with _LOCK:
            rows = []
            for name in self.enabled:
                t = REGISTRY.get(name)
                if t and t["calls"]:
                    rows.append(f"{name}×{t['calls']}")
        s = self.stats
        base = f"{len(self.enabled)} 个工具可用，调用 {s['calls']} 次（成功 {s['ok']}）"
        return base + ("：" + " ".join(rows) if rows else "")


# ================================================================
#  自检：python moss_tools.py
# ================================================================
def _selftest():
    import tempfile
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from memory import MemoryStore

    print("=" * 66)
    print("  MOSS 工具层自检")
    print("=" * 66)

    d = tempfile.mkdtemp(prefix="moss_tool_")
    mem = MemoryStore(os.path.join(d, "m.json"), quiet=True)
    ctx = {
        # 【阶段 18】vision 里现在有 objects/labels/detector 三样。
        # 自检必须**带上它们** —— 否则 look 工具新写的那几条分支一次都跑不到，
        # 而"没被测到的分支 = 不存在"这个教训阶段 17 已经吃过一遍了。
        "vision": {"faces": 1, "motion": 0.031, "detector": "yolo",
                   "objects": [
                       {"name": "person", "cls": 0, "conf": 0.91,
                        "box": (10, 10, 50, 80)},
                       {"name": "cup", "cls": 41, "conf": 0.80,
                        "box": (60, 60, 20, 20)},
                       {"name": "cup", "cls": 41, "conf": 0.72,
                        "box": (90, 60, 20, 20)},
                   ],
                   "labels": "person x1, cup x2"},
        "audio": {"dbfs": -32.0, "listening": True},
        "mode": "LISTEN",
        "memory": mem,
        "uptime": 3725.0,
        "heard": 4,
    }
    box = ToolBox(context=ctx, log=lambda s: print("      " + s))

    print("\n[1/4] 清单（这段就是塞给模型看的东西）")
    print(box.catalog())

    print("\n[2/4] 逐个调用")
    cases = [
        ("get_time", {}),
        ("look", {}),
        ("stats", {}),
        ("remember", {"key": "用户的名字", "value": "张三"}),
        ("remember", {"key": "用户的喜好", "value": "喜欢安静"}),
        ("recall", {"query": "名字"}),
        ("recall", {}),
        ("forget", {"key": "用户的喜好"}),
    ]
    ok_n = 0
    for name, args in cases:
        ok, res = box.call(name, args)
        ok_n += 1 if ok else 0
        if name == "look":
            print("      look 看到了：%s" % res)
            # 人是物体清单里的一员，不该被再说一遍（第一版就是按后端判的，
            # 结果 YOLO 模式下「1 个人」说了两遍）
            assert res.count("1 个人") == 1, "「人」被说重复了：%s" % res
            assert "2 个杯子" in res, "杯子没被讲出来：%s" % res
    assert ok_n == len(cases), "本地工具不该失败"
    assert mem.all_facts().get("用户的名字", {}).get("value") == "张三"
    assert "用户的喜好" not in mem.all_facts()
    print("      ✓ %d/%d 个本地工具全部成功，记忆读写也正确" % (ok_n, len(cases)))

    print("\n[3/4] look 的几种画面状态（阶段 18 新加的分支）")
    # 这是 look 工具最容易被写歪的地方：**"没认出东西"和"没开识别"是两回事**，
    # 混在一起会让大模型以为"画面里什么都没有"，然后开始编。
    box.update(vision={"faces": 0, "motion": 0.0, "detector": "yolo",
                       "objects": [], "labels": ""})
    _ok, r = box.call("look", {})
    print("      YOLO 在跑但没东西 → %s" % r)
    assert "没认出" in r, "跑着 YOLO 却什么都没看见时，应该说「没认出」，而不是沉默"

    box.update(vision={"faces": 1, "motion": 0.3, "detector": "haar",
                       "objects": [{"name": "face", "cls": -1, "conf": 1.0,
                                    "box": (1, 2, 3, 4)}],
                       "labels": "face x1"})
    _ok, r = box.call("look", {})
    print("      只有 Haar 的人脸 → %s" % r)
    assert "1 张人脸" in r, "Haar 模式下人脸的中文名/量词不对：%s" % r

    box.update(vision={"faces": 0, "motion": 0.0, "detector": "none",
                       "objects": [], "labels": ""})
    _ok, r = box.call("look", {})
    print("      根本没开识别 → %s" % r)
    assert "没开物体识别" in r, "没开识别时要说清楚，不能和「没认出」混为一谈"

    print("\n[4/4] 异常路径（模型乱来也不能崩）")
    for name, args in [("不存在的工具", {}),
                       ("get_time", "我是个字符串不是对象"),
                       ("get_weather", {"city": "火星基地"}),
                       ("remember", {"key": "只有key"})]:
        ok, res = box.call(name, args)
        print("      %-16s → ok=%-5s %s" % (name, ok, res[:70]))
    assert box.call("没有这个工具", {})[0] is False
    print("      ✓ 未知工具/畸形参数都稳妥返回，没有异常逃逸")

    print("\n统计：%s" % box.describe())
    print("\n[OK] 工具层自检通过。"
          "（get_weather 需要联网，失败属正常，见上面那行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
