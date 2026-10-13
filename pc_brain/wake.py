#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOSS 唤醒门  ·  流浪地球 550W 复刻项目 · 阶段 15
============================================================
到阶段 14，MOSS 已经"什么都听得见"了 —— 这正是问题。

麦克风常开，任何人说任何一句话，它都会老老实实走一遍：
切段 → 识别 → 调大模型 → 可能还动手。你说"这句代码有 bug"，
它可能真的去查一遍天气；旁边两个人聊天，它插一句"我看见你了"。

**唤醒词不是锦上添花，它是一条省钱和讲礼貌的分界线。**

------------------------------------------------------------
设计取舍：为什么是"文本后置门"而不是声学唤醒词
------------------------------------------------------------
市面上的做法是声学唤醒词（Porcupine / openWakeWord / snowboy）：
一直在本地跑个小模型，听到"你好小智"才唤醒。它当然更省 ——
连识别都不用跑。但对这个项目有三个现实问题：

1. **要新依赖 + 常常要 AccessKey**（Porcupine 联网激活），
   而本项目一路坚持"零新依赖、离线可验证"。
2. **没有麦克风就完全验证不了** —— 而我们的整个策略就是
   "先在没有硬件的情况下把逻辑跑对"。
3. 真正费钱的那一步是 **大模型调用**，不是识别。本地 whisper 跑一句
   才 0.3~1.5 秒、零成本；一次 LLM 请求才是真金白银。
   门开在识别之后，已经拦掉了 99% 的浪费。

所以这里做的是**文本后置门**：识别出文字之后、送语言层之前，判一下。
零新依赖、纯逻辑、可单元测试、可以离线跑。

将来要在真机上省到底，就在这个类前面再串一个声学唤醒词即可 ——
`WakeGate` 的接口（feed → hit/ignore）不用改。

------------------------------------------------------------
三条规则（按优先级）
------------------------------------------------------------
1. **句首命中**：`MOSS，现在几点了` → 唤醒，并**剥掉**称呼，
   只把「现在几点了」交给语言层。人就是这么说话的。
2. **句中命中**：`屏幕上写 MOSS` → 也算唤醒，但**保留整句**。
   因为用户确实在让它干活，把"MOSS"抠掉反而改变语义。
3. **会话窗口**：唤醒后 20 秒内随便说，不用反复喊名字。
   每说一句窗口**滑动续期** —— 真正在对话时它一直醒着，停下来才睡。

`--wake-strict` 可以关掉规则 2（只认句首），适合"不希望闲聊时被误唤醒"的场景。

------------------------------------------------------------
别名表：为什么不能只匹配 "moss"
------------------------------------------------------------
真 whisper 在中文模式下几乎不会老实输出 "MOSS"。它可能给
"莫斯 / 摩斯 / 摩丝 / 莫司"，取决于音色和口音。而假 ASR 服务端
又只会输出你脚本里写的原文。两边都得能过 —— 所以用别名表。

反过来，**不能**为了宽容就做模糊音匹配（"末斯""抹丝"）：
那会把"摩丝发胶"这种正常话也拉进来。宁可漏一两个口音，
不可满街答应。

------------------------------------------------------------
自检：python wake.py     # 不需要麦克风，纯文本
"""

import collections
import re
import time

# 默认唤醒词。顺序无所谓 —— 类内部会按长度倒序排，
# 免得 "moss" 抢先匹配掉 "小moss"。
# 注意这里**没有** "摩丝"。它看着像无害的别名，实际上撞了常用词
# （"摩丝发胶"）—— 而且句首命中会剥掉称呼，于是"摩丝发胶在哪买的"
# 变成了"发胶在哪买的"，语义被吃掉。别名越宽容这种事故越多，
# 所以只留日常几乎不会单独出现的音译词。
DEFAULT_WORDS = (
    "boss",
    "博斯", "波斯", "波丝", "博丝", "伯斯", "帛斯",
    "小boss", "小博斯", "小波斯",
)

# 称呼前缀。人喊人不会光秃秃喊名字："你好 MOSS""喂 MOSS"。
# 注意这里**只做前缀**，所以 "你好，几点了" 不会因为含"你好"就被唤醒 ——
# 它还得真的出现唤醒词。
_HONORIFIC = r"(?:你好|您好|嗨|嘿|喂|诶|哎|那个|我说)"

# 归一化：只留字母/数字/下划线/汉字。中英标点、空白统统去掉。
# 用于"句中包含"判断 —— "MOSS，" 和 "moss" 得判成同一个东西。
_NORM_RE = re.compile(r"[\W_]+", re.UNICODE)


def _norm(s):
    return _NORM_RE.sub("", (s or "").lower())


def _lstrip_punct(s):
    """剥掉唤醒词之后残留的标点空白：'，几点了' → '几点了'。"""
    return (s or "").lstrip(" \t　，,。.!！?？、;；:：·…—～~")


# feed() 的返回值。四个字段各有用途：
#   hit    —— 要不要继续往下走（送语言层）
#   text   —— 真正要送给语言层的那句话（可能已剥掉称呼）
#   reason —— 为什么这么判（日志与排查用）
#   word   —— 命中的是哪个唤醒词（可能是别名）
#   bare   —— 用户只喊了名字、没说要干什么（该"哎"一声，不该当问题答）
Decision = collections.namedtuple("Decision", "hit text reason word bare")


class WakeGate:
    """把"听到的每一句"筛成"该回应的那一句"。

    线程安全：主循环单线程调用，不需要锁。唯一的可变状态是
    `_awake_until`（会话窗口的截止时间戳）。
    """

    def __init__(self, words=None, window=20.0, enabled=True,
                 strict=False, log=None):
        self.words = tuple(words) if words else DEFAULT_WORDS
        self.window = max(0.0, float(window))
        self.enabled = bool(enabled)
        self.strict = bool(strict)
        self.log = log
        self._awake_until = 0.0
        self.stats = {"woken": 0, "bare": 0, "window": 0,
                      "ignored": 0, "passed": 0}
        # 长词排前面：Python 的 "|" 取第一个能匹配的，不是最长的。
        # 若 "moss" 排在 "小moss" 前面，"小moss你好" 会在位置 0 上
        # 逐个试（moss 对不上"小"）→ 整条不匹配 → 漏唤醒。
        words_sorted = sorted(self.words, key=len, reverse=True)
        alt = "|".join(re.escape(w) for w in words_sorted)
        self._head_re = re.compile(
            r"^\s*(?:" + _HONORIFIC + r"[\s，,、]*)*(" + alt + r")", re.I)

    # ---------- 状态 ----------
    @property
    def awake(self):
        return time.time() < self._awake_until

    def remaining(self, now=None):
        now = time.time() if now is None else now
        return max(0.0, self._awake_until - now)

    @property
    def first_word(self):
        """给 OLED 用的短提示（'SAY BOSS'）。默认词表里第一个是 boss。"""
        return self.words[0] if self.words else "boss"

    def arm(self, now=None):
        """手动唤醒（比如拍手、或者用户在窗口里点了按钮）。"""
        now = time.time() if now is None else now
        self._awake_until = now + self.window

    def sleep(self):
        """手动让它睡下（--wake-test 之类会用到）。"""
        self._awake_until = 0.0

    # ---------- 主入口 ----------
    def feed(self, text, now=None):
        """判一句话。永远返回 Decision，不抛异常。"""
        now = time.time() if now is None else now
        raw = (text or "").strip()

        if not self.enabled:
            self.stats["passed"] += 1
            return Decision(True, raw, "disabled", "", False)
        if not raw:
            # 空文本不该被当成"忽略"计入统计 —— 那是分段器/引擎的毛刺，
            # 记成忽略会让"忽略 N 句"这个数字变得不可信。
            return Decision(False, "", "empty", "", False)

        # ---------- 规则 1：句首命中（最自然，顺便把称呼剥掉）----------
        m = self._head_re.match(raw)
        if m:
            word = m.group(1)
            rest = _lstrip_punct(raw[m.end():])
            self.arm(now)
            self.stats["woken"] += 1
            if not rest:
                # 只喊了名字。这不是提问，是"在吗"。
                self.stats["bare"] += 1
                self._say(f"[唤醒] 听到「{word}」（只有称呼）→ 已被叫醒")
                return Decision(True, "", "wake-bare", word, True)
            self._say(f"[唤醒] 「{word}」开头 → 剥掉称呼，只送：{rest}")
            return Decision(True, rest, "wake-head", word, False)

        # ---------- 规则 2：句中命中（保留了整句）----------
        if not self.strict:
            core = _norm(raw)
            for w in self.words:
                nw = _norm(w)
                if nw and nw in core:
                    self.arm(now)
                    self.stats["woken"] += 1
                    self._say(f"[唤醒] 句中出现「{w}」→ 整句交给语言层")
                    return Decision(True, raw, "wake-inside", w, False)

        # ---------- 规则 3：会话窗口内 ----------
        if now < self._awake_until:
            self.stats["window"] += 1
            # 滑动续期：真正在对话时它一直醒着，停下来才睡。
            # 不续期的话，一次超过 20 秒的长回答会被自己的睡眠打断。
            self._awake_until = now + self.window
            return Decision(True, raw, "window", "", False)

        # ---------- 没叫它 ----------
        self.stats["ignored"] += 1
        return Decision(False, raw, "asleep", "", False)

    # ---------- 输出 ----------
    def _say(self, msg):
        if self.log:
            self.log(msg)

    def describe(self):
        if not self.enabled:
            return "关闭（任何人说话都会回应）"
        mode = "只认句首" if self.strict else "句首+句中"
        return (f"「{self.first_word}」（{mode}，窗口 {self.window:g}s，"
                f"别名 {len(self.words)} 个）")

    def stats_line(self):
        s = self.stats
        if not self.enabled:
            return f"唤醒 关闭 · 放行 {s['passed']} 句"
        return (f"唤醒 {s['woken']} 次（只喊名字 {s['bare']}）· "
                f"窗口内免唤醒 {s['window']} 句 · 忽略 {s['ignored']} 句")


# ================================================================
#  自检：python wake.py
# ================================================================
def _selftest():
    print("=" * 66)
    print("  MOSS 唤醒门自检")
    print("=" * 66)

    # 用一个假的时钟，避免测试依赖真实时间流逝
    class Clock:
        def __init__(self):
            self.t = 1000.0

        def __call__(self):
            return self.t

        def tick(self, dt):
            self.t += dt

    clk = Clock()
    gate = WakeGate(window=20.0, log=lambda s: print("      " + s))

    print("\n[1/5] 句首命中：剥掉称呼，只送指令")
    cases = [
        ("BOSS，现在几点了", "现在几点了"),
        ("boss 把灯变成红色", "把灯变成红色"),
        ("你好 BOSS，深圳的天气怎么样", "深圳的天气怎么样"),
        ("嗨，Boss！转两圈", "转两圈"),
        ("BOSS。你是谁", "你是谁"),
    ]
    for raw, want in cases:
        gate.sleep()
        d = gate.feed(raw, now=clk())
        assert d.hit, f"{raw!r} 该被唤醒"
        assert d.reason == "wake-head", f"{raw!r} → {d.reason}"
        assert d.text == want, f"{raw!r}\n  期望 {want!r}\n  实际 {d.text!r}"
        print(f"      ✓ {raw!r} → {d.text!r}")

    print("\n[2/5] 只喊名字：要回应，但别当问题")
    for raw in ("BOSS", "boss!", "你好博斯", "小BOSS"):
        gate.sleep()
        d = gate.feed(raw, now=clk())
        assert d.hit and d.bare, f"{raw!r} 该是 bare 唤醒"
        assert d.text == "", f"{raw!r} 不该有台词：{d.text!r}"
        print(f"      ✓ {raw!r} → bare（只喊名字）")

    print("\n[3/5] 别名：whisper 听不准也得认")
    for raw in ("博斯，几点了", "波斯在吗", "博丝，开灯", "伯斯 你好"):
        gate.sleep()
        d = gate.feed(raw, now=clk())
        assert d.hit, f"{raw!r} 该被唤醒（别名）"
        print(f"      ✓ {raw!r} → {d.text!r}")

    print("\n[4/5] 句中命中：保留整句（这是阶段 14 的老用例）")
    gate.sleep()
    d = gate.feed("屏幕上写 BOSS", now=clk())
    assert d.hit and d.reason == "wake-inside", d
    assert d.text == "屏幕上写 BOSS", d.text     # 不剥！剥了就变"屏幕上写"了
    print(f"      ✓ 「屏幕上写 BOSS」→ 整句保留（抠掉 BOSS 语义就变了）")

    gate = WakeGate(window=20.0, strict=True, log=None)
    gate.sleep()
    d = gate.feed("屏幕上写 BOSS", now=clk())
    assert not d.hit, "strict 模式下句中命中不该唤醒"
    print("      ✓ strict 模式：句中命中不算（适合怕误唤醒的场景）")

    print("\n[5/5] 会话窗口：醒着的时候不用反复喊名字")
    gate = WakeGate(window=20.0, log=lambda s: print("      " + s))
    gate.sleep()
    d = gate.feed("BOSS", now=clk())
    assert d.hit and d.bare

    # 窗口内，不带唤醒词也放行
    for raw in ("现在几点了", "那明天呢", "深圳天气怎么样"):
        clk.tick(3.0)
        d = gate.feed(raw, now=clk())
        assert d.hit and d.reason == "window", f"{raw!r} → {d.reason}"
        print(f"      ✓ 窗口内 {raw!r} → 放行（免唤醒）")

    # 滑动续期：说了 3 句、累计 9 秒，仍然醒着。
    # 注意这里用 remaining(now=clk()) 而不是 gate.awake ——
    # awake 读的是真实 time.time()，而这个测试用的是假时钟，两者不在一个纪元上。
    assert gate.remaining(now=clk()) > 0, "连续说话时窗口该被续期"
    print("      ✓ 每说一句窗口滑动续期（连续对话不会中途睡着）")

    # 静置超过窗口 → 睡着
    clk.tick(21.0)
    d = gate.feed("这个变量没定义", now=clk())
    assert not d.hit and d.reason == "asleep", d
    print("      ✓ 静置 21s → 睡着，无关的闲聊被拦下")

    # 睡着后，含唤醒词的 AI 又把它叫醒
    d = gate.feed("BOSS，把灯关掉", now=clk())
    assert d.hit and d.text == "把灯关掉", d
    print("      ✓ 再喊一次名字 → 立刻重新唤醒")

    print("\n[6/6] 边界与反例：宁可漏，不可满街答应")
    gate2 = WakeGate(window=20.0, log=None)
    bad = [
        ("你好，现在几点了", "打招呼不等于喊名字"),      # 最关键的反例
        ("把灯变成红色", "不含唤醒词（阶段 14 的老用例，现在得先叫名字）"),
        ("谢谢大家", "whisper 的经典幻觉"),
        ("这个 bug 在哪一行", "旁人的闲聊"),
    ]
    for raw, why in bad:
        gate2.sleep()
        d = gate2.feed(raw, now=clk())
        assert not d.hit, f"{raw!r}（{why}）不该被唤醒，但命中了：{d.reason}"
        print(f"      ✓ 拦下 {raw!r}  ← {why}")

    d = gate2.feed("", now=clk())
    assert not d.hit and d.reason == "empty"
    # 空文本必须走 empty 分支，不能计进 ignored ——
    # 否则"忽略 N 句"里会混进一堆引擎毛刺，这个数字就不敢信了。
    assert gate2.stats["ignored"] == len(bad), gate2.stats
    print("      ✓ 空文本走 empty 分支，不污染「忽略」统计")

    # 已知的假阳性。别名表越宽容，越容易撞上同音词 ——
    # 这是**刻意接受**的取舍：宁可偶尔多回一句，也不要用户喊三遍没反应。
    # 「博斯」撞「博士」（博士毕业 / 博士学历），句首命中时称呼会被剥掉。
    # 但代价可接受：日常单独把「博士」当称呼喊出来的概率不高。
    gate2.sleep()
    d = gate2.feed("博斯，帮我写论文", now=clk())
    assert d.hit and d.reason == "wake-head" and d.text == "帮我写论文", d
    print(f"      ✓ 已知假阳性：「博斯」撞「博士」→ 剥离成 {d.text!r}（句首剥离的代价）")

    # 「波斯」撞「波斯猫/波斯地毯」，同样是可接受的误唤醒。
    gate2.sleep()
    d = gate2.feed("波斯猫多少钱一只", now=clk())
    assert d.hit, "「波斯」已加入别名表，这里应当是（可接受的）误唤醒"
    print("      ✓ 已知假阳性：「波斯猫」→ 误唤醒（加「波斯」别名、提高真实唤醒率的代价）")

    # 关闭开关：一切照旧通过
    off = WakeGate(enabled=False, log=None)
    d = off.feed("随便说点什么", now=clk())
    assert d.hit and d.reason == "disabled"
    print("      ✓ --no-wake 时全部放行（回到阶段 14 的行为）")

    print("\n统计示例：%s" % gate.stats_line())
    print("\n[OK] 唤醒门自检通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
