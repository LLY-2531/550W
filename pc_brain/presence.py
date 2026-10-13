#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOSS 变化感知层（物体在场跟踪）  ·  流浪地球 550W 复刻项目 · 阶段 19
============================================================
阶段 18 给了 MOSS 一双能认东西的眼睛，但它只会**一问一答**：
你问"画面里有什么"，它答"1 个杯子、1 个键盘"。
你不问，它就永远不吭声 —— 桌上多了个杯子它也不知道。

这一层要解决的就是"**知道画面变了**"：
    桌子本来空着，你放了个杯子 → "我注意到画面里多了 1 个杯子。"
    杯子被你拿走               → "我注意到画面里少了 1 个杯子。"

------------------------------------------------------------
这一层的真正难题：检测是**有噪声的**
------------------------------------------------------------
人脸检测（阶段 9 的 Haar）和人脸的出现/消失是"干净"的：
一张脸在不在，逐帧看基本一致。所以那些代码可以放心地写成
"上一帧没有、这一帧有 → 有人来了"。

**物体检测不是这样。** 同一个杯子，yolov8n 可能给：
    第 1 帧 0.91 → 第 2 帧 没有（被手挡住一点）→ 第 3 帧 0.89 → 第 4 帧 没有
框会闪，个数会在 1 和 2 之间跳（两个重叠框、或杯子被挡了一半）。

如果直接按"这一帧有没有 cup"判定，结果就是：
**杯子每几帧"消失"一次、又"出现"一次** —— 于是 MOSS 会像个坏掉的
报警器一样反复念叨"多了个杯子、杯子没了、多了个杯子……"

所以这一层的大部分代码是**滤波器**，不是识别器。
和阶段 16 的 behave.py 是同一个道理：**难点不在"会不会动"，而在"什么时候别动"。**
这里是：难点不在"能不能看出变化"，而在"**别把噪声当成变化**"。

------------------------------------------------------------
三道滤波，各有各的理由
------------------------------------------------------------
**① 迟滞（hysteresis）：进入慢、离开更慢**

每个类维护一个分数，检出 +1、漏检 -1，封在 0..SCORE_MAX：

    分数 ≥ ENTER_HITS  → 算"在场"
    分数回到 0          → 算"不在场"

于是"进入"要连续攒够 ENTER_HITS 次检出，"离开"要连续掉够 LEAVE_HITS 次漏检。

⚠️ **确认"在场"的那一刻，分数要直接顶到上限**（见 `_step` 里的注释）。
少了这一步，刚够 3 分就被漏检往回啃，漏 3 次就掉出去了 ——
"离开要 7 次"**从来没生效过**，两个门槛互相削弱，等于只剩一个门槛。
这个 bug 是自检第一次跑就抓到的，所以那条断言留着不许删。

**两个方向刻意不对称，而且离开比进入钝得多。** 理由不是"性能"，
而是**这两类错误付出的代价不一样**：

    晚一点发现"东西没了"  → 只是反应慢，0.7 秒后补上就完了
    把"暂时没看见"说成"没了" → 那是**编造事实**

这个项目里 MOSS 说的话是要被当成真的听的（它还会调工具、记事情）。
"我注意到杯子少了"这句话如果是因为你手挡了一下就说出来的，
那它就失去了可信度。**宁可迟钝，不许撒谎。**

（抖动不会把进度清零 —— 一帧漏检只是退一格，所以"检出2帧/漏1帧/检出2帧"
 依然能攒到确认。这一点很重要：真实检测本来就是抖的，
 如果用"连续 N 次"那种一断就归零的写法，真的杯子可能永远攒不够。）

**② 个数变化也要确认**

"1 个杯子变成 2 个"同样可能只是框在跳。所以个数变化要连续
COUNT_HITS 次看到同一个新数字才认账。

**③ 暖机：先建基线，再谈变化**

`ready` 之前只喂分数、不报变化。因为：
- 一开机画面里本来就有桌子、键盘、显示器。把它们当"新增"，
  MOSS 一启动就报一屋子东西 —— 那是噪音，不是观察。
- 而且检测器刚跑的头几帧本身就没稳定，那时候的计数是噪声。

------------------------------------------------------------
什么**不**报：一张"不该说的东西"表
------------------------------------------------------------
COCO 80 类不是每一样都值得开口：

- **人（person / face）**：已经有 `arrive` / `leave` 专门处理
  （阶段 16）。这里再报一次就是**同一件事说两遍** ——
  阶段 18 在 `look` 工具上已经踩过一模一样的坑（YOLO 模式下
  "1 个人"被说了两遍），这里不能再犯。
- **家具 / 固定物（椅子、沙发、桌子、电视、盆栽……）**：它们的位置
  本来就不动，报"沙发来了"是纯噪音。而且它们恰恰是 yolov8n
  误报最多的类（阶段 18 实测：`dining table 0.39`、`book 0.44`）——
  越是本来就该在那儿的东西，越容易被"认出来"，越不该报。

------------------------------------------------------------
职责边界：这一层**不**决定"要不要说话"
------------------------------------------------------------
它只回答一个问题：**"现在画面里稳定地在场的东西有哪几个、各几个？"**

"该不该现在开口"是行为层（behave.py）的事 —— 那里有节流、冷却、权限分级。
为什么这么切：`presence.py` 是纯滤波，没有时间概念、没有权限概念，
所以它可以喂任意造出来的序列（见自检），一秒跑完一万帧。
一旦把"节流"混进来，就没法这么测了。

自检：python presence.py    # 不需要摄像头、不需要模型、不需要网络
"""

import collections

# ================================================================
#  参数：三个门槛，每个都有具体理由
# ================================================================
# 迟滞分数：检出 +1、漏检 -1，封在 0..SCORE_MAX。
# 在场 ⇔ 分数 >= ENTER_HITS。
#
# 进入要净 +ENTER_HITS 次检出；离开要净 -(SCORE_MAX - ENTER_HITS + 1) 次漏检。
# 把 SCORE_MAX 定成 ENTER_HITS + LEAVE_HITS - 1，两个数字就正好是
# "连续 ENTER_HITS 次"和"连续 LEAVE_HITS 次" —— 一眼能对上名字。
# （那个 -1 是因为判定用的是 `>=` 而不是 `>`，不是笔误。）
ENTER_HITS = 3
LEAVE_HITS = 7
SCORE_MAX = ENTER_HITS + LEAVE_HITS - 1        # = 9

# 个数变化要连续几次看到同一个新数字才认账。
# 比 ENTER_HITS 略钝：个数是二阶信息，比"在不在"更容易被噪声影响。
COUNT_HITS = 3

# 暖机：头几次检测只养状态，不产生变化。单位是**检测次数**，不是帧 ——
# 目标检测每 detect_every 帧才跑一次，用帧做单位的话，
# 一改 --vision-detect-every 就等于偷偷改了这里的语义。
WARMUP_DETECTS = 10

# 不报的类。见文件开头"什么不该说"那一节。
NEVER_ANNOUNCE = (
    # ① 人：已经有 arrive / leave，报了就是说两遍
    "person", "face",
    # ② 家具 / 固定物：位置本来就不动，且是误报重灾区
    "chair", "couch", "bed", "dining table", "toilet", "bench",
    "potted plant", "tv", "refrigerator", "oven", "microwave", "sink",
    "clock", "vase", "toaster", "hair drier",
    # ③ 交通 / 户外大件：室内出现基本是误报
    "car", "bus", "truck", "motorcycle", "bicycle", "airplane", "boat",
    "train", "traffic light", "fire hydrant", "stop sign", "parking meter",
)

# 中文类名和量词表在 vision.py。**不在这里抄第二份** ——
# 阶段 18 因为"同一个排序写了两份实现"吃过一次亏（日志和 OLED 各说各话），
# 那次教训是：数量词表这种"两处都要用"的东西只该有一个源头。
try:
    from vision import zh_name, summarize as _vision_summarize
    HAS_VISION = True
except ImportError:                    # 没有视觉层时也能自检
    HAS_VISION = False

    def zh_name(name):
        return name

    def _vision_summarize(objs, max_n=3):
        tally = collections.Counter(o.get("name") or "?" for o in objs)
        rows = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))[:max_n]
        return "、".join("%d %s" % (c, n) for n, c in rows)


# ================================================================
#  1. 哪些类值得说
# ================================================================
def announceable(name):
    """这个类名值不值得报"多了/少了"。

    表里没有的一律**算值得报** —— 这是刻意的：将来换一个自己微调的模型，
    它的类名不在 COCO 里，也应该默认能用，而不是被一张写死的表挡在门外。
    """
    return (name or "").strip().lower() not in NEVER_ANNOUNCE


# ================================================================
#  2. 在场跟踪器
# ================================================================
class PresenceTracker:
    """把"一串有噪声的逐帧检测结果"变成"稳定地在场的物体计数"。

    用法（**每次检测之后**调一次，不是每帧）：

        t = PresenceTracker()
        t.update(objs)          # objs 就是 vision.Detector.detect() 的返回值
        t.ready                 # 暖机完了吗
        t.visible()             # {"cup": 1, "laptop": 1}（已滤掉不该报的类）

    它**不**产出事件、**不**知道时间、**不**知道权限。
    变化由调用方自己比（见 behave.BehaviorEngine）——
    这样这一层才能拿任意序列离线喂（自检就是这么干的）。
    """

    def __init__(self, enter_hits=ENTER_HITS, leave_hits=LEAVE_HITS,
                 count_hits=COUNT_HITS, warmup=WARMUP_DETECTS):
        self.enter_hits = max(1, int(enter_hits))
        self.leave_hits = max(1, int(leave_hits))
        self.score_max = self.enter_hits + self.leave_hits - 1
        self.count_hits = max(1, int(count_hits))
        self.warmup = max(0, int(warmup))

        self._score = {}        # 类名 → 迟滞分数 0..score_max
        self._count = {}        # 类名 → 已确认的稳定计数（不在场时不存在）
        self._cand = {}         # 类名 → 个数变化的候选值
        self._cand_hits = {}    # 类名 → 候选值连续出现了几次
        self._touched = set()   # 分数曾经 >0 的类（用来统计"闪了一下"）

        self.n_updates = 0
        self.stats = collections.Counter()

    # ---------- 对外 ----------
    @property
    def ready(self):
        """暖机完成没。没完成时调用方**只该建基线，不该报变化**。"""
        return self.n_updates >= self.warmup

    def visible(self):
        """稳定地在场、且值得报的计数：`{"cup": 1}`。

        返回的是**副本** —— 调用方拿去算 diff 的时候不能改到内部状态。
        """
        return {n: c for n, c in self._count.items()
                if c > 0 and announceable(n)}

    def counts(self):
        """稳定计数，**不过滤**（调试/日志用）。"""
        return dict(self._count)

    def brief(self):
        """一行短标签：`cup x1, laptop x1`。数量多的在前。"""
        items = self.visible()
        if not items:
            return ""
        if HAS_VISION:
            # 刻意复用 vision.count_by_name 来决定"同数量时谁在前"。
            # 阶段 18 的教训：同一个排序写两份实现，日志和 OLED 就会各说各话。
            from vision import count_by_name
            rows = count_by_name([{"name": n} for n in items])
            return ", ".join("%s x%d" % (n, c) for n, _zh, c in rows)
        rows = sorted(items.items(), key=lambda kv: (-kv[1], kv[0]))
        return ", ".join("%s x%d" % (n, c) for n, c in rows)

    def state(self):
        """每个类的内部状态。给自检/排错看，正式逻辑不用。"""
        return {n: {"score": self._score.get(n, 0),
                    "count": self._count.get(n, 0),
                    "present": n in self._count}
                for n in set(self._score) | set(self._count)}

    def stats_line(self):
        s = self.stats
        parts = [f"检测 {self.n_updates} 次"]
        if s["appear"]:
            parts.append(f"新出现 {s['appear']}")
        if s["vanish"]:
            parts.append(f"消失 {s['vanish']}")
        if s["blip"]:
            parts.append(f"闪一下没够着 {s['blip']}")
        return " · ".join(parts)

    # ---------- 主入口 ----------
    def update(self, objs):
        """**每次检测之后**调一次。

        ⚠️ 不要每帧调。目标检测每 `detect_every` 帧才跑一次，
        每帧调一次会让迟滞计数涨快 N 倍 —— 那等于偷偷把
        "连续 3 次检出"变成"连续 1 次"，而 `--vision-detect-every`
        一改，行为就跟着变，这类 bug 查起来非常费劲。
        """
        self.n_updates += 1

        now = collections.Counter()
        for o in objs or []:
            name = (o.get("name") or "").strip().lower()
            if name:
                now[name] += 1

        # 同时遍历"这一帧出现的"和"内部已经有状态的" ——
        # 后者即使这一帧完全没检出也要走一遍，因为它的分数要往下掉。
        for name in set(now) | set(self._score):
            self._step(name, now.get(name, 0))

    def _step(self, name, n):
        """更新一个类的分数与计数。n = 这一帧检出了几个。"""
        score = self._score.get(name, 0)
        was = score >= self.enter_hits

        if n > 0:
            score = min(self.score_max, score + 1)
            self._touched.add(name)
        else:
            score = max(0, score - 1)

        # 掉回 0 且从没够着过 → "闪了一下"。保留这个数字，
        # 是因为它是**检测质量的健康指标**：它很大就说明检测器在抖，
        # 该去调 --vision-conf 或者看看光照，而不是来改这里的门槛。
        if score == 0 and name in self._touched and not was:
            self._touched.discard(name)
            self.stats["blip"] += 1

        self._score[name] = score
        present = score >= self.enter_hits

        if present and not was:
            # ---------- 进入 ----------
            # 计数直接采信，不再走"个数确认" —— 已经攒够 ENTER_HITS 次检出了，
            # 再要求它连续 COUNT_HITS 次一致，等于把进入门槛乘起来。
            self._count[name] = n
            self._cand[name] = n
            self._cand_hits[name] = 0
            # ⚠️ 暖机期间**不记** appear/vanish。
            # 这两个数是给收尾统计看的（"它今天报了几次『来了』"），
            # 而暖机那批本来就不报（调用方只建基线）。
            # 一开始没加这个判断，实测收尾打出"新出现 5 · 消失 4"，
            # 而已知真正开口只有 2 次 —— 读日志的人会以为功能坏了。
            if self.ready:
                self.stats["appear"] += 1
            # ⚠️ **分数直接顶到上限**，这一步是必须的，自检第一次跑就抓到了缺它的后果：
            # 不顶的话，刚够 3 分就被漏检往回啃，漏 3 次就掉出去了 ——
            # 于是"离开要 7 次"从来没生效过，**两个门槛互相削弱，等于只有一个门槛**。
            # 道理：这里的分数表示的是"**距离改判还有多远**"，不是"证据有多强"。
            # 确认在场的那一刻，距离"改判不在场"就该是最远的那一步。
            self._score[name] = self.score_max
        elif present:
            # ---------- 在场：个数要确认 ----------
            # ⚠️ 漏检帧（n==0）**不提供任何个数信息**，必须跳过。
            # 自检第一次跑就抓到了这个：不跳过的话，在场期间漏一帧就会被
            # 当成"个数变成 0"，连续漏 COUNT_HITS 帧就把计数抹成 0 ——
            # 表现成"杯子闪了一下就不见了"。在场与否由**分数**管，
            # 个数只从"确实看见了的帧"里取，两件事不要互相越权。
            if n <= 0:
                pass
            elif n == self._count[name]:
                self._cand[name] = n
                self._cand_hits[name] = 0
            else:
                if n == self._cand.get(name):
                    self._cand_hits[name] = self._cand_hits.get(name, 0) + 1
                else:
                    self._cand[name] = n
                    self._cand_hits[name] = 1
                if self._cand_hits[name] >= self.count_hits:
                    self._count[name] = self._cand[name]
                    self._cand_hits[name] = 0
        elif was:
            # ---------- 离开 ----------
            self._count.pop(name, None)
            self._cand.pop(name, None)
            self._cand_hits.pop(name, None)
            self._touched.discard(name)
            if self.ready:            # 暖机期间不算，理由同上面 appear
                self.stats["vanish"] += 1


# ================================================================
#  3. 把变化说成人话
# ================================================================
def describe(changes, totals=None, max_n=3):
    """把 `{类名: 净变化}` 说成一句人话：`多了 1 个杯子、少了 1 个瓶子`。

    `changes` 里正数是多了、负数是少了（0 不该出现在这里，调用方先滤掉）。
    `totals` 是**当前各类的总数**（`PresenceTracker.visible()`）。传了它，
    句子末尾会补一句"现在还剩多少" —— 见下面那段"为什么要带总数"。

    ⚠️ **只说它真的知道的东西：什么、几个。**
    不说"在桌上"、"在你手里"、"是你放的" —— 检测器给不出这些，
    编出来就是撒谎。而这句话很可能被大模型原样念出去（它还会被写进
    对话历史），所以这里必须守住。方位、归属、原因，一律不猜。

    ------------------------------------------------------------
    为什么要带 `totals`（这是探针跑出来才发现的）
    ------------------------------------------------------------
    场景：桌上先有 1 个杯子（0→1，说"多了 1 个杯子"），
    20 秒后你又放了 1 个（1→2，还说"多了 1 个杯子"）。
    **两句都是真的**，但听起来就是复读机 —— 而 notice 的冷却有 15 秒，
    挡不住这种间隔。

    带上总数之后，第二句变成"多了 1 个杯子，现在 2 个杯子"，
    同一件事的两次变化就区分开了。**第一次不加**（本来就是从 0 开始，
    "多了 1 个"和"现在 1 个"是同一件事，加了反而啰嗦）。
    "少了"同理：还剩就报"还剩 N 个"，一个不剩就干脆不加。
    """
    if not changes:
        return ""
    up, down = [], []
    # 排序后再展开，保证同一批变化每次说法一致（别让"杯子、瓶子"
    # 这次这样说、下次反过来 —— 日志会很难比对）
    for name in sorted(changes, key=lambda n: (-abs(changes[n]), n)):
        d = int(changes[name])
        if d > 0:
            up.extend({"name": name} for _ in range(d))
        elif d < 0:
            down.extend({"name": name} for _ in range(-d))

    parts = []
    if up:
        parts.append("多了 " + _vision_summarize(up, max_n))
    if down:
        parts.append("少了 " + _vision_summarize(down, max_n))

    # ---------- 末尾那句"现在还剩多少" ----------
    # 用逗号接在句子后面、**不**加括号：括号在小屏/语音里都别扭，
    # 而且这句话是要被念出来的。
    notes = []
    if totals:
        for name in sorted(changes, key=lambda n: (-abs(changes[n]), n)):
            if len(notes) >= max_n:
                break
            now = totals.get(name)
            if now is None:
                continue
            d = int(changes[name])
            if d > 0 and now <= d:
                continue          # 0 → 1："多了 1 个"已经说清了，不用再说"现在 1 个"
            if d < 0 and now <= 0:
                continue          # 一个不剩 → "还剩 0 个"太啰嗦，前半个句子够了
            word = "现在还剩 " if d < 0 else "现在 "
            # 复用 _vision_summarize 是为了量词（"2 个杯子"、而不是"2 个 cup"）——
            # 量词表的源头在 vision.py，不在这里抄第二份。
            notes.append(word + _vision_summarize([{"name": name}] * now, max_n))

    said = "、".join(parts)
    if notes:
        said += "，" + "、".join(notes)
    return said


# ================================================================
#  自检：python presence.py
# ================================================================
def _box(name, conf=0.9):
    return {"name": name, "box": (10, 10, 60, 60), "conf": conf}


def _selftest():
    print("=" * 66)
    print("  MOSS 变化感知层自检（阶段 19 · 物体在场跟踪）")
    print("=" * 66)

    def feed(t, seq):
        """把一串"每次检测的物体列表"喂进去。"""
        for frame in seq:
            t.update(frame)

    def warm(t, frames=WARMUP_DETECTS):
        """暖机：喂空的，把 n_updates 推过门槛。"""
        for _ in range(frames):
            t.update([])

    # ----------------------------------------------------------
    print("\n[1/6] 迟滞：出现要攒够，走了也要攒够 —— 中途闪一下不算")
    t = PresenceTracker()
    warm(t)
    # 连续 3 次检出 → 认"来了"
    feed(t, [[_box("cup")], [_box("cup")], [_box("cup")]])
    assert t.visible() == {"cup": 1}, t.visible()
    print("       ✓ 连续 3 次检出 → 认「来了」（cup x1）")

    # 只漏 3 次还不够（要 7 次）
    feed(t, [[], [], []])
    assert t.visible() == {"cup": 1}, ("漏 3 次就不该判『没了』", t.visible())
    print("       ✓ 漏检 3 次仍算在场（离开要 7 次）—— 手挡一下不算『杯子没了』")

    feed(t, [[], [], [], []])
    assert t.visible() == {}, t.visible()
    print("       ✓ 漏检满 7 次 → 认「没了」")

    # ----------------------------------------------------------
    print("\n[2/6] 抖动容忍：一帧漏检不该把进度清零")
    t = PresenceTracker()
    warm(t)
    # 真实检测的抖法：**大部分时候认得出，偶尔漏一两帧**（遮挡、角度、光照）。
    # 检出 5 次 / 漏 3 次，交错着来。
    feed(t, [[_box("bottle")], [], [_box("bottle")], [_box("bottle")], [],
             [_box("bottle")], [_box("bottle")], []])
    assert t.visible() == {"bottle": 1}, \
        ("交错的真杯子也必须能认出来（分数是退一格，不是归零）", t.visible())
    print("       ✓ 「中/漏/中/中/漏/中/中/漏」→ 认出来了"
          "（一断就归零的写法在这里会永远攒不够）")

    # 反方向：抖得太厉害的**不该**认 —— 这条边界要明说，不然将来有人
    # 为了让上面那条过，把门槛一路调低，就会把噪声也认进来。
    t2 = PresenceTracker()
    warm(t2)
    feed(t2, [[_box("bottle")], []] * 20)          # 严格 50% 交替
    assert t2.visible() == {}, \
        ("命中率只有一半的目标，宁可不认 —— 认了就是让 MOSS 开始复读", t2.visible())
    print("       ✓ 严格 50% 交替 → **不认**（命中率一半的目标本来就不可信）")

    # ----------------------------------------------------------
    print("\n[3/6] 噪声：闪一下不该报成「来了又走了」")
    t = PresenceTracker()
    warm(t)
    feed(t, [[_box("cat")], [], [], [_box("cat")], [], []])
    assert t.visible() == {}, t.visible()
    assert t.stats["appear"] == 0 and t.stats["vanish"] == 0, t.stats
    assert t.stats["blip"] == 2, ("闪了两下就该记两笔", t.stats)
    print(f"       ✓ 6 次检测里闪 2 下：0 次『来了』、0 次『没了』、"
          f"{t.stats['blip']} 次『闪一下』")

    # ----------------------------------------------------------
    print("\n[4/6] 个数：1 个变 2 个也要确认，不能跟着框跳")
    t = PresenceTracker()
    warm(t)
    feed(t, [[_box("cup")]] * 3)
    assert t.visible() == {"cup": 1}, t.visible()
    # 一帧蹦成 2 个（多半是重叠框），随即回到 1
    feed(t, [[_box("cup"), _box("cup")], [_box("cup")], [_box("cup")]])
    assert t.visible() == {"cup": 1}, ("一帧的 2 个不该改账", t.visible())
    # 稳定地 2 个
    feed(t, [[_box("cup"), _box("cup")]] * 3)
    assert t.visible() == {"cup": 2}, t.visible()
    print("       ✓ 蹦一帧 2 个不改账；稳定 3 次才认 → cup x2")

    # ----------------------------------------------------------
    print("\n[5/6] 暖机：开机那一下不该把一屋子东西报成「新增」")
    t = PresenceTracker()
    scene = [[_box("laptop"), _box("keyboard"), _box("cup")]] * 12
    feed(t, scene)
    assert t.ready, "12 次检测后应该暖机完了"
    # 关键：整个暖机过程里，这些"本来就在"的东西**一次都不该算『来了』**。
    # ⚠️ 这里原来断言的是 `appear == 3`（"3 个类各算一次，用于内部记账"）。
    # 但实测收尾统计会因此打出"新出现 5 · 消失 4"，而真正开过口的只有 2 次 ——
    # 读日志的人会以为这个功能烂了。那两个数本来就该是"它报了几次"，
    # 所以改成暖机期间**一个都不计**。
    assert t.stats["appear"] == 0, \
        ("开机就在的东西不该算『新出现』—— 它不是被报出来的", t.stats)
    assert t.visible() == {"laptop": 1, "keyboard": 1, "cup": 1}, t.visible()
    print("       ✓ 开机画面里的笔记本/键盘/杯子 → 稳定在场，"
          "但『新出现』计数仍是 0（调用方只建基线、不报）")

    # 暖机**之后**的变化照常计 —— 别把上面那条改过头了
    feed(t, [[_box("bottle")]] * 3)
    assert t.stats["appear"] == 1, ("暖机后的『来了』要正常计", t.stats)
    print("       ✓ 暖机之后的『来了』照常计（这一条防的是把它改过头）")

    # ----------------------------------------------------------
    print("\n[6/6] 什么不该报：人是行为层的事，家具是背景")
    assert not announceable("person"), "人已经有 arrive/leave"
    assert not announceable("face")
    assert not announceable("chair")
    assert not announceable("dining table")
    assert announceable("cup") and announceable("bottle")
    assert announceable("笔记本电脑"), "表里没有的类默认要报（换自训练模型时用）"

    t = PresenceTracker()
    warm(t)
    feed(t, [[_box("person"), _box("chair"), _box("cup")]] * 3)
    assert t.visible() == {"cup": 1}, \
        ("人和椅子不该出现在这里 —— 报人就是说两遍", t.visible())
    print("       ✓ person / chair / dining table 不进 visible()；cup 进")

    # 措辞
    assert describe({"cup": 1}) == "多了 1 个杯子", describe({"cup": 1})
    assert describe({"cup": -2}) == "少了 2 个杯子", describe({"cup": -2})
    s = describe({"cup": 1, "bottle": -1})
    assert "多了 1 个杯子" in s and "少了 1 个瓶子" in s, s
    assert describe({}) == ""
    # 只说知道的：不许出现方位/归属/原因
    for bad in ("桌上", "手里", "你放", "因为"):
        assert bad not in describe({"cup": 1}), \
            ("编造了检测器给不出的信息：%s" % bad)
    print(f"       ✓ 措辞：{describe({'cup': 1, 'bottle': -1})}（不含方位/归属/原因）")

    # 带总数时：同一个类连着变两次，两句不能一字不差（那听起来是复读机）
    first = describe({"cup": 1}, totals={"cup": 1})
    second = describe({"cup": 1}, totals={"cup": 2})
    assert first == "多了 1 个杯子", ("0→1 不必再说「现在 1 个」", first)
    assert second == "多了 1 个杯子，现在 2 个杯子", second
    assert first != second, "1→2 和 0→1 必须是两句话"
    third = describe({"cup": -1}, totals={"cup": 1})
    assert third == "少了 1 个杯子，现在还剩 1 个杯子", third
    last = describe({"cup": -1}, totals={"cup": 0})
    assert last == "少了 1 个杯子", ("一个不剩就不必说「还剩 0 个」", last)
    print(f"       ✓ 连着变两次不说重复的话：{first!r} → {second!r}")

    print("\n统计示例：%s" % PresenceTracker().stats_line())
    print("\n[OK] 变化感知层自检通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
