#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOSS 主动行为层（"自主"）  ·  流浪地球 550W 复刻项目 · 阶段 16
============================================================
到阶段 15，MOSS 的交互是**完美对称**的：你叫它，它答；你不叫，它就是个摆件。

但 550W 不是摆件。它会**先看你一眼**，再决定要不要说话。
这一步让它自己动起来：有人出现就转过去看你，人走了转回原位，
听到拍手闪一下灯。

------------------------------------------------------------
这一层的真正难题不是"怎么动"，而是"什么时候不许动"
------------------------------------------------------------
把"看到人脸就转头"写成代码只要三行。难的是**它不该每次都转**：

- 你坐在它面前两小时，它不该转 7200 次
- 你正跟它说话，它不该突然"咔"一声转个圈给你看
- 你刚让它安静，它不该拿蜂鸣器跟你打招呼
- 总共就一个电机，不能同时转头和"急停"

所以这个模块的大部分代码是**约束**，不是动作：

| 约束 | 防的是什么 |
|------|-----------|
| 边沿触发（0→1 / 1→0） | 脸一直在，不该每帧都动 |
| 每行为独立冷却 | 一分钟里来来回回十次，不该动十次 |
| 全局节流 | 转头和闪灯不该叠在一起，像抽风 |
| 权限分级 | 万一模型/逻辑出岔子，最坏情况也得是"转个头" |
| 说话时降级 | 出声的动作会和它自己的嗓子抢，也会盖住你的话 |

------------------------------------------------------------
权限分级：为什么按"会不会发出声音"来切
------------------------------------------------------------
两档，分界线刻意选在**声音**上，而不是"动作大小"：

- **L0（默认）**：转头、改灯环颜色。它**不发出声音、不打断任何人**，
  最坏情况就是"转了一下"，你顶多觉得它有点活泼。
- **L1**：加上蜂鸣器。蜂鸣会打断你的思路、会在夜里吵醒人 ——
  所以它需要你明确点头（`--autonomy l1`）。

注意 L0 里**没有** `stop_all`、没有写屏幕、没有长时间占据电机。
不是不能做，而是"主动做"这件事本身要克制：
**它自己决定要做的事，应当是撤销成本为零的事。**
真正会改变世界的动作（停机、独占机构）永远该由你开口。

------------------------------------------------------------
和"模型调工具"的分工
------------------------------------------------------------
阶段 14 的执行器工具是**模型**在调：那是"你说，它做"。
这里是**本地反射**在调：不走网络、不走大模型、20Hz 的循环里直接判。
为什么不全交给模型：等一次大模型往返要一秒多，而"有人来了转个头"
应该是**立刻**的 —— 等你问完模型，人已经走了。

阶段 19 的"东西变了"也遵守这条分界，而且**分两步**：
**要不要提这件事**由本地反射立刻决定（上面那套节制），
**具体怎么措辞**可以交给语言层（`Action.say` 带过去，见下）。
这样它既不会因为等模型而错过时机，也不会永远只会说一句写死的话。

------------------------------------------------------------
【阶段 19】触发源从"人"扩到"物"：东西变了也值得说一声
------------------------------------------------------------
到阶段 18，MOSS 认得出杯子、键盘了，但只会**一问一答**：
你不问"画面里有什么"，它就永远不吭声 —— 桌上多了个杯子它也不知道。
阶段 19 把"东西变了"也接进这一层：

    桌子本来空着，你放了个杯子 → "我注意到画面里多了 1 个杯子。"
    杯子被你拿走               → "我注意到画面里少了 1 个杯子。"

为什么仍然放在**行为层**、仍然按权限分级：因为它和"有人来了转个头"是同一类事 ——
**它自己决定要不要开口**。同一套节制（节流、冷却、权限）直接复用，
不另起一套（另起一套的话，"转头的冷却"和"说话的冷却"会各算各的，
两条路叠在一起就变成抽风）。

和已有行为的两个衔接点：

1. **人不在物体事件里。** `person` / `face` 由 `arrive` / `leave` 管，
   物体事件把它们排除掉（`presence.announceable`）。否则一个人走进来
   会同时触发"有人来了"和"画面里多了 1 个人"—— **同一件事说两遍**。
   这个坑阶段 18 在 `look` 工具上已经踩过一次。
2. **权限的分界线仍然是"会不会发声"**：L0 只闪一下灯（它注意到了，但不打扰你），
   L1 才真的说出来。所以"东西变了"这件事天然横跨两档。

------------------------------------------------------------
阶段 19 踩到的第一个坑：**事件会被节流永久丢掉**
------------------------------------------------------------
"画面里多了个杯子"这件事只发生在那一瞬间。如果按"边沿"表达
（这一帧的 diff 非空 → 报），那么**只要这一刻正好被全局节流拦下，这件事就永远丢了** ——
你放了个杯子，MOSS 一句话不说，而且它自己也不知道自己漏了什么。

这和本文件开头 `leave` 踩过的坑是**同一个毛病**，只是换了个触发源。
当时的解法是"用状态表达而不是边沿表达"，这里沿用同一招，而且更彻底：

    **账本放在行为层，只有"真的动了"才结账。**

`_announced` 存着"上次宣布过的计数"。每帧拿当前计数去减它，差就是待办。
被节流拦住 → **不结账** → 下一帧看到的差还是那个 → 节流一过自动补上。
如果这期间东西又变回去了，差自动变回 0，**不需要任何"取消"逻辑** ——
因为待办不是一串事件，而是"账没结清"这个状态。

（一个推论：它天然幂等。20Hz 的循环里同一件事被看一百遍也不会报一百遍，
 因为第一遍报完账就平了。而这正是 `arrive` 当年缺的那一环。）

自检：python behave.py     # 不需要设备、不需要网络
"""

import collections
import time

# 变化感知层（阶段 19）。没有它只是没有"物体事件"，人/拍手那几条照常。
try:
    from presence import describe as describe_objects, announceable
    HAS_PRESENCE = True
except ImportError:
    HAS_PRESENCE = False

    def announceable(name):
        return True

    def describe_objects(changes, totals=None, max_n=3):
        # 兜底版的签名要和 presence.describe 对齐（多一个 totals），
        # 否则"没有 presence.py"时这两行会以 max_n=总数 的样子静默跑歪。
        return ""

# 行为定义。cmds 里每条命令带自己的权限等级，
# 所以同一个行为在不同权限下会发出不同的命令组合。
#
# 为什么用"步"（4096 步/圈）而不是角度：这是固件 hal_motion 的原生单位，
# 少一层换算就少一个出错的地方。1024 步 = 1/4 圈，作为"瞥一眼"刚好。
RULES = {
    "arrive": {
        "why": "看到有人出现",
        "cd": 30.0,                       # 30 秒内不重复"转头迎接"
        "cmds": (("SPIN 1024 CW", 0),     # 转 1/4 圈看向来人
                 ("LED 0 80 255 2500", 0),  # 灯环变"注意蓝"
                 ("BEEP 1500 60", 1)),      # 招呼一声（需要 L1）
    },
    "leave": {
        "why": "人走了，转回原位",
        "cd": 0.0,                        # 靠 _peeked 配对，不靠冷却
        "cmds": (("SPIN 1024 CCW", 0),
                 ("LED 255 120 0 800", 0)),  # 暖橙：像是"目送"
    },
    "clap": {
        "why": "听到拍手",
        "cd": 6.0,
        "cmds": (("LED 0 255 60 500", 0),),
    },
    # ---------- 阶段 19：画面里的东西变了 ----------
    # `cd` 给 15 秒：你在收拾桌子，不该每放下一样东西就被点评一次。
    "notice": {
        "why": "画面里的东西变了",
        "cd": 15.0,
        # L0：灯环闪一下青绿 —— "我注意到了"，但一句话不说，不打扰你。
        # 这一档很重要：它意味着**注意到了不等于要开口**。
        "cmds": (("LED 0 255 160 700", 0),),
        # L1：真的说出来。内容不写死在这儿 ——
        # 每回到底变了什么不一样，措辞由 _try() 现算（见 `say` 字段说明）。
        "say": 1,
    },
}

# 各权限等级允许的最高命令等级
_LEVELS = {"off": -1, "l0": 0, "l1": 1}

# tick() 的返回值。cmds 是**已经按权限过滤过**的命令，直接发就行。
#
# 【阶段 19】多了 `say` 和 `tag` 两个字段：
#   say = None    → 这次只要发命令（老行为，一条都不用改）
#   say = 字符串  → 发完命令之后，把这句话交给语言层说出来
#   tag = 短标签  → 给 OLED 用（固件那边只有 23 个字符的地方）
#
# 为什么不让 behavior 直接去调 TTS：这一层是**纯逻辑、不碰硬件、不走网络**的，
# 所以才能拿假时钟离线跑几千帧。让它去说话，就等于把网络和声卡塞进来了。
# 它只**产出一句话**，"说不说、怎么说、用哪个大模型"由主程序决定。
Action = collections.namedtuple("Action", "name why cmds level say tag")


def change_tag(changes, max_n=3, budget=23):
    """变化的**短标签**，给 OLED 用：`+cup -bottle`。

    为什么不复用 `presence.describe()` 那句话：那句话是**给耳朵听的**
    （"多了 1 个杯子、少了 1 个瓶子"），而固件那边 `char gLine1[24]` 只有
    23 个字符的位置，中文还要一个字占两个宽。这里刻意只用 ASCII：
    `+cup` / `-bottle` 一眼能看懂，而且不用担心字体里有没有这个字。

    ⚠️ 截断按**字符数**卡，不是只按个数。理由：COCO 里有 `dining table`
    这种长名字，只卡个数的话两个长名字就能把 23 个位置挤爆，而固件那边是
    **硬截断**的 —— 屏幕上会出现半截单词。宁可少列一个名字。

    末尾的 `&N` 表示"还有 N 样也变了"。用 `&` 而不是继续用 `+`：
    `+cup +keyboard +1` 里的那个 `+1` 会被读成"又多了一个杯子"，
    而它其实是"还有 1 样别的东西"。这个歧义是实测跑出来才发现的。
    """
    if not changes:
        return ""
    rows = sorted(changes.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
    out = []
    for name, d in rows:
        if len(out) >= max_n:
            break
        item = ("+" if d > 0 else "-") + name + ("" if abs(d) == 1 else "x%d" % abs(d))
        cand = out + [item]
        # 先算好"如果到这里截断，尾巴要多长"，再决定这个能不能放
        left = len(rows) - len(cand)
        tail = (" &%d" % left) if left > 0 else ""
        if len(" ".join(cand)) + len(tail) > budget:
            break
        out.append(item)
    if not out:
        # 一个名字都放不下（名字特别长）：至少让用户知道"有东西变了"，
        # 而不是屏幕上什么都没有 —— 那时候人会以为功能坏了。
        return "+%d" % len(rows)
    left = len(rows) - len(out)
    if left > 0:
        out.append("&%d" % left)
    return " ".join(out)


class BehaviorEngine:
    """把"感知快照"变成"要不要自己动一下"。

    它**不碰硬件** —— 只产出 Action，发不发由主程序决定。
    这样这一层是纯逻辑，可以离线跑几千帧来验证"它会不会抽风"。
    """

    def __init__(self, autonomy="l0", gap=6.0, log=None, clock=time.time):
        self.autonomy = autonomy
        self.allowed = _LEVELS.get(autonomy, 0)
        self.gap = max(0.0, float(gap))
        self.log = log
        self._clock = clock
        self._prev_faces = 0
        self._peeked = False          # 已经转头看过了吗？用来配对"转回来"
        self._last = {}               # 行为名 → 上次执行时刻
        self._last_any = -1e9         # 上一次**任何**主动动作的时刻
        self._gap_seen = set()        # 本次节流期已经记过账的行为（防统计刷屏）
        # ---------- 阶段 19：物体变化的账本 ----------
        # _announced = "上次宣布过的计数"。待办 = 当前计数 - 它。
        # 为什么用"账本"而不是"事件队列"：队列会积压（东西拿走又放回来会产生两条），
        # 而账本是**净额**，变回去就自动清零，不需要任何取消逻辑。
        self._announced = {}
        self._cur_counts = {}         # 这一帧过滤后的稳定计数
        self._obj_diff = {}           # 这一帧的待办变化 {类名: 净变化}
        self.stats = collections.Counter()

    # ---------- 主入口 ----------
    def tick(self, snap):
        """每帧调一次。返回 Action 或 None。

        snap 需要这几个键（缺了就按最保守的值算）：
            faces        画面里有几张脸
            clap         这一帧有没有拍到
            speaking     MOSS 自己在说话吗
            user_active  用户正在说话 / 刚说完
            mode         当前状态名
            obj_counts   稳定在物的物体计数 {类名: 个数}（阶段 19，见 presence.py）
            objs_ready   感知层的暖机完了没（阶段 19）
        """
        now = self._clock()
        faces = int(snap.get("faces") or 0)

        # **边沿**触发。这一句是整个模块最重要的三行：
        # 用"上一帧的脸数"和"这一帧的脸数"比，而不是看"这一帧有没有脸"。
        # 否则你坐在它面前，20Hz 的循环会每帧都判断"有人 → 转头"。
        came = self._prev_faces <= 0 < faces
        self._prev_faces = faces      # 注意：即使权限是 off 也要更新，
                                      # 不然切到 l0 的瞬间会补一次假触发

        # 物体账本同理，而且**必须在权限判断之前**对账（理由见 _sync_objects）。
        self._obj_diff = self._sync_objects(snap)

        if self.allowed < 0:
            return None

        evs = self._events(snap, came)
        if not evs:
            return None

        # ---------- 全局节流 ----------
        # 放在事件判断之后：没事件的时候不该记账。
        # `_gap_seen` 用来防同一个行为在 20Hz 的循环里被记上百次 ——
        # 那会让"节流拦住 N 次"这个数字彻底失去意义。
        if now - self._last_any < self.gap:
            for ev in evs:
                if ev not in self._gap_seen:
                    self._gap_seen.add(ev)
                    self.stats["blocked_gap"] += 1
            return None

        for ev in evs:
            act = self._try(ev, now, snap)
            if act is not None:
                self._gap_seen.clear()
                return act
        return None

    def _sync_objects(self, snap):
        """对账：当前稳定计数 vs 上次宣布过的计数，返回**待办**的净变化。

        这是阶段 19 的核心十几行，也是阶段 16 那个坑的正解。

        **为什么待办是"账没结清"这个状态，而不是一串事件**：
        如果存的是一个事件队列，那么"杯子拿走 → 又放回来"会积两条，
        你去消费的时候还得判断它们是不是互相抵消了。而净额天然会抵消 ——
        这一类"配对/取消"的逻辑全都不需要写。

        **为什么被节流拦住时不结账**：这就是"事件不丢"的全部秘密。
        拦住了 → `_announced` 不动 → 下一帧算出来的差还是那个 → 节流一过自动补。
        反过来，如果被拦住也结账，那件事就**永久丢了**，
        而且丢得悄无声息（MOSS 自己都不知道它漏了什么）。
        """
        counts = {}
        for name, c in (snap.get("obj_counts") or {}).items():
            n = (name or "").strip().lower()
            if n and int(c) > 0 and announceable(n):
                counts[n] = int(c)
        self._cur_counts = counts

        # 暖机没完、或者权限是 off → **只同步账本，不产生待办**。
        # 两种情况要的东西一样，但理由不同：
        #   暖机：一开机画面里本来就有的东西（桌子、键盘、显示器）不该算"新增"，
        #         否则它一启动就报一屋子 —— 那是噪音，不是观察。
        #   off ：要是不同步，切到 l0 的那一瞬间会补报一串"陈年旧事"
        #         （跟 _prev_faces 那边是同一个道理）。
        if self.allowed < 0 or not snap.get("objs_ready"):
            self._announced = dict(counts)
            return {}

        diff = {}
        for name in set(counts) | set(self._announced):
            d = counts.get(name, 0) - self._announced.get(name, 0)
            if d:
                diff[name] = d
        return diff

    def _events(self, snap, came):
        """这一帧值得考虑的事件，按优先级排。"""
        faces = int(snap.get("faces") or 0)
        out = []
        if came:
            out.append("arrive")
        # ⚠️ 这里用**状态**表达，而不是"上一帧有脸 / 这一帧没脸"的边沿。
        # 踩过的坑：leave 原本写成边沿（`prev>0 and now==0`），而边沿只在
        # 那一帧存在 —— 如果那一刻正好被全局节流拦下，这个事件就**永久丢了**，
        # 结果是它转过头去看你，然后人走了它却再也不转回来。
        # 改成状态就天然幂等：只要"转头看过"还成立、而此刻没人，
        # 它就一心想转回来，节流一过就补上。
        if self._peeked and faces <= 0:
            out.append("leave")
        if snap.get("clap"):
            out.append("clap")
        # 【阶段 19】排在最后 = 优先级最低，这是**刻意**的：
        # 有人来了 / 拍手都是"人直接找我"，比"桌上多了个杯子"急。
        # 放在最后一个位子还有个好处 —— 前面几条的优先级一个字节都没动，
        # 阶段 16 那些老断言和真机上的手感全部保持原样。
        if self._obj_diff:
            out.append("notice")
        return out

    def _try(self, name, now, snap):
        rule = RULES[name]

        # notice 的"为什么"和"要说什么"都随**这一次到底变了什么**现算，
        # 不能写死在 RULES 里（每回变的东西都不一样）。
        # 把当前总数一起递过去：同一个类连着变两次时，句子末尾会补一句
        # "现在还剩多少"，免得"多了 1 个杯子"连说两遍（见 presence.describe）。
        desc = (describe_objects(self._obj_diff, self._cur_counts)
                if name == "notice" else "")
        why = ("画面里" + desc) if desc else rule["why"]

        # ---------- 权限：先按当前语境把允许等级压下来 ----------
        eff = self.allowed
        if snap.get("speaking"):
            eff = min(eff, 0)        # 它在说话：别让蜂鸣盖住自己的嗓子
        if snap.get("user_active"):
            eff = min(eff, 0)        # 你在说话：它安静点，别抢戏

        cmds = [c for c, lv in rule["cmds"] if lv <= eff]
        if not cmds:
            # 想做但没资格做。记一笔 —— 这个数字大于 0 就说明
            # 用户其实想要更主动的 MOSS，可以建议他开 --autonomy l1。
            self.stats["blocked_perm"] += 1
            return None

        # ---------- 要不要说话（阶段 19）----------
        # 和 cmds 是**两套**权限判断，因为"闪个灯"和"开口说话"的代价差得远：
        # L0 只闪灯（注意到了但不打扰），L1 才说出口。
        say = None
        say_lv = rule.get("say")
        if desc and say_lv is not None:
            if say_lv <= eff:
                say = "我注意到画面里" + desc + "。"
            else:
                # 想说但没资格。单独记一笔，别和 blocked_perm 混 ——
                # "灯闪了但话没说"和"什么都没做"是两种不同的受限。
                self.stats["blocked_say"] += 1

        # ---------- 冷却：单个行为 ----------
        # 节流（所有行为之间）在 tick() 里判，因为那要在**遍历事件之前**就决定，
        # 否则第一个事件会把第二个"耗"掉一次注定失败的尝试。
        if now - self._last.get(name, -1e9) < rule["cd"]:
            self.stats["blocked_cool"] += 1
            return None

        # ---------- 通过 ----------
        self._last[name] = now
        self._last_any = now
        self.stats[name] += 1
        if name == "arrive":
            self._peeked = True
        elif name == "leave":
            self._peeked = False
        elif name == "notice":
            # **只有走到这里才结账。** 上面任何一条 return 都意味着"这次没办成"，
            # 账本就原样留着 —— 下一帧看到的待办还是它，该补的迟早会补上。
            # 换个说法：结账的位置就是"这件事真的发生了"的唯一判据，
            # 不需要再维护一个"哪些事件已经处理过"的清单。
            self._announced = dict(self._cur_counts)

        act = Action(name, why, tuple(cmds),
                     max(lv for _, lv in rule["cmds"]), say,
                     change_tag(self._obj_diff) if name == "notice" else "")
        if self.log:
            # 这里**不**把 say 的全文打出来：主循环拿到之后会打一条带 [OLED] 的，
            # 两层各打一遍就成了同一个句子连续出现两次（实测跑第一条就看到了）。
            # 这一行只说"要说话"这件事，全句由主程序负责。
            tail = "（要说话）" if say else ""
            self.log(f"[主动] {act.why} → {', '.join(cmds)}{tail}")
        return act

    # ---------- 展示 ----------
    def describe(self):
        if self.allowed < 0:
            return "关闭（完全被动：只有你开口它才动）"
        what = "转头/灯色" + ("+蜂鸣发声" if self.allowed >= 1 else "（不说话）")
        n = sum(self.stats[k] for k in RULES)
        return (f"{self.autonomy} · {what} · 节流 {self.gap:g}s · "
                f"已主动 {n} 次")

    def stats_line(self):
        s = self.stats
        done = sum(s[k] for k in RULES)
        parts = [f"主动 {done} 次"]
        for k in RULES:
            if s[k]:
                parts.append(f"{k} {s[k]}")
        if s["blocked_cool"]:
            parts.append(f"冷却拦住 {s['blocked_cool']}")
        if s["blocked_gap"]:
            parts.append(f"节流拦住 {s['blocked_gap']}")
        if s["blocked_perm"]:
            parts.append(f"权限不够 {s['blocked_perm']}")
        # 【阶段 19】单独一档：灯闪了、话没说。它和"什么都没做"是两种不同的受限，
        # 混在一起的话，用户看到"权限不够 5 次"根本不知道是连灯都没闪，
        # 还是闪了灯只是没开口 —— 而这两种情况该给他的建议是不一样的。
        if s["blocked_say"]:
            parts.append(f"想说没说 {s['blocked_say']}")
        return " · ".join(parts)


# ================================================================
#  自检：python behave.py
# ================================================================
def _selftest():
    print("=" * 66)
    print("  MOSS 主动行为层自检")
    print("=" * 66)

    class Clock:
        """假时钟。真实 sleep 会让"30 秒冷却"这种测试没法跑。"""

        def __init__(self):
            self.t = 500.0

        def __call__(self):
            return self.t

        def tick(self, dt):
            self.t += dt

    def snap(faces=0, clap=False, speaking=False, user_active=False, mode="IDLE",
             objs=None, ready=True):
        """造一帧感知快照。`objs` 是**稳定计数**（`presence.PresenceTracker.visible()` 的输出），
        不是逐帧检测结果 —— 那两者的区别就是阶段 19 整个 `presence.py` 存在的理由。"""
        d = {"faces": faces, "clap": clap, "speaking": speaking,
             "user_active": user_active, "mode": mode, "objs_ready": ready}
        if objs is not None:
            d["obj_counts"] = objs
        return d

    print("\n[1/8] 权限：off 就是完全被动")
    clk = Clock()
    eng = BehaviorEngine("off", log=None, clock=clk)
    for _ in range(50):
        assert eng.tick(snap(faces=1)) is None
        clk.tick(0.05)
    assert eng.describe().startswith("关闭")
    print("       ✓ --autonomy off：有人来也一动不动")

    print("\n[2/8] 边沿触发：脸一直在，不该每帧都动")
    clk = Clock()
    eng = BehaviorEngine("l1", gap=0.0, log=lambda s: print("      " + s), clock=clk)
    acts = []
    for _ in range(200):                 # 200 帧 ≈ 10 秒
        a = eng.tick(snap(faces=1))
        if a:
            acts.append(a)
        clk.tick(0.05)
    assert len(acts) == 1, f"一张脸待了 10 秒，只该动一次，实际 {len(acts)}"
    assert acts[0].name == "arrive"
    print(f"       ✓ 200 帧 / 一张脸 → 只触发 1 次（不是 200 次）")

    print("\n[3/8] 权限分级：l0 不出声，l1 才允许蜂鸣")
    clk_a = Clock()
    e0 = BehaviorEngine("l0", gap=0.0, clock=clk_a)
    a0 = e0.tick(snap(faces=1))
    assert a0 is not None
    assert "BEEP" not in " ".join(a0.cmds), a0.cmds
    assert a0.cmds == ("SPIN 1024 CW", "LED 0 80 255 2500"), a0.cmds
    print(f"       ✓ l0 → {a0.cmds}（没有 BEEP）")

    a1 = BehaviorEngine("l1", gap=0.0, clock=Clock()).tick(snap(faces=1))
    assert any("BEEP" in c for c in a1.cmds), a1.cmds
    print(f"       ✓ l1 → {a1.cmds}（多了一条 BEEP）")

    print("\n[4/8] 礼仪：它自己在说话 / 你在说话 → 自动降级成不出声")
    # 它自己在说话
    clk = Clock()
    eng = BehaviorEngine("l1", gap=0.0, clock=clk)
    a = eng.tick(snap(faces=1, speaking=True))
    assert a is not None and "BEEP" not in " ".join(a.cmds), a.cmds
    print("       ✓ 它在说话时：照样转头，但不蜂鸣（不盖住自己的嗓子）")

    # 你在说话
    clk = Clock()
    eng = BehaviorEngine("l1", gap=0.0, clock=clk)
    a = eng.tick(snap(faces=1, user_active=True))
    assert a is not None and "BEEP" not in " ".join(a.cmds), a.cmds
    print("       ✓ 你在说话时：同样只做无声动作")

    print("\n[5/8] 冷却与节流：别抽风")

    # --- (a) 全局节流：刚动完要歇一会儿 ---
    clk = Clock()
    eng = BehaviorEngine("l1", gap=6.0, log=None, clock=clk)
    first = eng.tick(snap(faces=1))
    assert first is not None and first.name == "arrive", first

    clk.tick(0.5)
    assert eng.tick(snap(faces=0)) is None, "刚动完 0.5 秒不该又动"
    assert eng.stats["blocked_gap"] == 1, eng.stats
    print("       ✓ 全局节流：刚转完头 0.5 秒内不会再动")

    # 节流一过，它**还会**转回来 —— 这正是 leave 用状态而非边沿的理由
    clk.tick(6.0)
    back = eng.tick(snap(faces=0))
    assert back is not None and back.name == "leave", back
    print("       ✓ 节流一过就补上「转回原位」（事件没被丢在那一帧）")

    # --- (b) 行为冷却：30 秒内不再迎接第二次 ---
    clk2 = Clock()
    e2 = BehaviorEngine("l1", gap=0.0, clock=clk2)     # gap=0，排除节流干扰
    assert e2.tick(snap(faces=1)) is not None          # arrive
    e2.tick(snap(faces=0))                             # 人走了 → leave
    clk2.tick(10.0)                                    # 距 arrive 才 10 秒
    assert e2.tick(snap(faces=1)) is None, "arrive 冷却 30 秒，不该这么快又迎接"
    assert e2.stats["blocked_cool"] >= 1, e2.stats
    e2.tick(snap(faces=0))
    print("       ✓ 行为冷却：10 秒内第二次来人不再迎接")

    clk2.tick(30.0)                                    # 距 arrive 40 秒
    a = e2.tick(snap(faces=1))
    assert a is not None and a.name == "arrive", a
    print("       ✓ 超过 30 秒后可以再来一次")

    print("\n[6/8] 不该动的时候绝不动 + 拍手")
    clk = Clock()
    eng = BehaviorEngine("l1", gap=0.0, clock=clk)
    # 没人、没声：一片安静，什么都不该发生
    for _ in range(100):
        assert eng.tick(snap()) is None
        clk.tick(0.05)
    print("       ✓ 没人没声：100 帧零动作")

    # 拍手
    a = eng.tick(snap(clap=True))
    assert a is not None and a.name == "clap", a
    print(f"       ✓ 拍手 → {a.cmds}")

    # 拍手连拍不该连续闪
    clk.tick(0.2)
    assert eng.tick(snap(clap=True)) is None
    print("       ✓ 连拍两下不会闪两次灯（有冷却）")

    # "人走"但从来没"来人"记录 → 不该转回来
    clk2 = Clock()
    e2 = BehaviorEngine("l1", gap=0.0, clock=clk2)
    e2.tick(snap(faces=0))               # 一直是空画面
    a = e2.tick(snap(faces=0))
    assert a is None
    print("       ✓ 没迎接过就不存在「转回原位」（不会莫名其妙转一下）")

    # ================================================================
    #  【阶段 19】物体事件
    # ================================================================
    print("\n[7/8] 东西变了：L0 只闪灯，L1 才说出来")
    # ---- L0：注意到了，但不打扰你 ----
    clk = Clock()
    e0 = BehaviorEngine("l0", gap=0.0, clock=clk)
    a = e0.tick(snap(objs={"cup": 1}))
    assert a is not None and a.name == "notice", a
    assert a.cmds == ("LED 0 255 160 700",), a.cmds
    assert a.say is None, ("l0 不该开口", a.say)
    assert e0.stats["blocked_say"] == 1, ("想说没说该单独记一笔", e0.stats)
    print(f"       ✓ l0 → {a.cmds}（注意到了，但一句话不说）")

    # ---- L1：真的说出来，而且措辞是**这一次到底变了什么**现算的 ----
    clk = Clock()
    e1 = BehaviorEngine("l1", gap=0.0, clock=clk)
    a = e1.tick(snap(objs={"cup": 1}))
    assert a is not None and a.say == "我注意到画面里多了 1 个杯子。", a.say
    assert a.tag == "+cup", ("OLED 短标签要短、要纯 ASCII", a.tag)
    print(f"       ✓ l1 → {a.say}（OLED: {a.tag}）")

    # ---- 少了也要会说（同一个行为、同一句模板，只是方向反了）----
    clk = Clock()
    e1 = BehaviorEngine("l1", gap=0.0, clock=clk)
    e1.tick(snap(objs={"cup": 1, "bottle": 1}))     # 第一次看到 = 建基线
    clk.tick(20.0)                                   # 过掉 notice 的 15 秒冷却
    a = e1.tick(snap(objs={"cup": 1}))               # 瓶子被拿走
    assert a is not None and a.say == "我注意到画面里少了 1 个瓶子。", a
    assert a.tag == "-bottle", a.tag
    print(f"       ✓ 拿走瓶子 → {a.say}（OLED: {a.tag}）")

    # ---- 短标签：多个变化、个数 >1、长名字按**字符数**截断 ----
    assert change_tag({"cup": 1}) == "+cup"
    assert change_tag({"cup": 2, "bottle": -1}) == "+cupx2 -bottle"
    assert change_tag({"a": 1, "b": 1, "c": 1}) == "+a +b +c"
    assert change_tag({}) == ""
    # 两个长名字放不下 23 个字符 → 只列一个，剩下的用 `&1` 报个数
    long_tag = change_tag({"dining table": 1, "potted plant": 1})
    assert long_tag == "+dining table &1", long_tag
    assert len(long_tag) <= 23, len(long_tag)
    # 真机实测跑出来的那个歧义：`+1` 会被读成"又多了一个杯子"
    three = change_tag({"cup": 1, "laptop": 1, "keyboard": 1})
    assert three == "+cup +keyboard +laptop", three
    assert len(three) <= 23, (three, len(three))
    print(f"       ✓ 短标签：+cup / +cupx2 -bottle / 三个变化 {three!r}（{len(three)} 字符）")
    print(f"       ✓ 长名字放不下时列到哪儿算哪儿，尾巴用 &N 报个数（{long_tag!r}）")

    # ---- 不说两遍：人归 arrive 管，不走物体事件 ----
    clk = Clock()
    e2 = BehaviorEngine("l1", gap=0.0, clock=clk)
    a = e2.tick(snap(faces=1, objs={"person": 1, "cup": 1}))
    assert a is not None and a.name == "arrive", ("人优先", a)
    print(f"       ✓ 人来了 → {a.name}（不是 notice；人不说两遍）")
    # **但杯子没丢**，只是排在后面 —— 人一走开（同一帧的脸数不再变化）它就会说
    a = e2.tick(snap(faces=1, objs={"person": 1, "cup": 1}))
    assert a is not None and a.name == "notice" and "1 个杯子" in a.say, a
    print(f"       ✓ 回过头才说杯子 → {a.say}")

    print("\n[8/8] 事件不丢：被节流拦下，节流一过必须补上")
    clk = Clock()
    eng = BehaviorEngine("l1", gap=6.0, clock=clk)
    assert eng.tick(snap(faces=1)) is not None       # arrive 先把节流窗占掉
    clk.tick(0.5)
    # 节流窗里放了个杯子 —— 这一刻注定被拦下
    assert eng.tick(snap(faces=1, objs={"cup": 1})) is None, "节流窗内不该动"
    assert "notice" not in eng._announced, ("没办成就不许结账", eng._announced)
    print("       ✓ 节流窗里放杯子：当场不动，而且**账没结**")

    clk.tick(6.0)                                     # 节流过去
    a = eng.tick(snap(faces=1, objs={"cup": 1}))
    assert a is not None and a.name == "notice", ("节流一过必须补上", a)
    assert a.say == "我注意到画面里多了 1 个杯子。", a.say
    print(f"       ✓ 节流一过自动补上 → {a.say}")

    # 补上之后杯子一直在 → 不该说第二遍。20Hz 的循环会看一百遍同一帧，
    # 靠的就是"账本是净额"这一点，而不是靠谁记得"这件事说过了"。
    n = 0
    for _ in range(100):
        clk.tick(0.5)
        if eng.tick(snap(faces=1, objs={"cup": 1})) is not None:
            n += 1
    assert n == 0, ("同一件事不该说第二遍", n)
    print("       ✓ 杯子一直在：100 帧零重复（净额账本天然幂等）")

    # 变回去了 → 待办自动抵消，不需要任何"取消"逻辑
    clk2 = Clock()
    e2 = BehaviorEngine("l1", gap=0.0, clock=clk2)
    e2.tick(snap(objs={"cup": 1}))
    clk2.tick(20.0)
    a = e2.tick(snap(objs={}))
    assert a is not None and "少了 1 个杯子" in a.say, a
    print("       ✓ 拿走 → 自动报「少了」（变回去不需要取消逻辑）")

    # 暖机：开机就有的东西不该算「新增」
    clk3 = Clock()
    e3 = BehaviorEngine("l1", gap=0.0, clock=clk3)
    a = e3.tick(snap(objs={"keyboard": 1, "tv": 1}, ready=False))
    assert a is None, ("暖机没完只该建基线", a)
    clk3.tick(20.0)
    a = e3.tick(snap(objs={"keyboard": 1, "tv": 1}, ready=True))
    assert a is None, ("开机就有的键盘不算新增", a)
    print("       ✓ 暖机：开机就在的键盘不算「新增」，电视这类不报")

    print("\n统计示例：%s" % eng.stats_line())
    print("\n[OK] 主动行为层自检通过（8/8）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
