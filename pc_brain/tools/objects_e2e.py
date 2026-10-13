#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 19 端到端探针：从"检测器一闪一闪"到"它开口说了句人话"
============================================================
阶段 18 让 MOSS 认得出一帧里有什么。但**一帧**是没有意义的：
同一个杯子，yolov8n 会给

    第 1 帧 0.91 → 第 2 帧 没有（被手挡了一下）→ 第 3 帧 0.89 → 第 4 帧 没有

如果谁把"这一帧有没有 cup"直接拿去判断变化，MOSS 就会像个坏掉的报警器：
"多了个杯子、杯子没了、多了个杯子、杯子没了……"

这个探针要证明的是那条承诺：**抖动不会被当成变化，真变化一个也不会漏。**

------------------------------------------------------------
为什么必须单独做这个探针（自检和它各管一段）
------------------------------------------------------------
`presence.py --selftest` 管滤波本身：给一串数，它该输出什么。
`behave.py --selftest` 管行为层：给一串**已经稳定**的计数，它该动不该动。

**但真正会出问题的是接缝**：vision 线程的检测频率、presence 的门槛单位、
behavior 的节流窗 —— 这三件事分属三个文件，各自的注释都写对了，
合起来却可能偷偷把"连续 3 次"变成"连续 1 次"。这种 bug 每一段自检都绿，
只有把三段串起来才看得见。

所以这里**三段都是真的**：真的 PresenceTracker、真的 BehaviorEngine、
真的 ActuatorLink、真的串口协议、真的设备解析。
唯一脚本化的是"检测器这一刻看到了什么" —— 而它本来就该被脚本化，
因为它才是那个飘的东西。

三部分：

  A. **滤波层**：造一串带噪序列喂给真 PresenceTracker，看它认不认得出真变化
  B. **接缝**：presence → behavior 串起来跑，看它会不会抽风 / 会不会漏
  C. **整条链**：真起一个 `moss_brain.py` 进程 + 假设备，
     看那句话到底有没有从扬声器/屏幕上出来

用法：
    python tools/objects_e2e.py           # A + B + C
    python tools/objects_e2e.py --fast    # C 跑短一点（时间紧的时候）
    python tools/objects_e2e.py --no-chain  # 只跑 A + B（不起子进程，两秒跑完）
"""

import argparse
import os
import re
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from presence import PresenceTracker          # noqa: E402
from behave import BehaviorEngine             # noqa: E402


# ================================================================
#  夹具
# ================================================================
def _o(name, conf=0.9):
    return {"name": name, "box": (10, 10, 60, 60), "conf": conf}


def flat(n, names):
    """n 次检测，每次都看到同样的东西。"""
    return [([_o(x) for x in names] if names else []) for _ in range(n)]


def noisy(n, names, miss_every=3):
    """n 次检测，每 miss_every 次漏一次。

    这个 `miss_every=3` 不是随便挑的：它代表"检出的次数比漏检多"这种
    **健康但不完美**的检测器。真实摄像头前表现就是这个量级 ——
    如果检测器漏得比这还狠，那该去调的是 `--vision-conf`，不是这里。
    """
    return [([_o(x) for x in names] if (i % miss_every) else []) for i in range(n)]


def run_chain(dets, autonomy="l1", gap=0.0, dt=1.0, **kw):
    """把一串"每次检测的物体列表"喂给**真的** presence + **真的** behave。

    ⚠️ 注意喂的节奏：每次**检测**一次，不是每帧一次。
    这正是要测的那条接缝 —— 如果谁在这里写成每帧一次，
    `presence` 的门槛单位就悄悄从"检测次数"变成了"帧数"。
    """
    t = PresenceTracker(**kw)
    clk = [1000.0]
    eng = BehaviorEngine(autonomy, gap=gap, clock=lambda: clk[0])
    acts = []
    for objs in dets:
        t.update(objs)
        a = eng.tick({
            "faces": 0, "clap": False, "speaking": False,
            "user_active": False, "mode": "IDLE",
            "obj_counts": t.visible(),
            "objs_ready": t.ready,
        })
        if a is not None:
            acts.append(a)
        clk[0] += dt
    return acts, t, eng


def n_raw_flips(dets):
    """这一串检测**原始**结果翻转了多少次。

    这是"如果谁不做滤波会报多少次"的对照量 ——
    用来证明我们的 2 次不是因为序列里本来就只变了 2 次。
    """
    flips, prev = 0, None
    for objs in dets:
        cur = tuple(sorted(o["name"] for o in objs))
        if prev is not None and cur != prev:
            flips += 1
        prev = cur
    return flips


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# ================================================================
#  A. 滤波层
# ================================================================
def part_a(check):
    print("=" * 74)
    print("  A. 滤波层 —— 带噪检测序列 → 真 PresenceTracker")
    print("=" * 74)

    # ---------------- A1 抖动不能认成"东西没了" ----------------
    # ⚠️ 这里**只喂到噪声段结束**，不接着喂那一串空的。
    # 第一版把 `+ flat(26, [])` 也喂了进去，然后断言 visible()=={"cup":1} ——
    # 那 26 次漏检本来就是**该**让它判"走了"的，断言从根上就写错了。
    # 这正是这个项目反复踩的那句：断言写错和代码写错，看起来一模一样。
    t = PresenceTracker()
    for objs in flat(12, []) + noisy(30, ["cup"], 3):
        t.update(objs)
    check("A1 一闪一闪的杯子也能认出来",
          t.visible() == {"cup": 1},
          "稳定计数 = %r（每 3 次漏 1 次也不该丢）" % t.visible())

    # ---------------- A2 整个来回只算一次"来"、一次"走" ----------------
    t = PresenceTracker()
    dets = flat(12, []) + noisy(30, ["cup"], 3) + flat(26, [])
    for objs in dets:
        t.update(objs)
    check("A2 一进一出只记 1 次「来」1 次「走」",
          t.stats["appear"] == 1 and t.stats["vanish"] == 1,
          "appear=%d vanish=%d（原始检测翻转 %d 次，滤波后只记 2 次）"
          % (t.stats["appear"], t.stats["vanish"], n_raw_flips(dets)))

    # ---------------- A3 离开比进入钝（这是**刻意**的不对称）----------------
    # 第一版这里用 `i - 10` 想减掉暖机，结果算出 33 —— 因为 `visible()`
    # **根本不受暖机影响**（暖机只影响"报不报变化"，不影响"在不在场"）。
    # 这类"数字对不上"最容易被人当成代码有 bug，所以两段分开数。
    t = PresenceTracker()
    for objs in flat(12, []):
        t.update(objs)
    n_enter = 0
    for i, objs in enumerate(flat(30, ["cup"]), 1):
        t.update(objs)
        if t.visible() and not n_enter:
            n_enter = i
    t2 = PresenceTracker()
    for objs in flat(12, []) + flat(30, ["cup"]):
        t2.update(objs)
    n_leave = 0
    for i, objs in enumerate(flat(30, []), 1):
        t2.update(objs)
        if not t2.visible():
            n_leave = i
            break
    check("A3 离开(7 次)比进入(3 次)钝得多",
          n_enter == 3 and n_leave == 7,
          "进入要 %s 次检出、离开要 %s 次漏检 —— 不对称是刻意的："
          "晚说一句只是迟钝，把「手挡了一下」说成「东西没了」是编造事实"
          % (n_enter, n_leave))

    # ---------------- A4 个数变化也要确认 ----------------
    t = PresenceTracker()
    for objs in flat(12, []) + flat(20, ["cup"]) + flat(20, ["cup", "cup"]):
        t.update(objs)
    # 20 次"2 个"足够确认（COUNT_HITS=3）
    check("A4 1 个杯子变成 2 个能被确认",
          t.visible() == {"cup": 2},
          "稳定计数 = %r" % t.visible())

    # 但只闪一下不算
    t = PresenceTracker()
    seq = flat(12, []) + flat(20, ["cup"]) + [([_o("cup"), _o("cup")])] + \
        flat(40, ["cup"])
    for objs in seq:
        t.update(objs)
    check("A4 单帧多出一个框不算（框在跳）",
          t.visible() == {"cup": 1},
          "稳定计数 = %r（个数是二阶信息，比「在不在」更容易被噪声影响）"
          % t.visible())

    # ---------------- A5 暖机：开机就有的不算"新增" ----------------
    t = PresenceTracker()
    for objs in flat(11, ["keyboard", "cup", "laptop"]):
        t.update(objs)
    check("A5 开机就摆着的东西不算「新增」",
          t.stats["appear"] == 0 and t.ready,
          "appear=%d（暖机 10 次之后才 ready=%s）—— "
          "没这一步它一启动就报一屋子" % (t.stats["appear"], t.ready))

    # ---------------- A6 人和家具不报 ----------------
    t = PresenceTracker()
    for objs in flat(12, []) + flat(20, ["person", "chair", "tv", "cup"]):
        t.update(objs)
    check("A6 人 / 家具不进「变化」清单",
          t.visible() == {"cup": 1},
          "稳定计数 = %r（人已有 arrive/leave，家具位置本来就不动）"
          % t.visible())

    # ---------------- A7 「闪一下没够着」是健康指标，不是错误 ----------------
    t = PresenceTracker()
    for objs in flat(12, []) + [[_o("cup")], [], []] * 4:
        t.update(objs)
    check("A7 没够着门槛的计入「闪一下」而不是「来了」",
          t.stats["appear"] == 0 and t.stats["blip"] >= 1,
          "appear=%d blip=%d —— 这个数很大就说明检测器在抖，"
          "该去调 --vision-conf 或者看光照" % (t.stats["appear"], t.stats["blip"]))


# ================================================================
#  B. 接缝：presence → behavior
# ================================================================
def part_b(check):
    print()
    print("=" * 74)
    print("  B. 接缝 —— 真 PresenceTracker → 真 BehaviorEngine")
    print("=" * 74)

    # ---------------- B1 核心承诺：抖动不抽风 ----------------
    dets = flat(12, []) + noisy(30, ["cup"], 3) + flat(26, [])
    acts, t, eng = run_chain(dets)
    says = [a.say for a in acts if a.name == "notice"]
    raw = n_raw_flips(dets)

    check("B1 一进一出只开口两次（不是二十几次）",
          len(says) == 2,
          "原始检测翻转 %d 次；行为层开口 %d 次：%s"
          % (raw, len(says), " / ".join(x or "(L0 不出声)" for x in says)))
    check("B1 说的内容对得上",
          len(says) == 2 and "多了 1 个杯子" in says[0] and "少了 1 个杯子" in says[1],
          "第一次 %r，第二次 %r" % (says[0] if says else None,
                                    says[1] if len(says) > 1 else None))
    check("B1 对照：不滤波会报 %d 次" % raw, raw >= 15,
          "这就是为什么必须滤波 —— 它**不是**过度设计")

    # ---------------- B2 权限分级：L0 只闪灯，L1 才说话 ----------------
    acts0, _, eng0 = run_chain(dets, autonomy="l0")
    notices0 = [a for a in acts0 if a.name == "notice"]
    check("B2 L0：注意到了，但一句话不说",
          len(notices0) == 2 and all(a.say is None for a in notices0)
          and all(a.cmds == ("LED 0 255 160 700",) for a in notices0),
          "2 次都只发了 %r；blocked_say=%d（想说没说是单独记的一笔）"
          % (notices0[0].cmds if notices0 else None, eng0.stats["blocked_say"]))

    # ---------------- B3 节流窗里的变化**不能丢** ----------------
    # 场景：先有人来（占掉节流窗），紧接着桌上多了个杯子。
    # 这一刻注定发不出去 —— 但账不能结，节流一过必须补上。
    clk = [500.0]
    eng = BehaviorEngine("l1", gap=6.0, clock=lambda: clk[0])
    t = PresenceTracker()
    eng.tick({"faces": 1, "clap": False, "speaking": False, "user_active": False,
              "mode": "VISION", "obj_counts": {}, "objs_ready": True})
    for objs in flat(12, ["cup"]):
        t.update(objs)
    clk[0] += 0.5
    a_in_gap = eng.tick({"faces": 1, "obj_counts": t.visible(), "objs_ready": True,
                         "clap": False, "speaking": False, "user_active": False,
                         "mode": "VISION"})
    late = None
    clk[0] += 6.0
    a_late = eng.tick({"faces": 1, "obj_counts": t.visible(), "objs_ready": True,
                       "clap": False, "speaking": False, "user_active": False,
                       "mode": "VISION"})
    check("B3 节流窗里的变化不丢，节流一过补上",
          a_in_gap is None and a_late is not None
          and a_late.name == "notice" and "多了 1 个杯子" in (a_late.say or ""),
          "节流窗内 = %s；节流过后 = %s"
          % (a_in_gap, (a_late.say if a_late else None)))

    # ---------------- B4 人来了不说两遍 ----------------
    clk = [700.0]
    eng = BehaviorEngine("l1", gap=0.0, clock=lambda: clk[0])
    t = PresenceTracker()
    acts = []
    for i in range(20):
        objs = ([_o("person"), _o("cup")] if i >= 10 else [])
        t.update(objs)
        a = eng.tick({"faces": 1 if i >= 10 else 0, "obj_counts": t.visible(),
                      "objs_ready": t.ready, "clap": False, "speaking": False,
                      "user_active": False, "mode": "VISION"})
        if a is not None:
            acts.append(a)
        clk[0] += 1.0
    says = [a.say for a in acts]
    n_person = sum(1 for s in says if s and "个人" in s)
    check("B4 人来了走 arrive，不再重复说「多了 1 个人」",
          any(a.name == "arrive" for a in acts) and n_person == 0,
          "动作序列 = %s" % [a.name for a in acts])

    # ---------------- B5 真变化夹在抖动里，一个都不漏 ----------------
    # 最像真实情况的一条：杯子一直在闪，中间**真的**发生了变化。
    #
    # ⚠️ 期望是 **4** 次，不是 2 次。第一版我写了 2，是想当然。
    # 真实发生的是 1 个 → 2 个 → 1 个 → 0 个，四次都是真变化，
    # 而且每两次之间隔了 20 秒（> notice 的 15 秒冷却），所以四次都该说。
    # 这里真正要盯的是**它们不能是同一句话**（"多了 1 个杯子"连说两遍
    # 听起来就是复读机）—— 这件事靠 `describe` 带上总数解决，
    # 也正是这条断言把它逼出来的。
    dets = (flat(12, []) + noisy(20, ["cup"], 3)
            + noisy(20, ["cup", "cup"], 3)      # 真的多了一个杯子
            + noisy(20, ["cup"], 3)             # 又拿回去一个
            + flat(30, []))                     # 全拿走
    acts, t, eng = run_chain(dets)
    says = [a.say for a in acts if a.name == "notice"]
    check("B5 抖动里夹着真变化：次数对得上",
          len(says) == 4,
          "原始翻转 %d 次 → 开口 %d 次（1→2→1→0，四次都是真变化）：%s"
          % (n_raw_flips(dets), len(says), " / ".join(says)))
    dup = [says[i] for i in range(1, len(says)) if says[i] == says[i - 1]]
    check("B5 连着两次变化不说同一句话",
          not dup and len(says) == 4,
          "4 句都不重样；重复的：%s" % (dup or "无"))
    if len(says) == 4:
        print("       · %s" % "\n       · ".join(says))


# ================================================================
#  C. 整条链：真起一个 moss_brain 进程
# ================================================================
def part_c(check, frames=300):
    print()
    print("=" * 74)
    print("  C. 整条链 —— 真 moss_brain 进程 + 假设备")
    print("=" * 74)
    port = _free_port()
    # ⚠️ 一定要 `-u`：假设备是被管道读的，Python 默认按块缓冲 stdout，
    # 而我们最后是 terminate() 掉它的 —— 缓冲区里的回显会**全部丢掉**，
    # 表现出来就是"设备什么都没收到"。第一版就是这么被骗过去的。
    dev = subprocess.Popen(
        [sys.executable, "-u", os.path.join(HERE, "mock_esp32.py"),
         "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=ROOT)
    try:
        time.sleep(0.6)
        cmd = [sys.executable, os.path.join(ROOT, "moss_brain.py"),
               "--demo", "--vision-model", "fake",
               "--autonomy", "l1", "--tts", "off",
               # 节流调 0：这条用例测的不是节流，是"话到底出没出去"。
               # 默认 6 秒会把 timing 变得又长又脆（阶段 16 那次已经吃过亏）。
               "--proactive-gap", "0",
               # 检测每帧一次。默认的 3 会让假剧本每 3 帧才推进一步，
               # 一轮剧本要 28 秒 —— 而这条用例要的只是"变化有没有变成话"。
               "--vision-detect-every", "1",
               "--objects-warmup", "3",
               "--frames", str(frames),
               "--port", "socket://127.0.0.1:%d" % port]
        print("  跑：moss_brain.py --demo --vision-model fake --autonomy l1 "
              "--proactive-gap 0 --vision-detect-every 1 --frames %d" % frames)
        r = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=180)
        out = r.stdout or ""
    except subprocess.TimeoutExpired:
        out = ""
        check("C 大脑进程能跑完", False, "超时了（180 秒）")
    finally:
        dev.terminate()
        try:
            dev_out = dev.communicate(timeout=5)[0] or ""
        except Exception:
            dev_out = ""

    lines = [l for l in out.splitlines() if "[主动] 说出口" in l]
    # ⚠️ 设备回显要从**大脑自己的 stdout** 里找，不是从假设备的子进程管道里。
    # mos_brain 会把设备回话原样打出来（前缀 `[设备]`），而假设备那一路
    # 是被 terminate() 掉的、管道里抓不到东西 —— 第一版从这里读，
    # 结果明明是通的却报"设备什么都没收到"，差点让我去怀疑串口。
    dev_lines = [l for l in out.splitlines() if "[设备]" in l and "TEXT =" in l]
    plus = [l for l in dev_lines if re.search(r"TEXT = \+", l)]

    # 说明：这里**只**断言"多了"这一个方向。
    # 两个方向本身在 A/B 里都已经验过（含措辞、权限、节流），而主循环里
    # 从 `act.tag` 到 `send_raw` 是**同一条**代码路径，方向不参与任何分支。
    # 硬要在 C 里也看到"少了"，就得把 notice 的 15 秒冷却一起等过去 ——
    # 用例从 15 秒变成 35 秒，换来的是一个别处已经覆盖过的断言。
    # **那是拿稳定性换重复覆盖**，不划算。（手工跑 700 帧时两个方向都出现过。）
    check("C 真的开口说了（不是只有灯闪）", len(lines) >= 1,
          "说出口 %d 次：%s" % (len(lines), lines[0].strip() if lines else "（一条都没有）"))
    check("C 设备真的收到了「多了」的标签", bool(plus),
          "设备侧回显 %d 条，其中「多了」：%s"
          % (len(dev_lines), plus[0].split("] ")[-1] if plus else "（没收到）"))
    tag_re = re.compile(r"\[OLED\] (\S.*)$")
    longest, seen_tags = 0, set()
    for l in lines:
        m = tag_re.search(l)
        if m:
            seen_tags.add(m.group(1))
            longest = max(longest, len(m.group(1)))
    check("C 标签没超过 OLED 的 23 个字符", 0 < longest <= 23,
          "最长的一条 %d 个字符（固件 gLine1[24]）：%s"
          % (longest, "、".join(sorted(seen_tags))))
    check("C 说出口的和发给设备的是同一批变化",
          any(("+" + n) in " ".join(seen_tags)
              for n in ("cup", "laptop", "keyboard", "bottle"))
          and any(("+cup") in l or ("+laptop") in l or ("+keyboard") in l
                  for l in dev_lines),
          "OLED 标签 = %s；设备侧 %d 条回显" % (sorted(seen_tags), len(dev_lines)))
    check("C 收尾统计里有变化感知那一行",
          "变化感知" in out,
          next((l.strip() for l in out.splitlines() if "变化感知" in l), ""))


# ================================================================
def main():
    ap = argparse.ArgumentParser(description="阶段 19 端到端探针")
    ap.add_argument("--no-chain", action="store_true", help="跳过 C（不起子进程）")
    ap.add_argument("--fast", action="store_true", help="C 跑短一点（180 帧）")
    ap.add_argument("--frames", type=int, default=0, help="C 里大脑跑多少轮，默认 300")
    a = ap.parse_args()

    print("=" * 74)
    print("  阶段 19 端到端探针 —— 东西变了，它会不会说、会不会说错")
    print("=" * 74)
    print("  不需要摄像头、不需要模型、不需要板子。")
    print()

    fails = []

    def check(title, cond, detail=""):
        print("  %s %s" % ("✓" if cond else "✗", title))
        if detail:
            print("      " + detail)
        if not cond:
            fails.append(title)

    part_a(check)
    part_b(check)
    if not a.no_chain:
        part_c(check, frames=a.frames or (180 if a.fast else 300))

    print()
    print("=" * 74)
    if fails:
        print("  失败 %d 项：" % len(fails))
        for f in fails:
            print("    ✗ " + f)
        print("=" * 74)
        return 1
    print("  全部通过。")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
