#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOSS 执行器层（"手脚"）  ·  流浪地球 550W 复刻项目 · 阶段 14
============================================================
阶段 13 结束时 MOSS 有了一双手 —— 但那双"手"只能查和记：
查时间、查天气、记名字。它读得懂世界，**动不了世界**。

这一步补上另一半：**让模型能真正驱动硬件。**
灯环变红、电机转半圈、蜂鸣器响一声、屏幕上蹦出一行字。

这才是"550W 控制行星发动机"该有的样子 —— 它不只是在旁边解说，
它是那个按按钮的人。

------------------------------------------------------------
为什么单独一个模块，而不是塞进 moss_tools.py
------------------------------------------------------------
`moss_tools.py` 里的工具是**纯函数**：给它上下文，它算个结果。
执行器工具不一样，它们要**往外写字**（串口），而且：

1. 没有设备时它们根本不该出现在清单里 —— 让模型看到一堆
   "set_led / turn_motor"却每次都回"没接上设备"，只会浪费 token 并带偏它。
   所以执行器工具是**按需安装**的：`install(link)` 有设备才注册。

   ⚠️ 这一条必须用"**延迟注册**"实现，不能靠 install() 返回空表：
   如果照常用 @tool 装饰器写在模块顶层，那么 `import moss_actuators`
   这一下就已经把它们塞进注册表了，之后 install() 返不返回空表都没用 ——
   工具照样出现在清单里。这个坑真实发生过（`--no-actuators` 形同虚设）。
   所以下面**不写装饰器**，改成一张规格表，到 install() 里才真正注册。
2. 它们要共享一个串口，得有一把锁（见 ActuatorLink）。
3. 职责干净：moss_tools 是"感官与记忆"，moss_actuators 是"运动与输出"。

------------------------------------------------------------
依赖方向：moss_actuators → moss_tools（单向）
------------------------------------------------------------
复用 moss_tools 的 `@tool` 装饰器和 `_s()` 清洗函数，
这样执行器工具会自动出现在同一份工具清单里，模型看到的是一张表。

自检：
    python moss_actuators.py     # 不需要硬件：用一个"假串口"记录发出去的指令
"""

import collections
import os
import re
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moss_tools import tool, _s                     # noqa: E402

# 固件认得的 8 个状态名（见 moss_state.h）
STATES = ("BOOT", "IDLE", "LISTEN", "THINK",
          "SPEAK", "VISION", "MOTION", "ERROR")

# 步进电机一圈的步数（四相全步励磁，见 hal_motion.cpp）
STEPS_PER_TURN = 4096

# 颜色名 → RGB。让模型可以直接说"变红"，不用算 255,0,0。
COLOR_NAMES = {
    "红": (255, 0, 0), "red": (255, 0, 0),
    "橙": (255, 96, 0), "orange": (255, 96, 0),
    "黄": (255, 200, 0), "yellow": (255, 200, 0),
    "绿": (0, 255, 0), "green": (0, 255, 0),
    "青": (0, 255, 200), "cyan": (0, 255, 200),
    "蓝": (0, 60, 255), "blue": (0, 60, 255),
    "紫": (160, 0, 255), "purple": (160, 0, 255),
    "白": (255, 255, 255), "white": (255, 255, 255),
    "暖白": (255, 190, 120), "warm": (255, 190, 120),
    "灭": (0, 0, 0), "黑": (0, 0, 0), "off": (0, 0, 0),
    "黑掉": (0, 0, 0),
}


def _clampi(v, lo, hi, default):
    """把模型给的参数钳到合理范围。转不动、算不出就退默认值。"""
    try:
        n = int(float(v))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


# ================================================================
#  串口连接：一把锁 + 一份"设备说了什么"的回执缓冲
# ================================================================
class ActuatorLink:
    """执行器与设备之间的那根线。

    它只做四件事：加锁写一行、记流水、收设备回话、给工具层一个可用性判断。
    **不做任何参数校验** —— 校验在工具里做，因为那里才知道每个参数的含义。

    io_lock：和主程序共用同一把锁。因为 moss_brain 的主循环也在往同一个
    串口写 MODE/TEXT，两条线程各写各的会交错（pyserial 的 write 不保证原子）。
    """

    def __init__(self, ser, log=None, io_lock=None, read_back=True, name="device"):
        self.ser = ser
        self.log = log
        self.name = name
        self.lock = io_lock or threading.RLock()
        self.hist = collections.deque(maxlen=80)      # 我发出去的命令
        self.events = collections.deque(maxlen=200)   # 设备回给我的行
        self.stats = {"sent": 0, "failed": 0, "lines": 0}
        self._reader = None
        self._alive = True
        if read_back and ser is not None:
            self._reader = threading.Thread(target=self._read_loop,
                                            name="ActuatorReader", daemon=True)
            self._reader.start()

    # ---------- 可用性 ----------
    @property
    def usable(self):
        return self.ser is not None

    # ---------- 写 ----------
    def send(self, line):
        """发一行 ASCII 命令。返回 True/False，**不抛异常**。"""
        if self.ser is None:
            return False
        try:
            with self.lock:
                self.ser.write((line + "\n").encode("ascii", "ignore"))
                self.ser.flush()
            self.hist.append((time.time(), line))
            self.stats["sent"] += 1
            return True
        except Exception as e:
            self.stats["failed"] += 1
            if self.log:
                self.log(f"[执行器] 发送失败 {line!r}：{e}")
            return False

    # ---------- 读（设备回话）----------
    def _read_loop(self):
        """后台收设备回话。收不到也不算错 —— 有些板子就不回话。"""
        while self._alive:
            try:
                raw = self.ser.readline()
            except Exception:
                break
            if not raw:
                time.sleep(0.05)
                continue
            line = raw.decode("ascii", "replace").strip()
            if not line:
                continue
            self.events.append((time.time(), line))
            self.stats["lines"] += 1
            if self.log:
                self.log(f"[设备] {line}")

    def recent_events(self, n=4):
        """最近 n 条设备回话（给模型看一眼"设备怎么应的"）。"""
        return [ln for _, ln in list(self.events)[-n:]]

    def wait_for(self, prefix, timeout=0.6):
        """等一条**这次刚收到**的、以 prefix 开头的回话。

        为什么要比 ts >= t0：设备回话缓冲区是历史累积的，
        如果不看时间戳，第二次 PING 会立刻"匹配"到上一次那条 [PONG]，
        变成"永远都能等到" —— 一个很隐蔽的假阳性。
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            for ts, ln in reversed(list(self.events)):
                if ts >= t0 and ln.startswith(prefix):
                    return ln
            time.sleep(0.03)
        return None

    def describe(self):
        s = self.stats
        return (f"已发 {s['sent']} 条指令（失败 {s['failed']}）"
                + (f"，收到设备回话 {s['lines']} 行" if s["lines"] else ""))

    def close(self):
        self._alive = False


# ================================================================
#  工具实现
# ================================================================
def _dev(ctx):
    """从上下文里取设备连接。没接设备就返回 None（工具会回一句人话）。"""
    return (ctx or {}).get("device")


def _a_led(args, ctx):
    dev = _dev(ctx)
    if dev is None or not dev.usable:
        return "现在没接上设备，变不了灯。"

    a = args or {}
    ms = _clampi(a.get("ms") or a.get("time") or a.get("duration"),
                 200, 60000, 3000)

    # 路子一：给了颜色名
    cname = a.get("color") or a.get("colour") or a.get("颜色")
    if cname is not None and not any(k in a for k in ("r", "g", "b")):
        key = str(cname).strip().lower()
        rgb = COLOR_NAMES.get(key)
        if rgb is None:
            # 模型可能写了 "红色" 或 "red "，再宽松匹配一次
            for k, v in COLOR_NAMES.items():
                if k in key:
                    rgb = v
                    break
        if rgb is None:
            return (f"不认识颜色「{_s(cname, 12)}」。"
                    f"能用的有：红 橙 黄 绿 青 蓝 紫 白 灭。")
        r, g, b = rgb
    else:
        r = _clampi(a.get("r"), 0, 255, 0)
        g = _clampi(a.get("g"), 0, 255, 0)
        b = _clampi(a.get("b"), 0, 255, 0)

    ok = dev.send(f"LED {r} {g} {b} {ms}")
    if not ok:
        return "指令没发出去（设备断了）。"
    return f"灯环已变成 RGB({r},{g},{b})，保持 {ms / 1000:.1f} 秒后自动恢复。"


def _a_motor(args, ctx):
    dev = _dev(ctx)
    if dev is None or not dev.usable:
        return "现在没接上设备，转不了电机。"

    a = args or {}
    steps = _clampi(a.get("steps") or a.get("步数") or a.get("angle_deg"),
                    1, 200000, STEPS_PER_TURN // 2)
    # 允许模型用"圈"来表达
    if a.get("turns") is not None:
        t = a.get("turns")
        try:
            steps = _clampi(float(t) * STEPS_PER_TURN, 1, 200000, steps)
        except (TypeError, ValueError):
            pass

    d = str(a.get("dir") or a.get("direction") or a.get("方向") or "cw").lower()
    ccw = d.startswith("c") and d.startswith("cc") or d in ("逆", "逆时针", "反", "反向")
    dir_txt = "CCW" if ccw else "CW"

    if not dev.send(f"SPIN {steps} {dir_txt}"):
        return "指令没发出去（设备断了）。"
    turns = steps / float(STEPS_PER_TURN)
    return (f"电机开始{'逆' if ccw else '顺'}时针转 {steps} 步"
            f"（约 {turns:.2f} 圈），转完自动停。")


def _a_beep(args, ctx):
    dev = _dev(ctx)
    if dev is None or not dev.usable:
        return "现在没接上设备，响不了。"
    a = args or {}
    freq = _clampi(a.get("freq") or a.get("hz") or a.get("频率"),
                   50, 8000, 880)
    ms = _clampi(a.get("ms") or a.get("duration") or a.get("时长"),
                 10, 3000, 200)
    if not dev.send(f"BEEP {freq} {ms}"):
        return "指令没发出去（设备断了）。"
    return f"蜂鸣器响了：{freq}Hz，{ms}毫秒。"


def _a_text(args, ctx):
    dev = _dev(ctx)
    if dev is None or not dev.usable:
        return "现在没接上设备，写不了屏幕。"
    a = args or {}
    raw = a.get("text") if a.get("text") is not None else a.get("内容")
    raw = _s(raw, 60)
    if not raw:
        return "没给要写的字。"
    # OLED 只认 ASCII：非 ASCII 会被固件整段丢掉（encode('ascii','ignore')）。
    # 这里提前挡一道并说清楚，比让用户盯着屏幕纳闷强。
    ascii_txt = raw.encode("ascii", "ignore").decode("ascii").strip()
    if not ascii_txt:
        return f"「{raw}」全是中文，OLED 只认 ASCII 字母数字，写不上去。"
    ascii_txt = ascii_txt[:13]
    if not dev.send(f"TEXT {ascii_txt}"):
        return "指令没发出去（设备断了）。"
    return f"屏幕上已写上「{ascii_txt}」（{len(ascii_txt)} 字符）。"


def _a_stop(args, ctx):
    dev = _dev(ctx)
    if dev is None or not dev.usable:
        return "现在没接上设备，没什么可停的。"
    if not dev.send("STOP"):
        return "指令没发出去（设备断了）。"
    return "已急停：电机断电、蜂鸣器静音、灯环交还给状态动画。"


def _a_status(args, ctx):
    dev = _dev(ctx)
    if dev is None or not dev.usable:
        return "现在没接上设备。"
    if not dev.send("PING"):
        return "指令没发出去（设备断了）。"
    pong = dev.wait_for("[PONG]", timeout=0.6)
    if pong:
        return f"设备在线，当前状态：{pong.replace('[PONG]', '').strip()}。"
    # 发得出去但没回话：多半是只接了一根 TX 线，或者板子忙
    return "指令已发出，但设备没回话（可能没接回传线）。"


# ================================================================
#  安装：有设备才把执行器工具注册进清单
# ================================================================
# 这里是"延迟注册"的关键：**不写 @tool 装饰器**，改成规格表，
# 到 install() 里才真正调用 moss_tools.tool() 注册。
# 见文件开头那段 ⚠️ 说明 —— 装饰器写在顶层会导致 import 即注册，
# 那样 --no-actuators 和"没串口"都拦不住它们。
_TOOL_SPECS = (
    ("set_led",
     "把机身灯环变成指定颜色，几秒后自动恢复成状态动画。"
     "可以直接给颜色名（红/绿/蓝/青/紫/黄/橙/白/灭），也可以给 RGB 值",
     '{"color": "红"} 或 {"r":0-255,"g":0-255,"b":0-255,"ms":毫秒(默认3000)}',
     _a_led),
    ("turn_motor", "转动步进电机（机械臂/转头）。4096 步约等于一圈",
     '{"steps": 2048, "dir": "cw"|"ccw"}', _a_motor),
    ("beep", "让蜂鸣器响一声", '{"freq": 880, "ms": 200}', _a_beep),
    ("show_text",
     "在机身 OLED 屏幕上写一行字（只认 ASCII，最多 13 个字符；"
     "下次状态变化会被 MOSS 自己的台词覆盖）",
     '{"text": "HELLO"}', _a_text),
    ("stop_all", "急停：立刻停下电机、停掉蜂鸣器、把灯环还给状态动画",
     "无参数", _a_stop),
    ("device_status", "问一下设备现在是什么状态（发 PING 等设备回 PONG）",
     "无参数", _a_status),
)

ACTUATOR_NAMES = tuple(s[0] for s in _TOOL_SPECS)
_INSTALLED = False


def install(link, log=None):
    """把执行器工具注册进工具清单。没有可用设备就一条也不注册、返回空表。

    这样模型看到的清单永远和"此刻真能做到的事"一致 ——
    不会出现"清单里有、一调却失败"的假承诺。

    幂等：重复调用只会注册一次（第二次直接返回名字表），
    避免把 moss_tools 里的调用计数重置掉。
    """
    global _INSTALLED
    if link is None or not link.usable:
        if log:
            log("[执行器] 没有设备连接，执行器工具不注册（清单里不会出现）。")
        return []
    if not _INSTALLED:
        for name, desc, args, fn in _TOOL_SPECS:
            tool(name, desc, args=args)(fn)
        _INSTALLED = True
    return list(ACTUATOR_NAMES)


# ================================================================
#  自检：python moss_actuators.py
# ================================================================
class _FakeSerial:
    """假串口：把发出去的每一行记下来，并模拟设备回话。"""

    def __init__(self):
        self.lines = []
        self._echo = collections.deque()

    def write(self, b):
        line = b.decode("ascii").strip()
        self.lines.append(line)
        if line == "PING":
            self._echo.append(b"[PONG] IDLE\n")
        elif line.startswith("LED"):
            self._echo.append(b"[LED] ok\n")

    def flush(self):
        pass

    def readline(self):
        if self._echo:
            return self._echo.popleft()
        return b""


def _selftest():
    import tempfile
    from memory import MemoryStore
    from moss_tools import ToolBox

    print("=" * 66)
    print("  MOSS 执行器层自检")
    print("=" * 66)

    # ---------- [1] 没设备时：一条都不挂 ----------
    print("\n[1/4] 没设备时：清单里不该出现执行器")
    # 这一条必须**在 import 之后立刻**验，因为它防的是一个很隐蔽的坑：
    # 如果执行器工具是用 @tool 写在模块顶层，那么 import 这一下就已经注册完了，
    # 之后 install() 返不返回空表都没用 —— 工具照样出现在清单里。
    # （真实发生过：--no-actuators 形同虚设、没串口也照样列出来。）
    from moss_tools import REGISTRY
    leaked = [n for n in ACTUATOR_NAMES if n in REGISTRY]
    assert not leaked, f"import 阶段就泄漏了执行器工具：{leaked}（必须是延迟注册）"

    assert install(None) == [], "没设备却挂了执行器工具"
    assert install(type("L", (), {"usable": False})()) == []
    assert not any(n in REGISTRY for n in ACTUATOR_NAMES), "被拒之后仍不该注册"
    print("      ✓ import 后注册表里没有执行器（延迟注册成立）")
    print("      ✓ install(None) → []（不会给模型假承诺）")

    # ---------- [2] 装了设备：6 个工具逐个试 ----------
    print("\n[2/4] 接了设备：逐个工具真发指令")
    fake = _FakeSerial()
    # read_back=True：把收设备的线程也真跑起来，这样 PING → [PONG] 才算真验证
    link = ActuatorLink(fake, log=lambda s: print("      " + s), read_back=True)
    names = install(link)
    assert names == list(ACTUATOR_NAMES), names

    d = tempfile.mkdtemp(prefix="moss_act_")
    mem = MemoryStore(os.path.join(d, "m.json"), quiet=True)
    box = ToolBox(context={"device": link, "memory": mem, "mode": "IDLE"},
                  log=lambda s: print("      " + s))

    cases = [
        ("set_led", {"color": "红"}),
        ("set_led", {"r": 10, "g": 20, "b": 30, "ms": 1500}),
        ("turn_motor", {"steps": 2048, "dir": "ccw"}),
        ("turn_motor", {"turns": 0.5}),
        ("beep", {"freq": 1200, "ms": 150}),
        ("show_text", {"text": "hello moss"}),
        ("device_status", {}),
        ("stop_all", {}),
    ]
    for name, args in cases:
        ok, res = box.call(name, args)
        assert ok, f"{name} 不该失败：{res}"

    print("\n      设备实际收到的指令：")
    for ln in fake.lines:
        print(f"        > {ln}")

    # 逐条核对"我们发的"和"应该发的"
    expect = [
        "LED 255 0 0 3000",
        "LED 10 20 30 1500",
        "SPIN 2048 CCW",
        "SPIN 2048 CW",
        "BEEP 1200 150",
        "TEXT hello moss",
        "PING",
        "STOP",
    ]
    # 注意 set_led 之前 ToolBox 的 catalog 不含设备指令，所以 fake.lines 就是上面这些
    assert fake.lines == expect, f"\n  期望 {expect}\n  实际 {fake.lines}"
    print("      ✓ 8 次调用 → 8 条指令，逐字节符合固件协议")

    # ---------- [3] 参数被正确钳制 / 越界不崩 ----------
    print("\n[3/4] 烂参数：钳制 + 说人话，绝不抛异常")
    before = len(fake.lines)
    for name, args in [
        ("set_led", {"r": 999, "g": -5, "b": 300}),          # 越界 → 钳到 0..255
        ("set_led", {"color": "彩虹色"}),                     # 不认识的颜色 → 拒绝
        ("turn_motor", {"steps": 10 ** 9}),                  # 太大 → 钳到上限
        ("beep", {"freq": "很响", "ms": None}),              # 非数字 → 退默认
        ("show_text", {"text": "你好世界"}),                  # 全中文 → 说清楚
        ("show_text", {}),                                   # 空 → 拒绝
        ("不存在的执行器", {}),                                # 名字错 → 兜住
    ]:
        ok, res = box.call(name, args)
        print(f"      {name:<14} ok={str(ok):<5} {res[:60]}")

    tail = fake.lines[before:]
    # 999 / -5 / 300 → 被钳成 255 / 0 / 255
    assert tail[0] == "LED 255 0 255 3000", tail
    assert tail[1] == "SPIN 200000 CW", tail         # 10^9 被钳到上限
    assert tail[2] == "BEEP 880 200", tail           # 非数字参数退默认值
    assert "TEXT" not in " ".join(tail), "全中文 / 空文本都不该写到屏幕上"
    print("      ✓ 越界钳制、非法参数拒绝、未知工具兜底，全部安全返回")

    # ---------- [4] 没有设备的上下文：工具回人话，不是崩 ----------
    print("\n[4/4] 上下文里没有 device（没连串口就跑）")
    box2 = ToolBox(context={"memory": mem}, log=None)
    for name in ("set_led", "turn_motor", "beep", "show_text", "stop_all"):
        ok, res = box2.call(name, {} if name != "set_led" else {"color": "红"})
        assert ok and ("没接上设备" in res), f"{name} → {res}"
    print("      ✓ 五个工具都稳稳回「没接上设备」，模型不会看到 traceback")

    print("\n统计：%s" % link.describe())
    print("\n[OK] 执行器层自检通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
