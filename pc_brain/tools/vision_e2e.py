#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 18 端到端探针：从"看见一个杯子"到"说出有个杯子"
============================================================
强视觉的价值**不在检测器本身**。模型是别人训的，我们能做的只是别把
预处理和后处理写错（那部分由 `vision.py --selftest` 盯着）。
这一阶段真正的风险在**下游**：

    检测器给出带名字的框
        → VisionWorker 把它变成快照（谁算"人"？几个？）
        → fuse() 决定状态和 OLED 第三行写什么
        → ToolBox 的 look 工具把它讲成人话
        → 大模型照着这句话回答"桌上有什么"

这条链上一共四段代码，以前**一段都没法离线验证** —— 只能对着摄像头
挥挥手看它反不反应。而这个探针就是来解决这件事的。

两部分：

  A. **离线链路**（脚本化检测器，不需要摄像头、不需要模型）
     跑真的 VisionWorker、真的 fuse()、真的 ToolBox。
     检测这一段的剧本是写死的 —— 因为我们测的是**下游**，
     而目标检测本身是飘的（模型/光照/阈值都会让框忽有忽无），
     拿它当被测对象等于在流沙上盖房子。

  B. **真模型**（有模型 + 有测试图才跑）
     拿两张**真实照片**喂给真 yolov8n，断言真的认出了东西。
     这一段的期望标签是"实测记录下来的"，不是拍的脑袋。

用法：
    python tools/vision_e2e.py                # A 全跑；有模型则连 B 一起
    python tools/vision_e2e.py --no-model     # 只跑 A
"""

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # pc_brain/
sys.path.insert(0, _HERE)

import numpy as np
import cv2

import vision as V
import moss_brain as MB
from moss_tools import ToolBox


# ================================================================
#  测试夹具
# ================================================================
class StaticScene:
    """一帧都不变的画面来源。

    为什么需要它：`DemoScene` 里那个小球**一直在动**，所以它测不出
    "画面静止 → IDLE" 这条路。而"没东西在动"恰恰是最常出现、也最容易
    被物体标签搅乱的状态（一个不动的杯子不该让 MOSS 切到 MOTION）。
    """

    def __init__(self, w=320, h=240):
        self.frame = np.full((h, w, 3), 40, np.uint8)
        cv2.circle(self.frame, (w // 2, h // 2), 30, (200, 180, 160), -1)

    def read(self):
        return True, self.frame.copy()

    def release(self):
        pass


class MovingScene:
    """每读一帧都**保证**有足够像素变化 的画面来源。

    为什么不直接用 `DemoScene`：它是"动 100 帧、停 60 帧"**循环**的，
    而 `drive()` 用 `fps=0` 让视觉线程全速跑 —— 采样那一刻它完全可能
    正好落在"静止"那 60 帧里，于是这条用例**偶尔**红。

    第二版改用 `sin()` 平滑移动，**还是偶尔红** —— 因为正弦在峰值附近
    每帧只挪不到一个像素，运动量会掉到阈值 0.008 以下。
    所以这里改成**恒定步进**：每帧固定横移 18px，位移量永远一样，
    和采样时机彻底无关。步长也特意给足（半径 34 的圆挪 18px，
    变化像素约 2×34×18 = 1224，占 320×240 的 1.6%，稳稳高过 0.8% 的门槛）。

    教训：**测试要么稳定地过、要么稳定地不过；"偶尔红"最消耗人。**
    第一次是"忘了它有一段静止"，第二次是"以为平滑就是均匀" ——
    两次都是同一个毛病的不同变体。
    """

    STEP = 18

    def __init__(self, w=320, h=240):
        self.w, self.h, self.i = w, h, 0

    def read(self):
        self.i += 1
        img = np.full((self.h, self.w, 3), 40, np.uint8)
        x = 40 + (self.i * self.STEP) % (self.w - 80)
        cv2.circle(img, (x, self.h // 2), 34, (200, 180, 160), -1)
        return True, img

    def release(self):
        pass


class BrokenDetector(V.Detector):
    """每次检测都抛异常的检测器。

    为什么值得单独造一个：检测器是**外部代码**（第三方 ONNX 会话），
    它出问题的方式我们控制不了 —— 单帧花屏、内存瞬时不够、模型文件被换掉。
    视觉线程要是被一次异常带走，MOSS 就从"少了点功能"变成"瞎了"。
    所以要有一条测试专门盯着"检测器炸了会怎样"。
    """

    name = "broken"
    backend = "broken"

    def __init__(self):
        self.n = 0
        self.n_frames = 0            # drive() 靠它判断"跑够帧数了没"

    def detect(self, frame):
        self.n += 1
        self.n_frames += 1
        raise RuntimeError("模拟的检测器故障")

    def describe(self):
        return "（故意会炸的检测器）"

    def stats_line(self):
        return "被调用 %d 次，每次都炸" % self.n


def drive(detector, frames=6, fps=0, motion_th=0.008, scene=None,
          detect_every=1):
    """起一个**真的** VisionWorker，让它跑若干帧，取一份快照，然后停掉。

    为什么要真起线程而不是直接调 detector：`VisionWorker.run()` 里有三件
    容易被忽略的事 ——
      · 哪些物体算"人"（fuse 的优先级表就靠这个）
      · 检测异常被吞掉之后线程还活着
      · 检测和运动是**两个频率**（detect_every）
    这三件事只有真的把线程跑起来才测得到。
    """
    src = scene or StaticScene()
    w = MB.VisionWorker(src, detector, fps=fps, motion_th=motion_th,
                        detect_every=detect_every)
    w.start()
    # ⚠️ 等待条件要用**工作线程自己的**帧计数，不能用 `detector.n_frames`。
    # 检测器的计数是在 `detect()` **开头**就 +1 的，而工作线程的
    # `n_detect` 是它**返回之后**才 +1 的 —— 拿检测器的计数当条件，
    # 会在一次检测**跑到一半**的时候就退出等待，于是"跑了 5 帧只检测 4 次"。
    # 第一次就是这么写的，害我以为是线程挂了。
    t0 = time.time()
    while w.n_frames < frames and time.time() - t0 < 10.0:
        time.sleep(0.005)
    time.sleep(0.01)                      # 让最后一次快照写进去
    snap = w.snapshot()
    n_frames, n_detect = w.n_frames, w.n_detect
    alive = w.is_alive()
    w.shutdown()
    return snap, n_frames, n_detect, alive


def look_reply(snap, backend="fake"):
    """拿一份真快照，走真的 ToolBox，看 look 工具会怎么回答。

    `backend` 必须跟着检测器走 —— 第一版这里写死成 "fake"，
    于是在"检测器关掉"那条用例里，look 说的仍然是"没认出什么东西"。
    **夹具造假，测出来的结论就是假的**（这个项目的阶段 17 已经吃过一次）。
    """
    box = ToolBox(context={"vision": {
        "faces": int(snap["faces"]),
        "motion": float(snap["motion"]),
        "objects": list(snap["objects"]),
        "labels": snap["labels"],
        "detector": backend,
    }}, log=lambda s: None)
    _ok, res = box.call("look", {})
    return res


def _o(name, x=0.3, y=0.2, w=0.2, h=0.5, conf=0.9, cls=-1):
    return {"name": name, "cls": cls, "conf": conf, "box": (x, y, w, h)}


# ================================================================
#  A. 离线链路
# ================================================================
def part_a():
    print("=" * 74)
    print("  A. 离线链路 —— 脚本化检测器 → 真 VisionWorker → 真 fuse → 真 look")
    print("=" * 74)
    fails = []

    def check(title, cond, detail=""):
        print("  %s %s" % ("✓" if cond else "✗", title))
        if detail:
            print("      " + detail)
        if not cond:
            fails.append(title)

    # ---------------- A1 空画面 ----------------
    snap, nf, nd, alive = drive(
        V.FakeDetector([(999, [])], loop=True), frames=5)
    mode, text = MB.fuse(snap, {"listening": False, "text": "quiet"}, 0.008)
    check("A1 空画面 → IDLE / idle",
          mode == "IDLE" and text == "idle",
          "mode=%s text=%r faces=%d motion=%.4f" % (mode, text, snap["faces"],
                                                    snap["motion"]))
    check("A1 线程活着、快照在推进", alive and nf >= 5 and nd >= 5,
          "跑了 %d 帧 / 检测 %d 次" % (nf, nd))

    # ---------------- A2 有人进来 ----------------
    snap, _, _, _ = drive(
        V.FakeDetector([(999, [_o("person", 0.3, 0.15, 0.25, 0.7, 0.93, 0)])]))
    mode, text = MB.fuse(snap, {"listening": False, "text": "quiet"}, 0.008)
    check("A2 只看见人 → VISION / face x1",
          mode == "VISION" and text == "face x1",
          "mode=%s text=%r faces=%d" % (mode, text, snap["faces"]))
    check("A2 「人」被正确归到 faces 上", snap["faces"] == 1,
          "faces=%d（应 1）" % snap["faces"])

    # ---------------- A3 人 + 桌上的东西 ----------------
    objs = [_o("person", 0.28, 0.16, 0.26, 0.66, 0.93, 0),
            _o("cup", 0.62, 0.58, 0.09, 0.14, 0.72, 41),
            _o("laptop", 0.66, 0.44, 0.28, 0.26, 0.85, 63),
            _o("keyboard", 0.70, 0.70, 0.22, 0.09, 0.61, 66)]
    snap, _, _, _ = drive(V.FakeDetector([(999, objs)]))
    mode, text = MB.fuse(snap, {"listening": False, "text": "quiet"}, 0.008)
    # ⚠️ 只断言"带上了物体名"，**不**断言四个都带上：`_object_tag` 默认只塞 2 样，
    # 因为 OLED 第三行一共只有 23 个字符（固件的 gLine1[24]），
    # "face x1 + cup, keyboard" 正好 23 个。四样全塞会被设备截掉一半，
    # 那还不如不带。这里断言"前三样里至少有一个进了标签"就够了。
    check("A3 人 + 杯子/笔记本/键盘 → VISION 且带物体名",
          mode == "VISION" and text.startswith("face x1 +")
          and any(n in text for n in ("cup", "laptop", "keyboard")),
          "text=%r（%d 个字符，OLED 上限 23）" % (text, len(text)))
    check("A3 只有「人」进 faces，杯子不进",
          snap["faces"] == 1,
          "faces=%d（杯子不该算人；算错了 MOSS 会对着桌子说'我看见你了'）"
          % snap["faces"])
    reply = look_reply(snap)
    check("A3 look 能答出桌上有东西",
          "1 个杯子" in reply and "1 个人" in reply,
          "look 说：%s" % reply)

    # ---------------- A4 没人，但有东西在动 ----------------
    # 用 MovingScene（画面一直在动），检测器只看见一个杯子。
    # 这一条盯的是"物体标签**挂在哪个状态上**"——
    # 挂在 MOTION 上是对的（画面在动，动的是个杯子）；
    # 要是挂到 VISION 上就说明 faces 被非人物体污染了。
    snap, _, _, _ = drive(
        V.FakeDetector([(999, [_o("cup", 0.5, 0.5, 0.1, 0.15, 0.8, 41)])]),
        frames=8, scene=MovingScene())
    mode, text = MB.fuse(snap, {"listening": False, "text": "quiet"}, 0.008)
    check("A4 没人但有东西在动 → MOTION 且带物体名",
          mode == "MOTION" and text == "moving + cup",
          "mode=%s text=%r motion=%.4f" % (mode, text, snap["motion"]))
    check("A4 杯子**没有**被算成人", snap["faces"] == 0,
          "faces=%d（应 0）" % snap["faces"])

    # ---------------- A5 检测器炸了 ----------------
    bd = BrokenDetector()
    snap, nf, nd, alive = drive(bd, frames=6)
    mode, text = MB.fuse(snap, {"listening": False, "text": "quiet"}, 0.008)
    check("A5 检测器每帧都抛异常 → 视觉线程不倒",
          alive and nf >= 6 and bd.n >= 6,
          "线程活着=%s · 跑了 %d 帧 · 检测器被调了 %d 次（每次都炸）" % (alive, nf, bd.n))
    check("A5 异常被吞成「什么都没看见」，不是崩掉",
          mode == "IDLE" and snap["objects"] == [] and snap["faces"] == 0,
          "mode=%s objects=%d" % (mode, len(snap["objects"])))

    # ---------------- A6 检测器关掉 ----------------
    snap, _, _, _ = drive(V.NullDetector(), frames=5)
    mode, text = MB.fuse(snap, {"listening": False, "text": "quiet"}, 0.008)
    reply = look_reply(snap, backend="none")
    check("A6 检测器关闭 → 状态照常，look 说清楚「没开识别」",
          mode == "IDLE" and "没开物体识别" in reply,
          "mode=%s · look 说：%s" % (mode, reply))

    # ---------------- A7 检测频率 ----------------
    # 这一条盯的是"检测和运动是两个频率"这件事真的生效 ——
    # 默认每 3 帧检测一次。写错的话要么白烧 CPU，要么物体一闪一闪。
    _snap, nf, nd, _alive = drive(
        V.FakeDetector([(999, [_o("cup")])]), frames=9, detect_every=3)
    check("A7 detect_every=3 → 大约每 3 帧检测一次",
          abs(nd - nf / 3.0) <= 1.5 and nd >= 3,
          "跑了 %d 帧 / 检测 %d 次（应约 %d 次）" % (nf, nd, nf // 3))

    return fails


# ================================================================
#  B. 真模型 + 真实照片
# ================================================================
# 期望标签是**实测记录下来的**（见 README 的实测表），不是猜的。
# 只断言"至少认出这些"，允许多认出来 —— 同一个模型换个版本多认出一样东西
# 是好事，不该让测试红。但如果**少了**，那就是真的退化了。
ASSETS = [
    ("desk_objects.png", ["cup", "keyboard", "bottle"],
     "桌上的静物：杯子 / 键盘 / 瓶子"),
    ("person_at_desk.png", ["person", "laptop", "cup"],
     "人在桌前：人 / 笔记本电脑 / 杯子"),
]


def part_b(mdir=None):
    print()
    print("=" * 74)
    print("  B. 真模型 —— 真实照片喂给真 yolov8n")
    print("=" * 74)
    if not V.have_model("yolov8n", mdir):
        print("  跳过：模型没下载。先跑 `python vision.py --download`")
        return []
    fails = []
    det = V.YoloOnnxDetector(V.model_path("yolov8n", mdir), threads=2)

    for fname, expect, note in ASSETS:
        path = os.path.join(_HERE, "assets", fname)
        if not os.path.exists(path):
            print("  ✗ 缺测试图 %s（%s）" % (fname, note))
            fails.append(fname)
            continue
        img = cv2.imread(path)
        # 两个分辨率都跑：原图（1024）和实际工作分辨率（640×480）。
        # 后者才是**真机上发生的事** —— letterbox 的填充量完全不同，
        # 只在原图上测就等于没测坐标换算。
        for tag, frame in [("原图 %dx%d" % (img.shape[1], img.shape[0]), img),
                           ("缩到 640x480", cv2.resize(img, (640, 480)))]:
            t0 = time.time()
            objs = det.detect(frame)
            ms = (time.time() - t0) * 1000.0
            names = {o["name"] for o in objs}
            missing = [e for e in expect if e not in names]
            got = "、".join("%s %.2f" % (o["name"], o["conf"]) for o in objs)
            ok = not missing
            print("  %s %s [%s]" % ("✓" if ok else "✗", fname, tag))
            print("      认出：%s" % (got or "（空）"))
            print("      一句话：%s" % (V.summarize(objs) or "空"))
            if not ok:
                print("      少了：%s" % "、".join(missing))
                fails.append("%s/%s 少了 %s" % (fname, tag, missing))
                continue
            # 框必须在画面内 —— 这一条专门抓"逆 letterbox 算错了"。
            # 算错的时候框不会消失，只会整体偏移，肉眼很难发现，
            # 但**跑出画面外**是藏不住的。
            h, w = frame.shape[:2]
            bad = [o for o in objs
                   if o["box"][0] < 0 or o["box"][1] < 0
                   or o["box"][0] + o["box"][2] > w
                   or o["box"][1] + o["box"][3] > h]
            if bad:
                print("      ✗ 有 %d 个框跑到画面外了（逆 letterbox 有问题）" % len(bad))
                fails.append("%s/%s 框越界" % (fname, tag))
            else:
                print("      ✓ 标签齐全 · %d 个框全在画面内 · %.0f ms" % (len(objs), ms))

    det.close()
    return fails


# ================================================================
#  C. 对着一张真图跑完整个下游
# ================================================================
def part_c(mdir=None):
    print()
    print("=" * 74)
    print("  C. 真照片 → 真检测 → 真 fuse → 真 look（整条链一次走完）")
    print("=" * 74)
    if not V.have_model("yolov8n", mdir):
        print("  跳过：模型没下载。")
        return []
    path = os.path.join(_HERE, "assets", "person_at_desk.png")
    if not os.path.exists(path):
        print("  跳过：缺测试图。")
        return []

    det = V.YoloOnnxDetector(V.model_path("yolov8n", mdir), threads=2)
    snap, nf, nd, alive = drive(det, frames=5,
                                scene=MB.FileScene(path, 640, 480))
    mode, text = MB.fuse(snap, {"listening": False, "text": "quiet"}, 0.008)
    reply = look_reply(snap, backend="yolo")
    names = [o["name"] for o in snap["objects"]]
    print("  检测器：%s" % det.describe())
    print("  快照：faces=%d  objects=%d  labels=%r" %
          (snap["faces"], len(snap["objects"]), snap["labels"]))
    print("  fuse ：%s / %s" % (mode, text))
    print("  look ：%s" % reply)
    det.close()

    fails = []
    if not alive or nd < 3 or len(snap["objects"]) < 3:
        fails.append("C: 视觉线程没跑起来或没检测到东西"
                     "（alive=%s 跑了 %d 帧 检测 %d 次 认出 %d 个）"
                     % (alive, nf, nd, len(snap["objects"])))
    if snap["faces"] < 1:
        fails.append("C: 照片里明明有人，faces 却是 %d" % snap["faces"])
    if mode != "VISION":
        fails.append("C: 有人在场却没切到 VISION（实际 %s）" % mode)
    # ⚠️ 这里是拿**物体清单**断言，不是拿 look 的那句话 ——
    # look 的措辞只列前 3 类（再多 OLED 和大模型都看不过来），
    # 拿它断言"笔记本有没有被认出来"会被折叠规则干扰，
    # 测出来的是"折叠对不对"而不是"认没认出来"。
    for need in ("person", "laptop"):
        if need not in names:
            fails.append("C: 没认出 %s（实际 %s）" % (need, names))
    if "人" not in reply:
        fails.append("C: look 没把人讲出来：%s" % reply)
    print("  %s" % ("✓ 整条链跑通：照片进来，一句「%s」出去" % reply
                    if not fails else "✗ " + "；".join(fails)))
    return fails


# ================================================================
def main():
    ap = argparse.ArgumentParser(description="阶段 18 视觉端到端探针")
    ap.add_argument("--no-model", action="store_true",
                    help="只跑 A 段（不需要模型和图片）")
    ap.add_argument("--dir", default=None, help="模型目录")
    args = ap.parse_args()

    fails = part_a()
    if not args.no_model:
        fails += part_b(args.dir)
        fails += part_c(args.dir)

    print()
    print("=" * 74)
    if fails:
        print("  [失败] %d 项：" % len(fails))
        for f in fails:
            print("    · " + f)
        return 1
    print("  [OK] 阶段 18 端到端通过。")
    print("       它不只是「看到框」，而是能把框变成一句人话讲出来 ——")
    print("       而「检测器炸了会怎样」这种边界，也是在这条链上验的。")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
