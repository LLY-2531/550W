#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOSS 视觉层  ·  流浪地球 550W 复刻项目 · 阶段 18（强视觉）
============================================================
阶段 9 那会儿，"看见" == "检测到几张脸"。用的是 OpenCV 自带的 Haar 级联，
只能回答"有没有人、有几个"。你问它"画面里有什么"，它答不上来 ——
不是模型不想答，是它**从来就没看出来过**：Haar 只会认正脸，
一个杯子在它眼里和一块石头没有区别。

这一层把"看见"升级成"认出"：跑一个真实的目标检测模型，输出**带名字的框** ——
杯子 / 笔记本电脑 / 键盘 / 人 / 猫 / 瓶子 / 书 …… 一共 COCO 的 80 类。
于是 `look` 工具第一次能真的回答"画面里有什么"。

五个后端（`--vision-model`）
------------------------------------------------------------
    auto   有模型就用 yolo，没有退回 haar（默认）
    yolo   强制 YOLOv8n（ONNX）
    haar   只检测人脸 —— 阶段 9 的老行为，保底用
    null   不做任何检测，只留运动检测（最省 CPU）
    fake   脚本化的假检测器：不加载模型、不看画面，按帧号吐固定结果

为什么是 ONNX + onnxruntime，而不是 ultralytics / torch
------------------------------------------------------------
因为 **onnxruntime 已经在环境里了** —— `faster-whisper` 自带它（ASR 的 VAD 用它）。
用 ultralytics 就得把整个 PyTorch 拖进来（Windows CPU 版 200MB+，装完 2GB+）。
一个"离线、可验证、零折腾"的项目，不该为了 12MB 的模型装 2GB 的框架。

而且这事**没那么多黑盒**：真正的模型只有一张图进、一个张量出，
剩下的 letterbox / 后处理 / NMS 全是十几行 numpy + cv2，自己写反而看得更清楚。
（对比一下 ASR 那边：whisper 的胶水太多，自己写不现实，所以那边老老实实用现成的。）

模型文件从哪来
------------------------------------------------------------
**不是 pip 装的**，和 whisper 的模型一样，首次用的时候下到
`~/.cache/moss_vision/`（yolov8n 12.2MB，实测 3.4 秒）。
默认走 hf-mirror.com —— 和 asr.py 一个理由：国内直连 huggingface.co 不通
（实测 GitHub 也连不上，hf-mirror 通，980ms）。

    python vision.py --list-models      # 看有哪些、多大、下没下
    python vision.py --download         # 下默认那个
    python vision.py --image a.jpg      # 拿一张图试试

为什么 `fake` 后端是必需品，不是玩具
------------------------------------------------------------
① 目标检测本身是**飘的** —— 模型、光照、置信度阈值，都会让同一个框忽有忽无。
   拿它去验证"融合规则对不对"，等于在流沙上盖房子。
② 强视觉的价值有一半在**下游**：`fuse()` 怎么用物体标签、`look` 怎么讲、
   `BehaviorEngine` 会不会因为"看到杯子"而做事。这些逻辑跟"框是谁给的"
   毫无关系，却一直没法离线验证 —— 因为以前只能靠"对着摄像头挥挥手"。
③ 所以把"框从哪来"抽成一个接口，测试时换成脚本。这和 `behave_e2e.py`
   造人脸、`mock_esp32.py` 扮设备、`_FakeVoice` 假嗓子是**同一个思路**：
   把不确定的那一段换掉，让确定的那一段可以反复跑。

自检：
    python vision.py --selftest         # 7 组，不需要摄像头
"""

import argparse
import json
import math
import os
import sys
import time

try:
    import numpy as np
except ImportError:
    sys.exit("[缺少依赖] 没装 NumPy。请在终端执行：pip install numpy")

try:
    import cv2
except ImportError:
    sys.exit("[缺少依赖] 没装 OpenCV。请在终端执行：pip install opencv-python")

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


# ================================================================
#  0. 常量
# ================================================================

# COCO 的 80 类名字。顺序**不能动** —— 模型的输出就是按这个顺序排的，
# 换一行等于给所有东西改了名。
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# 中文名。为什么要老老实实列 80 个而不是只翻常用的几个：
# MOSS 是**用中文说话**的。模型吐的是 "cup"，直接念出来就是中英夹生；
# 而"杯子"这种词恰恰是它日常最可能提到的。少一行就是一个它答不上来的东西。
COCO_ZH = {
    "person": "人", "bicycle": "自行车", "car": "汽车", "motorcycle": "摩托车",
    "airplane": "飞机", "bus": "公交车", "train": "火车", "truck": "卡车",
    "boat": "船", "traffic light": "红绿灯", "fire hydrant": "消防栓",
    "stop sign": "停车标志", "parking meter": "停车计时器", "bench": "长椅",
    "bird": "鸟", "cat": "猫", "dog": "狗", "horse": "马", "sheep": "羊",
    "cow": "牛", "elephant": "大象", "bear": "熊", "zebra": "斑马",
    "giraffe": "长颈鹿", "backpack": "背包", "umbrella": "雨伞",
    "handbag": "手提包", "tie": "领带", "suitcase": "行李箱", "frisbee": "飞盘",
    "skis": "双板滑雪板", "snowboard": "单板滑雪板", "sports ball": "球",
    "kite": "风筝", "baseball bat": "棒球棒", "baseball glove": "棒球手套",
    "skateboard": "滑板", "surfboard": "冲浪板", "tennis racket": "网球拍",
    "bottle": "瓶子", "wine glass": "酒杯", "cup": "杯子", "fork": "叉子",
    "knife": "刀", "spoon": "勺子", "bowl": "碗", "banana": "香蕉",
    "apple": "苹果", "sandwich": "三明治", "orange": "橙子", "broccoli": "西兰花",
    "carrot": "胡萝卜", "hot dog": "热狗", "pizza": "披萨", "donut": "甜甜圈",
    "cake": "蛋糕", "chair": "椅子", "couch": "沙发", "potted plant": "盆栽",
    "bed": "床", "dining table": "餐桌", "toilet": "马桶", "tv": "电视",
    "laptop": "笔记本电脑", "mouse": "鼠标", "remote": "遥控器",
    "keyboard": "键盘", "cell phone": "手机", "microwave": "微波炉",
    "oven": "烤箱", "toaster": "烤面包机", "sink": "水槽",
    "refrigerator": "冰箱", "book": "书", "clock": "时钟", "vase": "花瓶",
    "scissors": "剪刀", "teddy bear": "泰迪熊", "hair drier": "吹风机",
    "toothbrush": "牙刷",
}

# 除了模型这 80 类，我们自己还会产出一种"类"：Haar 检出来的人脸。
# 它不在 COCO 里，但对下游（fuse 判"有人在"）是一样的意思。
FACE_ZH = "人脸"
FACE_EN = "face"

# 哪些"类"算作"有人"。fuse() 只关心这一件事（阶段 9 传下来的接口）。
PERSON_LIKE = ("person", FACE_EN)


# 量词例外表。中文里"个"是万能量词，但有几个类别用它会明显别扭
# （"1 个书"、"1 个汽车"）。表里没有的一律用"个" —— 这是**有意**的取舍：
# 给 80 类都配量词，维护成本和抄错概率都超过收益；而真正会出现在
# 桌面场景里的那二十来个，配一下是值得的（MOSS 是要**念给用户听**的）。
COCO_MEASURE = {
    FACE_EN: "张",          # Haar 后端产出的人脸，用"张"；"1 张人脸"比"1 个人脸"顺
    "book": "本", "car": "辆", "bicycle": "辆", "motorcycle": "辆",
    "bus": "辆", "truck": "辆", "train": "列", "airplane": "架",
    "bird": "只", "cat": "只", "dog": "只", "sheep": "只", "bear": "只",
    "zebra": "匹", "giraffe": "只", "horse": "匹", "cow": "头",
    "elephant": "头", "potted plant": "盆", "umbrella": "把",
    "scissors": "把", "knife": "把", "fork": "把", "spoon": "把",
    "tennis racket": "把", "baseball bat": "根", "toothbrush": "把",
    "chair": "把", "bench": "条", "tie": "条", "keyboard": "个",
    "cup": "个", "bottle": "个", "person": "个", "laptop": "台",
    "tv": "台", "cell phone": "部", "refrigerator": "台", "bed": "张",
    "dining table": "张", "couch": "张", "clock": "个",
}


# 模型登记表。`repo`/`path` 是 hf-mirror 上的位置（GitHub 连不上，实测过）。
# 只登记**真的下过、真的跑通**的型号 —— 没验证过的条目写进来就是骗自己。
MODELS = {
    "yolov8n": {
        "file": "yolov8n.onnx",
        "repo": "s1777/yolo-v8n-onnx",
        "path": "yolov8n.onnx",
        "size": 12769720,          # 正好是 Ultralytics FP32 导出的体积
        "imgsz": 640,
        "note": "COCO 80 类 · 最快 · CPU 上约 36ms/帧（本机实测）",
    },
}


# ================================================================
#  1. 图像几何：letterbox
# ================================================================
DEFAULT_COLOR = (114, 114, 114)     # YOLO 训练时的填充灰，别改


def letterbox(img, new=640, color=DEFAULT_COLOR):
    """等比缩放 + 灰边补成 new×new。返回 (图, scale, dx, dy)。

    ⚠️ 为什么必须**等比**缩放而不是直接 resize 到 640×640：
    摄像头给的是 640×480（4:3），直接拉成正方形，画面里的杯子会被横向拉长
    1/3。模型是在正常比例上训练的，遇到拉伸过的形状精度会掉 —— 这不是理论，
    "宽高比失真"是移植检测模型时最常翻的车之一。

    ⚠️ 返回的 `dx, dy` 一定要留着：后处理把框还原回原图坐标就靠它。
    少了这一步，框会整体偏移，而且**偏得不多**（几十像素），
    看起来"好像挺准"，其实是错的 —— 这种错最难发现。
    """
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("空图")
    scale = min(new / float(w), new / float(h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    if (nw, nh) != (w, h):
        # 缩小用 AREA（抗锯齿、更准），放大用 LINEAR
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        img = cv2.resize(img, (nw, nh), interpolation=interp)
    out = np.full((new, new, 3), color, dtype=np.uint8)
    dx, dy = (new - nw) // 2, (new - nh) // 2
    out[dy:dy + nh, dx:dx + nw] = img
    return out, scale, dx, dy


def to_blob(img_bgr_letterboxed):
    """BGR uint8 HWC → float32 NCHW，归一化到 0~1，并把通道翻成 RGB。

    ⚠️ 这里有两个"顺序"很容易写反，写反了模型不报错，只是**什么都检测不到**：
      1. BGR → RGB（OpenCV 读进来是 BGR，模型训练用的是 RGB）
      2. HWC → CHW（模型要的是 [N, C, H, W]）
    两个都错了会互相抵消，导致你用一张图试出来"能用"、换张图就废 ——
    所以自检里专门有一组拿**已知颜色**验通道顺序。
    """
    x = img_bgr_letterboxed[:, :, ::-1]                    # BGR → RGB
    x = np.ascontiguousarray(x.transpose(2, 0, 1))         # HWC → CHW
    x = x.astype(np.float32) / 255.0
    return x[None]                                         # → [1, 3, H, W]


# ================================================================
#  2. 后处理：一张张量 → 一串带名字的框
# ================================================================
# 类别感知 NMS 用的偏移量。把不同类的框在坐标上推开，NMS 就不会跨类合并
# （否则"人手里拿着杯子"会只剩一个框）。取一个大于任何图片尺寸的数即可。
_NMS_CLASS_OFFSET = 7680.0


def _coord_convention(r):
    """这份输出用的是哪种坐标约定：`"pixel"` 还是 `"normalized"`。

    Ultralytics **自己**导出的 ONNX 给的是输入**像素**坐标（0~imgsz）。
    但这个约定并不是 ONNX 规范的一部分 —— 有些第三方导出（包括本机默认
    下载的这个 yolov8n，见 MODELS 表）会顺手除一个 imgsz，变成**归一化**坐标。

    两种都合理，但搞错了就是两回事，而且**不会报错**：
      · 归一化坐标被当像素用 → 框全挤在左上角几个像素里，看起来"框歪了"
      · 更糟的是反过来，配合"丢掉退化框"那一步 → 所有框都被当成碎片丢掉 →
        表现成**"什么都没检测到"**
    第二种我第一次就撞上了：拿一张明明有笔记本的照片跑，输出空的，
    第一反应是"这模型不行"，其实是解码规矩搞错了 —— 冤枉了模型半天。

    怎么判断：8400 个候选框是**铺满整张图**的，所以
      · 像素坐标：最大值必然上百（几百）
      · 归一化坐标：最大值必然 ≤1（顶多因填充略微超一点）
    两边差两个数量级，取 2.0 当分界，不可能判错。
    """
    m = float(max(r[:, 0].max(), r[:, 1].max(), r[:, 2].max(), r[:, 3].max()))
    return "normalized" if m <= 2.0 else "pixel"


def postprocess(raw, scale, dx, dy, img_w, img_h, conf_th=0.35, iou_th=0.45,
                max_obj=12, min_box=8, imgsz=640, coord_scale=None):
    """YOLOv8 的原始输出 → 框列表。

    输入 `raw` 形状 `[1, 84, 8400]`：84 = 4 个框坐标 + 80 个类别分数，
    8400 = 三个尺度的网格点之和（80²+40²+20²）。这是 Ultralytics 导出 ONNX
    时的**默认**布局，也是它最容易被写错的地方（见下面那条警告）。

    返回 `[{"cls":int, "name":str, "conf":float, "box":(x,y,w,h)}, ...]`，
    坐标已经还原到**原图**像素坐标系。

    `coord_scale` 不传就按 `imgsz` 自动判（`_coord_convention`）；
    判错了会静默失败，所以自检里两种约定各测一遍。
    """
    if raw is None or raw.ndim != 3:
        return []
    r = raw[0]
    # ⚠️ 布局是 [属性, 候选框] 而不是 [候选框, 属性] —— 要转置。
    # 忘了转置的话形状变成 (84, 8400) 按行读，代码照样跑得动、
    # 只是框全乱。判断依据：候选框数（8400）一定远大于属性数（84）。
    if r.shape[0] < r.shape[1]:
        r = r.T                                            # → [8400, 84]
    if r.shape[1] < 5:
        return []

    n_cls = r.shape[1] - 4
    scores = r[:, 4:4 + n_cls]

    cls = np.argmax(scores, axis=1).astype(np.int32)
    conf = scores[np.arange(scores.shape[0]), cls]

    keep = conf >= conf_th
    idx = np.nonzero(keep)[0]
    if idx.size == 0:
        return []

    # ---- 坐标约定：像素空间 or 归一化空间（见 _coord_convention）----
    if coord_scale is None:
        coord_scale = float(imgsz) if _coord_convention(r) == "normalized" else 1.0
    b = np.stack([r[idx, 0], r[idx, 1], r[idx, 2], r[idx, 3]],
                 axis=1).astype(np.float64) * coord_scale
    c = cls[idx]
    f = conf[idx]

    # ⚠️ 丢掉"退化框"（宽或高接近 0，会让 NMS 的面积判断出错）这一步，
    # 必须在坐标换算**之后**做。归一化空间里所有宽高都 ≤1，
    # 先筛的话会把**全部**好框当成碎片扔掉 —— 症状就是"什么都没检测到"。
    okb = (b[:, 2] > 1.0) & (b[:, 3] > 1.0)
    if not np.any(okb):
        return []
    b, c, f = b[okb], c[okb], f[okb]

    # ---- cxcywh（letterbox 坐标）→ 左上角 xywh ----
    tl = np.empty_like(b)
    tl[:, 0] = b[:, 0] - b[:, 2] * 0.5
    tl[:, 1] = b[:, 1] - b[:, 3] * 0.5
    tl[:, 2] = b[:, 2]
    tl[:, 3] = b[:, 3]

    # ---- 逆 letterbox：减掉填充、除掉缩放 ----
    tl[:, 0] = (tl[:, 0] - dx) / scale
    tl[:, 1] = (tl[:, 1] - dy) / scale
    tl[:, 2] = tl[:, 2] / scale
    tl[:, 3] = tl[:, 3] / scale

    # ---- 裁到画面内，并去掉贴着边角的碎片 ----
    x1 = np.clip(tl[:, 0], 0.0, float(img_w))
    y1 = np.clip(tl[:, 1], 0.0, float(img_h))
    x2 = np.clip(tl[:, 0] + tl[:, 2], 0.0, float(img_w))
    y2 = np.clip(tl[:, 1] + tl[:, 3], 0.0, float(img_h))
    w = x2 - x1
    h = y2 - y1
    ok = (w >= min_box) & (h >= min_box)
    if not np.any(ok):
        return []
    x1, y1, w, h, c, f = x1[ok], y1[ok], w[ok], h[ok], c[ok], f[ok]

    # ---- 类别感知 NMS ----
    off = c.astype(np.float64) * _NMS_CLASS_OFFSET
    nms_in = np.stack([x1 + off, y1 + off, w, h], axis=1)
    try:
        sel = cv2.dnn.NMSBoxes(nms_in.tolist(), f.tolist(),
                               float(conf_th), float(iou_th))
    except cv2.error:
        sel = None
    if sel is None or len(sel) == 0:
        return []
    # cv2 不同版本返回的形状不一样（tuple / (N,1) 数组），统一压平
    sel = np.asarray(sel).reshape(-1).astype(int)
    if sel.size == 0:
        return []

    order = sel[np.argsort(-f[sel])]                       # 置信度高的在前
    if max_obj and order.size > max_obj:
        order = order[:max_obj]

    out = []
    for i in order:
        ci = int(c[i])
        name = COCO_NAMES[ci] if 0 <= ci < len(COCO_NAMES) else "obj%d" % ci
        out.append({
            "cls": ci,
            "name": name,
            "conf": round(float(f[i]), 4),
            # 取整再返回：下游要画框、要算面积，整数比浮点省心
            "box": (int(round(x1[i])), int(round(y1[i])),
                    int(round(w[i])), int(round(h[i]))),
        })
    return out


# ================================================================
#  3. 说人话
# ================================================================
def zh_name(name):
    """英文类名 → 中文。手写的那张表里没有就退回英文（宁可夹生，也别瞎猜）。"""
    # ⚠️ "face" 不在 COCO 的 80 类里（它是 Haar 后端的产物），
    # 单独判一下。否则 Haar 模式下 summarize() 会说出"1 个face"这种话 ——
    # 而这个工具的输出是要被大模型直接念出来的。
    if name == FACE_EN:
        return FACE_ZH
    return COCO_ZH.get(name, name)


def count_by_name(objs):
    """按类名归并计数，返回 `[(name, zh, n), ...]`，按数量多的在前。"""
    tally = {}
    for o in objs:
        n = o.get("name") or FACE_EN
        tally[n] = tally.get(n, 0) + 1
    rows = [(n, zh_name(n), c) for n, c in tally.items()]
    rows.sort(key=lambda r: (-r[2], r[1]))
    return rows


def summarize(objs, max_n=3):
    """把框列表压成一句人话：`2 个杯子、1 个人`。

    `max_n` 只限制**种类**数，不限制个数 —— "3 个人"比"人"信息量大得多，
    而列 12 种东西只会让人（和大模型）都懒得看。
    多出来的种类收在"等 N 样东西"里：这是中文里"列举未完 + 总数"的标准说法，
    既不丢信息也不啰嗦。
    """
    if not objs:
        return ""
    rows = count_by_name(objs)
    parts = []
    for name, zh, c in rows[:max_n]:
        parts.append("%d %s%s" % (c, COCO_MEASURE.get(name, "个"), zh))
    head = "、".join(parts)
    if len(rows) > max_n:
        # 注意是 `…杯子等 5 样东西`，中间**不加顿号** ——
        # 加了会读成"还有第 4 样叫'等 5 样东西'"，第一版就是这么写的。
        head += "等 %d 样东西" % len(rows)
    return head


def labels_line(objs, max_n=3):
    """给模型/日志看的短标签：`cup x2, person x1`（英文类名，模型更熟）。"""
    rows = count_by_name(objs)
    return ", ".join("%s x%d" % (n, c) for n, _zh, c in rows[:max_n])


# ================================================================
#  4. 检测器
# ================================================================
class Detector:
    """检测器接口。四个实现共用它。

    为什么要抽这一层：主程序（`VisionWorker`）只该关心"给我一批带名字的框"，
    不该关心"这批框是 Haar 出的、YOLO 出的、还是脚本写的"。
    抽出来之后，测试可以换成脚本，真机可以换模型，主程序一行不用动。
    """
    name = "null"
    backend = "null"

    def detect(self, frame):
        return []

    def describe(self):
        return self.name

    def stats_line(self):
        return ""

    def close(self):
        pass


class NullDetector(Detector):
    """什么都不检测。只想看运动、或者在省电的时候用。"""

    def __init__(self, reason=""):
        self.name = "null"
        self.reason = reason
        self.n_frames = 0

    def detect(self, frame):
        self.n_frames += 1
        return []

    def describe(self):
        return "关闭（只做运动检测）" + (("：" + self.reason) if self.reason else "")


class HaarDetector(Detector):
    """阶段 9 的老办法：Haar 级联，只认正脸。

    留着它不是为了怀旧，是因为它有两个真实优势：
      · **零额外文件** —— 模型是 OpenCV 自带的 xml，装完就有
      · **极快** —— 640×480 灰度图上 5~8ms，比 YOLO 便宜五倍
    在没有 YOLO 模型、或者在很弱的机器上，它仍然是对的答案。
    """

    name = "haar"
    backend = "haar"

    def __init__(self, min_size=(48, 48)):
        import os as _os
        path = _os.path.join(cv2.data.haarcascades,
                             "haarcascade_frontalface_default.xml")
        if not _os.path.exists(path):
            # opencv-python 5.0 起不再自带 xml —— requirements 里锁了 <5 就是为了它
            raise RuntimeError("找不到 haarcascade xml（OpenCV 5.0+ 不再自带，请用 4.x）")
        self.cascade = cv2.CascadeClassifier(path)
        self.min_size = tuple(min_size)
        self.n_frames = 0
        self.n_obj = 0

    def detect(self, frame):
        self.n_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=5, minSize=self.min_size)
        out = [{"cls": -1, "name": FACE_EN, "conf": 1.0,
                "box": (int(x), int(y), int(w), int(h))} for (x, y, w, h) in faces]
        self.n_obj += len(out)
        return out

    def describe(self):
        return "Haar 人脸级联（只认正脸 · 零额外文件 · 约 6ms/帧）"


class YoloOnnxDetector(Detector):
    """YOLOv8n（ONNX Runtime）。真正的"认出东西"。

    ⚠️ 线程数默认给 2 而不是"全给"：
    这个程序里还有**实时音频**线程（AEC 要求 64ms 一块不能拖）。
    ONNX 默认会吃掉所有核，跑满 36ms 一帧的同时把音频线程挤到掉块。
    视觉慢一点没关系（人不会因为画面 15fps 而觉得卡），
    音频掉一块就是"漏听半句话"。
    """

    name = "yolo"
    backend = "yolo"

    def __init__(self, path, imgsz=640, conf_th=0.35, iou_th=0.45,
                 max_obj=12, threads=2, min_box=8):
        if not HAS_ORT:
            raise RuntimeError("没装 onnxruntime（pip install onnxruntime）")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads and threads > 0:
            so.intra_op_num_threads = int(threads)
            so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, so,
                                         providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.imgsz = int(imgsz)
        self.conf_th = float(conf_th)
        self.iou_th = float(iou_th)
        self.max_obj = int(max_obj)
        self.min_box = int(min_box)
        self.path = path
        self.n_frames = 0
        self.n_obj = 0
        self.ms_sum = 0.0
        self.coord_conv = None      # 第一次推理后才知道，见 _coord_convention

    def detect(self, frame):
        self.n_frames += 1
        t0 = time.time()
        img, scale, dx, dy = letterbox(frame, self.imgsz)
        blob = to_blob(img)
        h, w = frame.shape[:2]
        try:
            raw = self.sess.run(None, {self.in_name: blob})[0]
        except Exception:
            # 单帧推理失败不能把整条视觉线程带走 —— 摄像头偶发花屏、
            # 内存瞬时不够都可能。返回空、记一笔，下一帧继续。
            return []
        objs = postprocess(raw, scale, dx, dy, w, h,
                           conf_th=self.conf_th, iou_th=self.iou_th,
                           max_obj=self.max_obj, min_box=self.min_box,
                           imgsz=self.imgsz)
        if self.coord_conv is None and raw is not None and raw.ndim == 3:
            r = raw[0]
            self.coord_conv = _coord_convention(r.T if r.shape[0] < r.shape[1] else r)
        self.ms_sum += (time.time() - t0) * 1000.0
        self.n_obj += len(objs)
        return objs

    def describe(self):
        conv = {"pixel": "像素坐标", "normalized": "归一化坐标"}.get(
            self.coord_conv, "坐标约定待定")
        return "yolov8n（COCO 80 类 · ONNX · %d×%d · CPU %d 线程 · %.0fms/帧 · %s）" % (
            self.imgsz, self.imgsz,
            self.sess._sess_options.intra_op_num_threads, self.avg_ms, conv)

    @property
    def avg_ms(self):
        return (self.ms_sum / self.n_frames) if self.n_frames else 0.0

    def stats_line(self):
        if not self.n_frames:
            return ""
        return "目标检测 %d 帧 · 平均 %.1f ms · 共 %d 个目标 · 每次 %.1f 个" % (
            self.n_frames, self.avg_ms, self.n_obj, self.n_obj / float(self.n_frames))


class FakeDetector(Detector):
    """脚本化检测器：不看画面，按帧号推进一个固定剧本。

    剧本是 `[(持续帧数, [objs]), ...]` 的列表，放完循环（除非 loop=False）。
    `objs` 里的框写的是**归一化坐标**（0~1），跟画面尺寸解耦 ——
    这样换个分辨率脚本不用改。

    见文件头"为什么 fake 后端是必需品"。核心一句：
    **它测的不是检测器，是检测器下游的所有东西。**
    """

    name = "fake"
    backend = "fake"

    def __init__(self, script=None, loop=True):
        self.script = list(script) if script else default_script()
        self.loop = loop
        self.i = 0                 # 总帧号
        self.n_frames = 0
        self.n_obj = 0

    def _phase(self):
        total = sum(n for n, _ in self.script) or 1
        k = self.i % total if self.loop else min(self.i, total - 1)
        acc = 0
        for n, objs in self.script:
            acc += n
            if k < acc:
                return objs
        return []

    def detect(self, frame):
        self.n_frames += 1
        h, w = frame.shape[:2]
        out = []
        for o in self._phase():
            nx, ny, nw, nh = o["box"]
            out.append({
                "cls": o.get("cls", -1),
                "name": o["name"],
                "conf": o.get("conf", 0.9),
                "box": (int(nx * w), int(ny * h), int(nw * w), int(nh * h)),
            })
        self.i += 1
        self.n_obj += len(out)
        return out

    def describe(self):
        return "脚本（fake · %d 段剧本 · 循环=%s）—— 给离线测试用的假检测器" % (
            len(self.script), self.loop)

    def stats_line(self):
        return "假检测器推进 %d 帧" % self.n_frames


def _o(name, x, y, w, h, conf=0.9, cls=-1):
    return {"name": name, "cls": cls, "conf": conf, "box": (x, y, w, h)}


def default_script():
    """默认剧本：安静 → 有人进来 → 人 + 桌上的杯子/笔记本/键盘 → 人走。

    帧数按"视觉线程 30Hz"估的：每段 40~90 帧 ≈ 1.5~3 秒，
    正好够主循环 20Hz × 防抖 4 次确认一个状态。
    """
    return [
        (40, []),
        (60, [_o("person", 0.30, 0.18, 0.24, 0.62, 0.91, 0)]),
        (90, [_o("person", 0.28, 0.16, 0.26, 0.66, 0.93, 0),
              _o("cup", 0.62, 0.58, 0.09, 0.14, 0.72, 41),
              _o("laptop", 0.66, 0.44, 0.28, 0.26, 0.85, 63),
              _o("keyboard", 0.70, 0.70, 0.22, 0.09, 0.61, 66)]),
        (40, [_o("person", 0.34, 0.20, 0.22, 0.60, 0.88, 0)]),
        (50, []),
    ]


# ================================================================
#  5. 工厂 + 模型文件管理
# ================================================================
def model_dir():
    """模型放哪。和 asr.py 的 `~/.cache/moss_asr/` 一个路子。"""
    d = os.environ.get("MOSS_VISION_DIR")
    if d:
        return d
    return os.path.join(os.path.expanduser("~"), ".cache", "moss_vision")


def model_path(spec, mdir=None):
    entry = MODELS.get(spec)
    fname = entry["file"] if entry else spec
    return os.path.join(mdir or model_dir(), fname)


def have_model(spec, mdir=None):
    p = model_path(spec, mdir)
    return os.path.exists(p) and os.path.getsize(p) > 1024 * 1024


def download_model(spec="yolov8n", mdir=None, mirror="hf-mirror.com",
                   quiet=False, force=False):
    """从镜像下模型。返回本地路径。

    为什么默认走 hf-mirror：和 asr.py 同样的原因 —— 国内直连
    huggingface.co 不通（本机实测 GitHub 也不通，hf-mirror 通）。
    """
    import urllib.request

    entry = MODELS.get(spec)
    if not entry:
        raise KeyError("没登记这个模型：%s（可选：%s）"
                       % (spec, ", ".join(MODELS)))
    dst = os.path.join(mdir or model_dir(), entry["file"])
    if os.path.exists(dst) and not force and os.path.getsize(dst) > 1024 * 1024:
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = "https://%s/%s/resolve/main/%s" % (mirror, entry["repo"], entry["path"])
    if not quiet:
        print("[下载] %s（约 %.1f MB）" % (url, entry["size"] / 1048576.0),
              file=sys.stderr)
    tmp = dst + ".part"
    t0 = time.time()
    try:
        urllib.request.urlretrieve(url, tmp)
        if os.path.getsize(tmp) < 1024 * 1024:
            raise IOError("下下来的文件太小，可能被镜像挡了")
        os.replace(tmp, dst)          # 先写 .part 再改名：下到一半失败不会留个坏模型
    except Exception as e:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise RuntimeError("下载失败（%s）：%s\n"
                           "  可以手动下好放到 %s\n"
                           "  或者 --vision-mirror 换一个镜像" % (url, e, dst))
    if not quiet:
        print("[OK] %.1f MB 下好了（%.1fs）→ %s"
              % (os.path.getsize(dst) / 1048576.0, time.time() - t0, dst),
              file=sys.stderr)
    return dst


def make_detector(spec="auto", mdir=None, mirror="hf-mirror.com", **kw):
    """造一个检测器。返回 `(detector, 说明文字)`。

    ⚠️ 这个函数**不抛异常**（除了参数写错）。理由是它在主程序启动路径上：
    模型没下、onnxruntime 没装、文件坏了 —— 都不该让整个 MOSS 起不来。
    视觉拿不到东西，听觉和对话还能用；退化成 Haar 或者干脆不做检测，
    都比"启动失败"好。所以每一步失败都降级 + 把原因带回去给人看。
    """
    spec = (spec or "auto").lower()
    notes = []

    if spec == "null":
        return NullDetector("手动指定 --vision-model null"), "关闭"

    if spec == "fake":
        return FakeDetector(**{k: v for k, v in kw.items()
                               if k in ("script", "loop")}), None

    if spec == "haar":
        try:
            return HaarDetector(), None
        except Exception as e:
            notes.append("Haar 起不来（%s）" % e)
            return NullDetector(notes[-1]), notes[-1]

    # ---- yolo / auto ----
    if spec in ("yolo", "auto"):
        if not HAS_ORT:
            notes.append("没装 onnxruntime")
        else:
            path = kw.pop("model_path", None) or model_path("yolov8n", mdir)
            if not os.path.exists(path):
                if spec == "auto":
                    # auto 不该替用户偷偷下 12MB —— 悄悄占带宽和磁盘是不礼貌的。
                    # 想用就明确说一句（--vision-download 或 --vision-model yolo）。
                    notes.append("没下模型（python vision.py --download）")
                else:
                    try:
                        path = download_model("yolov8n", mdir, mirror)
                    except Exception as e:
                        notes.append(str(e).splitlines()[0])
            if os.path.exists(path):
                try:
                    return YoloOnnxDetector(path, **kw), None
                except Exception as e:
                    notes.append("加载失败（%s）" % e)

    # ---- 兜底：auto 退 haar ----
    if spec == "auto":
        try:
            d = HaarDetector()
            return d, "没走成 YOLO（%s）→ 退回 Haar 人脸" % "；".join(notes)
        except Exception as e:
            notes.append("Haar 也不行（%s）" % e)

    return NullDetector("；".join(notes)), "；".join(notes)


def list_models(mdir=None):
    """`--list-models` 用。返回可打印的行。"""
    md = mdir or model_dir()
    lines = ["模型目录：%s" % md, ""]
    for spec in sorted(MODELS):
        e = MODELS[spec]
        p = os.path.join(md, e["file"])
        ok = os.path.exists(p)
        lines.append("  %-10s %s" % (spec, "已下载" if ok else "未下载"))
        lines.append("      %s" % e["note"])
        lines.append("      %.1f MB  %s/%s"
                     % (e["size"] / 1048576.0, e["repo"], e["path"]))
        if ok:
            lines.append("      本地：%s（%.1f MB）"
                         % (p, os.path.getsize(p) / 1048576.0))
        lines.append("")
    lines.append("下载：python vision.py --download [--model %s]"
                 % "|".join(sorted(MODELS)))
    return "\n".join(lines)


# ================================================================
#  6. 自检
# ================================================================
def _check_letterbox_roundtrip():
    """letterbox 的几何必须是**可逆**的。

    这是整个模块最容易"看起来对、其实错"的地方：填充偏移量 `dx/dy`
    算错几个像素，框就会整体平移，而模型照样能检出东西 ——
    你不会看到报错，只会觉得"框画得有点歪"。
    所以这里正着做一遍、反着做一遍，用**像素级**断言卡住它。
    """
    # 4:3 灰度图，左边一条白线（好定位）
    img = np.zeros((480, 640, 3), np.uint8)
    img[:, 0:4] = 255

    out, scale, dx, dy = letterbox(img, 640)
    assert out.shape == (640, 640, 3), "输出尺寸不对：%s" % (out.shape,)
    assert abs(scale - 1.0) < 1e-6, "640×480 缩放到 640×640，scale 应该正好 1.0"

    # 反向：把原图里一个点映射过去再映射回来，必须回到原处
    for (px, py) in [(0, 0), (639, 479), (320, 240), (100, 33)]:
        lx = px * scale + dx
        ly = py * scale + dy
        bx = (lx - dx) / scale
        by = (ly - dy) / scale
        assert abs(bx - px) < 1e-6 and abs(by - py) < 1e-6, \
            "点 (%d,%d) 来回一趟变成了 (%.3f,%.3f)" % (px, py, bx, by)

    # 填充确实是"居中 + 灰边"：480 高补到 640，上下各 80
    assert dy == 80 and dx == 0, "填充量不对：dx=%d dy=%d（应为 0,80）" % (dx, dy)
    assert tuple(out[0, 320]) == DEFAULT_COLOR, "上边应该是填充灰"

    # 换成竖图（3:4）再验一次，确保不是碰巧对
    img2 = np.zeros((640, 480, 3), np.uint8)
    out2, s2, dx2, dy2 = letterbox(img2, 640)
    assert out2.shape == (640, 640, 3) and dx2 == 80 and dy2 == 0, \
        "竖图填充不对：dx=%d dy=%d" % (dx2, dy2)
    return "640×480→dx0/dy80 · 480×640→dx80/dy0 · 四个点来回无误差"


def _check_channel_order():
    """通道顺序必须真的是 RGB。

    怎么验：造一张**纯色**图（BGR = 0,0,255 即纯红），送进 `to_blob`，
    吐出来的三个通道里**只有第一个**该是满值 1.0。
    如果 BGR/RGB 写反了，满的会变成第三个 —— 模型不会报错，
    只是会以为你拍的是蓝色，然后什么都检测不到。
    """
    img = np.zeros((32, 32, 3), np.uint8)
    img[:, :] = (0, 0, 255)                      # OpenCV 里这是纯红
    blob = to_blob(img)
    assert blob.shape == (1, 3, 32, 32), "blob 形状不对：%s" % (blob.shape,)
    assert blob.dtype == np.float32, "dtype 应该是 float32"
    means = [float(blob[0, c].mean()) for c in range(3)]
    assert abs(means[0] - 1.0) < 1e-6, "R 通道应该是满值，实测 %.3f" % means[0]
    assert means[1] < 1e-6 and means[2] < 1e-6, \
        "G/B 通道应该是 0，实测 %.3f / %.3f" % (means[1], means[2])
    assert 0.0 <= float(blob.min()) and float(blob.max()) <= 1.0, "没归一化到 0~1"
    return "BGR(0,0,255) → blob 只点亮第 1 通道（R）· 形状 [1,3,32,32]"


def _fake_raw(entries, n_cand=8400, n_cls=80):
    """按 YOLOv8 的真实布局 `[1,84,8400]` 造一份原始输出。

    用**真的**形状而不是"随便给个数组"，是为了顺便盯死转置那一步：
    转置写错在后处理里是静默的，只有用真形状才拦得住。
    """
    raw = np.zeros((1, 4 + n_cls, n_cand), np.float32)
    for k, (cid, cx, cy, w, h, sc) in enumerate(entries):
        raw[0, 0, k] = cx
        raw[0, 1, k] = cy
        raw[0, 2, k] = w
        raw[0, 3, k] = h
        raw[0, 4 + cid, k] = sc
    return raw


def _check_postprocess():
    """后处理：已知的原始输出 → 已知的框。"""
    # scale=1.0、无填充：让坐标换算退化成恒等，先把"解码"这一步单独验干净
    raw = _fake_raw([
        (41, 100.0, 200.0, 40.0, 60.0, 0.90),      # cup，中心 (100,200)
        (0, 400.0, 300.0, 80.0, 200.0, 0.85),      # person
        (63, 320.0, 320.0, 160.0, 100.0, 0.10),    # laptop，分数低于阈值
    ])
    objs = postprocess(raw, 1.0, 0, 0, 640, 480, conf_th=0.35, iou_th=0.45)
    names = sorted(o["name"] for o in objs)
    assert names == ["cup", "person"], "应该只留下 cup 和 person，实际 %s" % names

    cup = [o for o in objs if o["name"] == "cup"][0]
    # cxcywh(100,200,40,60) → 左上角 xywh(80,170,40,60)
    assert cup["box"] == (80, 170, 40, 60), "杯子框不对：%s（应为 (80,170,40,60)）" \
        % (cup["box"],)
    assert abs(cup["conf"] - 0.90) < 1e-4, "置信度没取对"
    assert cup["cls"] == 41, "类别号不对：%d" % cup["cls"]

    # 带上 letterbox 参数再验一次（这次是真有填充和缩放的情形）。
    # 手算一遍，别凭感觉写期望值（第一版就是只除了 x 没除 y，差点赖到代码头上）：
    #   原始 cxcywh = (100, 200, 40, 60)，letterbox 坐标下左上角 = (80, 170)
    #   逆变换 = (v - d) / scale，scale=0.75、dx=40、dy=0：
    #     x = (80 - 40) / 0.75 = 53.33 → 53
    #     y = (170 - 0) / 0.75 = 226.67 → 227     ← y 也要除 scale
    #     w = 40 / 0.75 = 53.33 → 53
    #     h = 60 / 0.75 = 80.0  → 80
    objs2 = postprocess(raw, 0.75, 40.0, 0.0, 640, 480, conf_th=0.35)
    cup2 = [o for o in objs2 if o["name"] == "cup"][0]
    assert cup2["box"] == (53, 227, 53, 80), \
        "逆 letterbox 算错了：%s（应为 (53,227,53,80)）" % (cup2["box"],)

    # 空输入 / 垃圾输入不能炸
    assert postprocess(None, 1.0, 0, 0, 640, 480) == []
    assert postprocess(np.zeros((1, 4, 10), np.float32), 1.0, 0, 0, 640, 480) == []

    # ---- 坐标约定：这一条是**真机上撞出来的**，必须盯死 ----
    # 有些 ONNX 导出把坐标归一化到 0~1（本机默认下载的那个就是），
    # 有些给输入像素。搞错了不会报错：归一化坐标被当成像素用，
    # 框会全挤进左上角几个像素，"丢掉退化框"那一步再把它们全扔了 ——
    # 表现成"什么都没检测到"，看着特别像"模型不行"。
    nraw = _fake_raw([
        (41, 0.5, 0.5, 0.1, 0.1, 0.9),             # 归一化：中心 (0.5,0.5)
    ])
    objs3 = postprocess(nraw, 1.0, 0, 0, 640, 480, conf_th=0.35, imgsz=640)
    assert len(objs3) == 1, "归一化坐标的框被整批丢掉了（这正是真机上的症状）"
    # 0.5*640=320，w=0.1*640=64 → 左上角 (288,288)
    assert objs3[0]["box"] == (288, 288, 64, 64), \
        "归一化坐标换算不对：%s（应为 (288,288,64,64)）" % (objs3[0]["box"],)
    assert _coord_convention(nraw[0].T) == "normalized", "没认出归一化坐标"
    assert _coord_convention(raw[0].T) == "pixel", "把像素坐标误判成归一化"
    # 显式指定必须能盖掉自动判断
    objs4 = postprocess(nraw, 1.0, 0, 0, 640, 480, conf_th=0.35, coord_scale=1.0)
    assert objs4 == [], "显式给了 coord_scale 却没生效（0.1 像素的框该被当碎片丢掉）"
    return "cxcywh→xywh · 阈值 · 逆 letterbox · **归一化/像素两种坐标约定** · 空输入不炸"


def _check_nms():
    """NMS：同一个东西的重复框要合并，**不同类**的重叠框要都留。"""
    # 两个几乎重合的 cup（同一物体），一个远处的 person
    raw = _fake_raw([
        (41, 100.0, 200.0, 40.0, 60.0, 0.95),
        (41, 104.0, 203.0, 41.0, 58.0, 0.80),      # 与上面 IoU 很高 → 该被压掉
        (0, 400.0, 300.0, 80.0, 200.0, 0.88),
    ])
    objs = postprocess(raw, 1.0, 0, 0, 640, 480, conf_th=0.35, iou_th=0.45)
    assert len(objs) == 2, "重复框没被压掉，剩了 %d 个" % len(objs)
    cups = [o for o in objs if o["name"] == "cup"]
    assert len(cups) == 1, "同一个杯子应该只留一个框"
    assert abs(cups[0]["conf"] - 0.95) < 1e-4, "留下的应该是分高的那个"

    # 类别感知：同一个位置、不同类别，两个都得留
    raw = _fake_raw([
        (41, 300.0, 300.0, 60.0, 60.0, 0.90),      # cup
        (0, 300.0, 300.0, 60.0, 60.0, 0.90),       # person，框完全一样
    ])
    objs = postprocess(raw, 1.0, 0, 0, 640, 480, conf_th=0.35, iou_th=0.45)
    got = sorted(o["name"] for o in objs)
    assert got == ["cup", "person"], \
        "跨类被误合并了（实际 %s）—— 人手里拿着杯子时会只剩一个框" % got

    # max_obj 限制
    many = [(i % 80, 50.0 + i * 40, 100.0, 30.0, 30.0, 0.9) for i in range(20)]
    objs = postprocess(_fake_raw(many), 1.0, 0, 0, 2000, 480,
                       conf_th=0.35, iou_th=0.45, max_obj=5)
    assert len(objs) == 5, "max_obj 没生效，返回了 %d 个" % len(objs)
    return "重复框被压 · 跨类不误合并 · max_obj 生效"


def _check_coco_table():
    """80 类表必须完整、无重复、中英一一对应。

    这张表是**手抄**的，抄漏一行不会报错 —— 只在某个类被真的检测到时
    才会表现为"名字是 obj37"。所以用断言把它钉住。
    """
    assert len(COCO_NAMES) == 80, "COCO 应该是 80 类，实际 %d" % len(COCO_NAMES)
    assert len(set(COCO_NAMES)) == 80, "COCO 表里有重复的名字"
    missing = [n for n in COCO_NAMES if n not in COCO_ZH]
    assert not missing, "这些类没有中文名：%s" % missing
    extra = [k for k in COCO_ZH if k not in COCO_NAMES]
    assert not extra, "中文表里有 COCO 里不存在的类：%s" % extra
    # 抽查几个 MOSS 最可能提到的
    for en, zh in [("cup", "杯子"), ("laptop", "笔记本电脑"),
                   ("keyboard", "键盘"), ("person", "人"), ("bottle", "瓶子")]:
        assert COCO_ZH[en] == zh, "%s 的中文名是 %s，不是 %s" % (en, COCO_ZH[en], zh)
    assert zh_name("nonexistent-thing") == "nonexistent-thing", "没中文名时应退回英文"
    # "face" 不在 COCO 表里（Haar 后端的产物），但也必须有中文名和量词 ——
    # 否则 Haar 模式下会说出"1 个face"
    assert zh_name("face") == "人脸", "face 的中文名不对：%s" % zh_name("face")
    assert summarize([{"name": "face"}]) == "1 张人脸", \
        "人脸的量词不对：%s" % summarize([{"name": "face"}])
    return "80 类齐全 · 无重复 · 中英一一对应 · 未知类退回英文 · face 单独映射"


def _check_summarize():
    """措辞：这是要念给用户听的，读起来得顺。"""
    objs = [{"name": "cup"}, {"name": "cup"}, {"name": "person"}]
    s = summarize(objs)
    assert "2 个杯子" in s and "1 个人" in s, "措辞不对：%s" % s
    assert summarize([]) == "", "没有东西时应该是空字符串"

    # 量词：「1 个书」很别扭，应该是「1 本书」
    assert summarize([{"name": "book"}]) == "1 本书", \
        "量词没生效：%s" % summarize([{"name": "book"}])
    assert summarize([{"name": "car"}]) == "1 辆汽车", \
        "量词没生效：%s" % summarize([{"name": "car"}])
    # 表里没有的退回"个"
    assert summarize([{"name": "frisbee"}]) == "1 个飞盘", \
        "没有量词的类应该退回「个」：%s" % summarize([{"name": "frisbee"}])

    # 种类超了要收在"等 N 样"里，而不是把 12 种全列出来
    many = [{"name": n} for n in COCO_NAMES[:9]]
    s2 = summarize(many, max_n=3)
    assert "等 9 样东西" in s2, "折叠没生效：%s" % s2
    assert s2.count("、") == 2, "前三样之间才该有顿号：%s" % s2
    assert "、等" not in s2, "「等」前面不该有顿号（会读成第 4 样东西）：%s" % s2

    assert labels_line(objs) == "cup x2, person x1", \
        "英文短标签不对：%s" % labels_line(objs)
    return "「2 个杯子、1 个人」· 量词（1 本书 / 1 辆车）· 空输入给空串 · 超 3 种收在「等 N 样」"


def _check_fake_detector():
    """假检测器：剧本推进、归一化坐标换算、循环。"""
    script = [(2, [_o("cup", 0.5, 0.5, 0.1, 0.1)]), (3, [])]
    d = FakeDetector(script, loop=True)
    frame = np.zeros((480, 640, 3), np.uint8)

    got = [len(d.detect(frame)) for _ in range(7)]
    assert got == [1, 1, 0, 0, 0, 1, 1], "剧本推进不对：%s" % got

    o = FakeDetector(script, loop=True).detect(frame)[0]
    assert o["box"] == (320, 240, 64, 48), \
        "归一化坐标换算不对：%s（应为 320,240,64,48）" % (o["box"],)

    # 不循环时应该停在最后一段
    d2 = FakeDetector(script, loop=False)
    last = [len(d2.detect(frame)) for _ in range(9)]
    assert last[-1] == 0 and last[5] == 0, "不循环时没停住：%s" % last
    return "剧本按帧推进 · 归一化坐标→像素正确 · loop 开关正确"


def _check_real_model(mdir=None):
    """真有模型就跑一次**真**推理。

    注意这里只断言"跑通了、形状对"，**不断言检测到了什么** ——
    拿一张合成图去要求模型认出杯子，那是自欺欺人。
    "能不能真的认出东西"由 `tools/vision_e2e.py` 拿**真实照片**验证。
    """
    p = model_path("yolov8n", mdir)
    if not have_model("yolov8n", mdir):
        return None
    d = YoloOnnxDetector(p, threads=2)
    frame = np.zeros((480, 640, 3), np.uint8)
    t0 = time.time()
    objs = d.detect(frame)
    ms = (time.time() - t0) * 1000.0
    assert isinstance(objs, list), "detect() 应该返回列表"
    objs2 = d.detect(np.full((480, 640, 3), 128, np.uint8))
    assert isinstance(objs2, list)
    return "真模型跑通：全黑图 %d 个目标 · %.0f ms/帧 · %s" % (
        len(objs), ms, d.describe())


def selftest(mdir=None):
    print("=" * 66)
    print("  MOSS 视觉层自检（阶段 18 · 强视觉）")
    print("=" * 66)

    groups = [
        ("[1/7] letterbox 几何 —— 填充和缩放必须是可逆的", _check_letterbox_roundtrip),
        ("[2/7] 通道顺序 —— BGR 必须真的翻成 RGB", _check_channel_order),
        ("[3/7] 后处理 —— 一张张量解出一串框", _check_postprocess),
        ("[4/7] NMS —— 重复框要合并，跨类不能合并", _check_nms),
        ("[5/7] COCO 类表 —— 80 类中英一一对应", _check_coco_table),
        ("[6/7] 说人话 —— 措辞和折叠", _check_summarize),
        ("[7/7] 假检测器 —— 剧本推进与坐标换算", _check_fake_detector),
    ]
    for title, fn in groups:
        print("\n" + title)
        msg = fn()
        print("      " + msg)
        print("      ✓")

    print("\n[附加] 真模型冒烟（有模型才跑）")
    msg = _check_real_model(mdir)
    if msg is None:
        print("      跳过：没下载模型。要跑的话先 `python vision.py --download`")
    else:
        print("      " + msg)
        print("      ✓")

    print("\n" + "=" * 66)
    print("[OK] vision.py 自检全部通过（7/7）")
    print("=" * 66)
    return 0


# ================================================================
#  7. 命令行
# ================================================================
def _load_image(path, w=0, h=0):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    if w and h:
        img = cv2.resize(img, (w, h))
    return img


def main():
    ap = argparse.ArgumentParser(
        description="MOSS 视觉层 —— 目标检测（YOLOv8n / ONNX），也可以只当工具用")
    ap.add_argument("--selftest", action="store_true", help="跑自检（不需要摄像头）")
    ap.add_argument("--list-models", action="store_true", help="列可用模型")
    ap.add_argument("--download", nargs="?", const="yolov8n", metavar="MODEL",
                    help="下载模型（默认 yolov8n）")
    ap.add_argument("--image", help="对一张图跑检测，打印结果")
    ap.add_argument("--model", default="auto",
                    help="后端：auto / yolo / haar / null / fake")
    ap.add_argument("--model-path", help="直接指定 .onnx 文件")
    ap.add_argument("--dir", help="模型目录（默认 ~/.cache/moss_vision）")
    ap.add_argument("--mirror", default="hf-mirror.com", help="模型镜像站")
    ap.add_argument("--conf", type=float, default=0.35, help="置信度阈值")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS 的 IoU 阈值")
    ap.add_argument("--imgsz", type=int, default=640, help="模型输入边长")
    ap.add_argument("--max-obj", type=int, default=12, help="最多返回几个目标")
    ap.add_argument("--threads", type=int, default=2,
                    help="ONNX 线程数（默认 2。别贪多，音频要实时）")
    ap.add_argument("--json", action="store_true", help="--image 时输出 JSON")
    args = ap.parse_args()

    if args.selftest:
        return selftest(args.dir)
    if args.list_models:
        print(list_models(args.dir))
        return 0
    if args.download:
        p = download_model(args.download, args.dir, args.mirror)
        print(p)
        return 0

    if not args.image:
        ap.print_help()
        return 0

    det, note = make_detector(
        args.model, args.dir, args.mirror,
        model_path=args.model_path, imgsz=args.imgsz, conf_th=args.conf,
        iou_th=args.iou, max_obj=args.max_obj, threads=args.threads)
    if note:
        print("[提示] %s" % note, file=sys.stderr)
    print("[检测器] %s" % det.describe(), file=sys.stderr)

    frame = _load_image(args.image)
    t0 = time.time()
    objs = det.detect(frame)
    ms = (time.time() - t0) * 1000.0

    if args.json:
        print(json.dumps({"image": args.image, "ms": round(ms, 1),
                          "objects": [dict(o, zh=zh_name(o["name"])) for o in objs]},
                         ensure_ascii=False, indent=2))
    else:
        print("%s  %dx%d  %.0f ms" % (args.image, frame.shape[1], frame.shape[0], ms))
        if not objs:
            print("  （什么都没检测到）")
        for o in objs:
            x, y, w, h = o["box"]
            print("  %-14s %-8s conf %.2f  框 (%d,%d,%d,%d)"
                  % (o["name"], zh_name(o["name"]), o["conf"], x, y, w, h))
        print("  一句话：%s" % (summarize(objs) or "空"))
    det.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
