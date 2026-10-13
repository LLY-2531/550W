# pc_vision · MOSS 的视觉大脑

《流浪地球 550W》复刻项目的 **PC 端程序**。它读你电脑的摄像头，判断画面里
有没有人脸、有没有在动，然后把结果通过串口推给 ESP32 —— 让 MOSS 转头看你。

这一步对应你手上那份 PDF 里说的「下一步用 Python 模拟摄像头和音频」。

---

## 一、装环境（只做一次）

在 VS Code 里打开 **这个文件夹**（`pc_vision`），然后 `Ctrl + \`` 开终端：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

装完终端提示符前面会出现 `(.venv)`。**没看到 `(.venv)` 就说明还在用系统 Python，
装出来的库 VS Code 里跑不到。**

> **为什么 `requirements.txt` 里锁死 `opencv-python<5`**：OpenCV 5.x 的 pip 包
> **已经不再自带人脸识别模型文件**（`haarcascade_*.xml`），装上去程序会提示
> "找不到人脸模型，本次只做运动检测"。4.x 自带 17 个模型文件，所以钉在 4.x。
> 这是实测踩出来的坑，别随手升级到 5.x。

---

## 二、先让它跑起来（不需要任何硬件）

```bash
python moss_vision.py --source demo
```

屏幕上会出现一个深蓝色的小窗口，里面有个橙球**动 2 秒、停 1 秒**。
终端同时开始打印 JSON：

```json
{"t": 3.12, "faces": 0, "motion": 0.081, "mode": "MOTION", "text": "moving", "sent": false}
{"t": 5.40, "faces": 0, "motion": 0.000, "mode": "IDLE",   "text": "clear",  "sent": false}
```

`mode` 在 `MOTION` 和 `IDLE` 之间跳，就说明整条链路通了。

按 **q** 退出。

---

## 三、用真摄像头测人脸

```bash
python moss_vision.py
```

站到摄像头前。检测到脸 → `faces` 变成 1，`mode` 变成 `VISION`，
而且脸的位置会被画上绿色方框。

**没有摄像头怎么办**：随便找一张有人脸的照片（比如合影），然后

```bash
python moss_vision.py --source 照片.jpg
```

图片会循环显示，人脸检测照常工作。

---

## 四、接上 Wokwi 仿真（零硬件也能让 MOSS 反应）

1. 确认 `wokwi.toml` 里有这一行（已经帮你加好了）：

   ```toml
   rfc2217ServerPort = 4000
   ```

2. 在 VS Code 里 `F1` → **Wokwi: Start Simulator**，把仿真跑起来。

3. **注意：仿真面板必须保持可见**。如果那个标签页被切走，仿真会暂停，串口也不通。

4. 另开一个终端，运行：

   ```bash
   python moss_vision.py --source demo --port rfc2217://localhost:4000
   ```

5. 现在再看仿真里的 MOSS：球一动，它就会切到 `MOTION` 状态、电机开始扫；
   安静下来就回 `IDLE`。OLED 上还会显示 `moving` / `clear`。

> 同一时刻只能有一个程序占用 4000 端口。如果连不上，先关掉别的串口工具。

### 没有 Wokwi 也能验证串口链路（本机假服务端）

Wokwi 仿真必须在图形界面里手动启动，不方便自动化测试。所以 `tools/` 里放了一个
**最小 RFC2217 服务端**，它扮演"假 Wokwi + 假 ESP32"：按标准协议跟 pyserial 握手，
然后把它收到的命令打印出来。

```bash
# 终端 1：起假服务端（监听 4000）
python tools/rfc2217_mock_esp32.py --verbose

# 终端 2：让视觉程序连进去
python moss_vision.py --source demo --port rfc2217://localhost:4000
```

终端 1 会打印：

```
[ESP32 收到] MODE MOTION
             └─ 切换到状态：MOTION
[ESP32 收到] TEXT moving
             └─ OLED 第三行显示：moving
```

**它能帮你分清是"Python 没发出去"还是"设备没收到"**：

- 假服务端能收到 → Python 端没问题，问题在 Wokwi/真机那边
- 假服务端也收不到 → 检查 `--port` 参数和端口占用

---

## 五、接上真机（以后有硬件了）

把 ESP32 用 USB 插上电脑，在"设备管理器"里看它是哪个 COM 口，然后：

```bash
python moss_vision.py --port COM3
```

---

## 六、全部参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `--source` | `0`=摄像头、`demo`=合成画面、或视频/图片路径 | `0` |
| `--port` | `COM3` 或 `rfc2217://localhost:4000`，不给就不发串口 | 无 |
| `--baud` | 波特率，要跟固件里 `SERIAL_BAUD` 一致 | `115200` |
| `--no-window` | 不弹预览窗，只打印 JSON | 关 |
| `--frames N` | 只跑 N 帧就退出（自测用） | `0`（一直跑） |
| `--motion-th` | 运动判定阈值，调大 = 更不敏感 | `0.008` |
| `--stable N` | 连续 N 帧一致才切状态，防抖动 | `3` |
| `--heartbeat` | 状态不变时每隔几秒也输出一行 JSON | `2.0` |
| `--verbose` | 每一帧都输出 JSON | 关 |

---

## 七、它和固件之间的"暗号"

程序往串口里发的就是固件里已经写好的那几条命令：

| 发出去的 | ESP32 收到后 |
|---|---|
| `MODE VISION` | 切到 VISION 状态（青灯流光 + 电机左右扫） |
| `MODE MOTION` | 切到 MOTION 状态（橙灯扫描 + 电机连续转） |
| `MODE IDLE` | 回待机 |
| `TEXT face x1` | OLED 第三行显示 `face x1` |

这是**故意设计成同一套协议**的：以后换成真机、换成大模型来发指令，
固件一行都不用改。

---

## 八、排错

| 现象 | 原因 | 修法 |
|---|---|---|
| `ModuleNotFoundError: cv2` | 库装到了系统 Python，而不是 `.venv` | 确认终端有 `(.venv)` 前缀再 pip install |
| 窗口一闪而过 / 报 `imshow` 错误 | 用的 `opencv-python-headless`（无界面版） | 卸载后装 `opencv-python` |
| `打不开摄像头 0` | 没有摄像头 / 被别的程序占用 | 先用 `--source demo`；关掉会议软件再试 |
| 串口连不上 | 端口号错 / 仿真面板被切走 | 检查 `--port`；确保 Wokwi 面板可见 |
| JSON 一直 `faces: 0` | 光线太暗 / 脸太小 / 侧脸 | 靠近一点、正对镜头、开灯 |
| 状态疯狂跳 | 阈值太敏感 | 把 `--stable` 调大，或 `--motion-th` 调大 |

---

## 九、下一步可以做什么

- **加音频**：✅ 已经做了 —— 见旁边的 [`../pc_audio`](../pc_audio)。
  它读麦克风判断"有没有人在说话"，并且用的是**和本程序完全一样的串口协议**。
  注意两者都要用 4000 端口，所以**不能同时连 Wokwi**，要换着跑。
- ~~**换更强的识别**：把 Haar 人脸换成 `yolov8n` 做通用物体识别~~ ✅ 阶段 18 已完成 ——
  见 [`../pc_brain/vision.py`](../pc_brain/vision.py) 与 pc_brain README 的「五之十一」。
  两点和当初设想不一样：**没用 ultralytics**（会拖进 2GB+ 的 PyTorch），
  而是用早就装好的 `onnxruntime` 直接跑 `.onnx`；模型文件单独下，不走 pip。
  动因也不只是"想认杯子"：**合成场景里 `faces` 永远是 0**，
  让主动行为在 `--demo` 下根本测不到 —— 换掉眼睛顺带把这个真问题一起解决了。
- ~~**接大模型**：把 JSON 结果喂给大模型，让 MOSS 说出自己的判断~~ ✅ 阶段 10 已完成 ——
  由 [`../pc_brain/moss_brain.py`](../pc_brain/moss_brain.py) + `mind.py` 接管
  （本程序现在只保留"单独调视觉算法"的轻量用途，不再自己接语言层）。

---

## 十、实测记录（2026-10-11）

| # | 测什么 | 命令 | 结果 |
|---|---|---|---|
| 1 | 合成画面 + 运动检测 | `--source demo --frames 400` | ✅ `MOTION` ↔ `IDLE` 正常来回切，~30fps |
| 2 | 人脸识别 | `--source <人脸图>` | ✅ 检出 1 张脸 → `VISION` / `face x1` |
| 3 | rfc2217 串口协议 | 假服务端 + `--port rfc2217://localhost:4010` | ✅ 握手全通过；收到 `MODE MOTION` / `TEXT moving` 等 |
| 4 | 真实摄像头 | `--source 0 --frames 150` | ✅ 640×480 正常出图，检出人脸 → `VISION` |
| 5 | **完整闭环** | `--source 0 --port rfc2217://localhost:4000` | ✅ 真实人脸 → `MODE VISION` + `TEXT face x1` 被对端完整收到 |

边界情况也测了，都是"优雅降级"而不是崩溃：

- 串口连不上（端口没人监听）→ 打印提示，**继续只输出 JSON**，视觉不受影响
- 文件路径不存在 → 提示"打不开画面来源"，并建议用 `--source demo`
- 没有图形界面 → 自动关掉预览窗，转无窗口模式继续跑

> 顺带记一笔：本项目里 `rfc2217ServerPort = 4000` 这个键名已对照 Wokwi 官方文档
> 核实无误（<https://docs.wokwi.com/vscode/project-config>），官方给的 Python 示例
> 就是 `serial.serial_for_url('rfc2217://localhost:4000', baudrate=115200)`，
> 和本程序完全一致。
