# MOSS 550W 复刻

> 复刻《流浪地球》中的超级人工智能 **550W（MOSS）**。
> 一个能**看**、能**听**、能**说**、能**想**、能**动**、还能**记住你**的桌面 AI 机器人。

[![PlatformIO](https://img.shields.io/badge/PlatformIO-ESP32--S3-orange)](https://platformio.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](#license)

---

## 简介

这个项目把「MOSS」拆成两层：

- **PC 端大脑（`pc_brain/`）**：Python 编写，摄像头（眼睛）+ 麦克风（耳朵）+ 大模型（大脑）+ 语音合成（嘴巴）+ 长期记忆 + 工具调用，全部跑在一台普通电脑上。
- **ESP32 固件（`src/` + `include/`）**：OLED 眼睛、WS2812 灯环、步进电机（会转头）、蜂鸣器，是「MOSS 的躯体」。

两者通过一根串口连接。**没有硬件也能完整跑软件端**——用摄像头和麦克风就能体验「跟 MOSS 对话」。

---

## 功能一览

| 能力 | 说明 |
|---|---|
| 👁 看 | 摄像头 + YOLOv8n 目标检测，认得出杯子、键盘、人 |
| 👂 听 | 麦克风 + whisper 语音识别，听懂你说什么 |
| 🗣 说 | TTS 语音合成，用声音回答你 |
| 🧠 想 | 接入大模型（DeepSeek/智谱/通义/OpenAI/本地 ollama），自由对话 |
| 🧾 记 | 长期记忆，跨会话记住你（「我叫张三」→ 下次记得） |
| 🔧 动 | 工具调用：查时间、查天气、记事、回忆 |
| 🎭 变 | 变化感知：桌上多了个杯子，不用你问它也会主动说 |
| ⚙️ 动 | 执行器：驱动 OLED / 灯环 / 步进电机 / 蜂鸣器（上真机后） |

---

## 架构

```
摄像头 ──→ 视觉线程（目标检测 + 在场跟踪 + 运动检测）
麦克风 ──→ 听觉线程（回声消除 + 音量/拍手/切句）
                        │
                        ▼
              识别线程（whisper：声音 → 文字）
                        │
                        ▼
        ┌── 主循环 20Hz 融合（防抖动）──┐
        │                               │
   状态变化                         听懂一句话
        │                               │
        ▼                               ▼
  MODE 发给设备              语言线程（大模型 / 离线模板）
                                    │
                                    ▼
                          语音线程（TTS 念出来）
                                    │
                                    ▼
                          一条串口 → ESP32 实体
```

---

## 目录结构

```
550W/
├── pc_brain/                 ← PC 端大脑（核心）
│   ├── moss_brain.py         ← 总入口：五线程 + 20Hz 融合循环
│   ├── vision.py             ← 眼睛：YOLOv8n 目标检测（阶段 8/18）
│   ├── presence.py           ← 变化感知：物体出现/消失 → 事件（阶段 19）
│   ├── asr.py                ← 耳朵：whisper 语音识别（阶段 12）
│   ├── aec.py                ← 回声消除：边放边听（阶段 17）
│   ├── wake.py               ← 唤醒门：叫名字才答应（阶段 15）
│   ├── mind.py               ← 嘴：状态/问题 → 人话，接大模型（阶段 10）
│   ├── voice.py              ← 嗓子：TTS（阶段 11）
│   ├── memory.py             ← 长期记忆（阶段 13）
│   ├── moss_tools.py         ← 工具：时间/天气/记事（阶段 13）
│   ├── moss_actuators.py     ← 执行器：驱动硬件（阶段 14）
│   ├── behave.py             ← 行为层：主动行为（阶段 16/19）
│   ├── tools/                ← 离线验证工具（假设备/端到端探针）
│   ├── DEMO.md               ← 演示与操作手册
│   └── README.md             ← 设计文档
├── src/  include/            ← ESP32 固件（躯体）
├── platformio.ini            ← 固件构建配置
├── INSTALL.md                ← ★ 软件端安装文档（从这里开始）
└── ...
```

---

## 快速开始

> 完整步骤见 [**INSTALL.md**](INSTALL.md)。这里是最短路径（Windows）：

```powershell
# 1. 拉源码
git clone https://github.com/LLY-2531/550W.git
cd 550W\pc_brain

# 2. 环境 + 依赖
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 3. 下模型（whisper small + yolov8n）
python asr.py --get-model small
python vision.py --download

# 4. 填大模型 Key（DeepSeek，设完重开窗口）
[Environment]::SetEnvironmentVariable("MOSS_API_KEY", "你的Key", "User")

# 5. 启动（真实摄像头 + 麦克风 + 自由对话）
python moss_brain.py --llm api --api-base "https://api.deepseek.com/v1" --model deepseek-chat --no-echo-guard
```

然后对着麦克风说：**「BOSS，现在几点了」**。

---

## 19 个阶段

项目按「阶段」推进，每个阶段解决一个具体问题：

| 阶段 | 能力 |
|---|---|
| 1–7 | 固件：状态机 / OLED / LED / 按键 / 步进 / 蜂鸣器 |
| 8 | PC 视觉：摄像头 + 目标检测 |
| 9 | 融合：眼睛 + 耳朵 → 一个判断 |
| 10 | 语言层：状态 → 人话 |
| 11 | 语音层：TTS 念出来 |
| 12 | 听觉理解：ASR 听懂你 |
| 13 | 长期记忆 + 工具调用 |
| 14 | 执行器：驱动硬件 |
| 15 | 唤醒门：叫名字才答应 |
| 16 | 主动行为：自己转头 / 变灯 |
| 17 | 全双工：边放边听（AEC 回声消除） |
| 18 | 强视觉：认出杯子 / 键盘 / 人 |
| 19 | 变化感知：东西出现 / 消失 → 主动说 |

---

## 设计哲学

- **零重型依赖**：目标检测用 `onnxruntime` 跑 `.onnx`（不装 PyTorch/ultralytics），回声消除纯 numpy 自写（PBFDAF），记忆/变化感知纯标准库。
- **换真机只改一个文件**：所有引脚集中在 `include/config.h`，PC 端代码回真机一行不用改。
- **模型不走 pip**：whisper / yolov8n 从镜像站单独下载，断网可降级运行。

---

## 文档导航

| 文档 | 内容 |
|---|---|
| [INSTALL.md](INSTALL.md) | ★ 软件端安装文档（从零装到跑起来） |
| [pc_brain/DEMO.md](pc_brain/DEMO.md) | 演示与操作手册（要说什么、怎么验证） |
| [pc_brain/README.md](pc_brain/README.md) | 设计文档（每一层为什么这么写） |

---

## License

MIT License
