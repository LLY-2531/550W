# MOSS 550W 复刻 · 软件安装文档（PC 端 · 无需硬件）

> **适用对象**：只有 PC、还没有硬件（ESP32 等）的软件开发者  
> **目标**：在 PC 上完整跑通 MOSS 的 AI 对话能力（摄像头识别 + 语音识别 + 大模型自由对话 + 语音合成 + 记忆 + 工具调用）  
> **适用环境**：Windows 10/11（原生运行，**不需要虚拟机、不需要 Linux**）  
> **不需要**：ESP32 开发板、OLED 屏幕、步进电机等任何硬件——摄像头用笔记本/外接摄像头，声音用系统麦克风+扬声器  
> **仓库**：`https://github.com/<你的用户名>/<仓库名>`　分支：`main`

> 📖 本项目的完整功能说明见 [`README.md`](README.md)（设计文档）与 [`pc_brain/DEMO.md`](pc_brain/DEMO.md)（演示手册）。  
> 本文只讲**软件端**如何从零装到跑起来。

---

## 目录

1. 准备工作
2. 获取项目源码
3. 安装 Python
4. 创建虚拟环境并安装依赖
5. 下载模型（语音识别 + 目标检测）
6. 配置大模型（DeepSeek API Key）
7. 首次启动与验证
8. 常用操作速查
9. 常见问题排错

---

## 一、准备工作

### 1.1 硬件要求（软件端最低配置）

| 项目  | 最低配置           | 推荐配置                  |
| --- | -------------- | --------------------- |
| CPU | 4 核            | 8 核                   |
| 内存  | 8 GB           | 16 GB                 |
| 硬盘  | 10 GB 空闲       | 20 GB 空闲（模型占约 0.5 GB） |
| 摄像头 | 可选（演示「眼睛」用）    | 任意 USB/笔记本摄像头         |
| 麦克风 | 可选（演示「耳朵」用）    | USB 耳麦最佳              |
| 网络  | 首次需访问国内镜像（下模型） | 能访问外网                 |

### 1.2 软件

| 软件          | 说明                     | 下载                                  |
| ----------- | ---------------------- | ----------------------------------- |
| Python      | 3.10 或更高（推荐 3.11/3.12） | <https://www.python.org/downloads/> |
| Git         | 拉取源码用                  | <https://git-scm.com/download/win>  |
| VS Code（可选） | 编辑、看日志更方便              | <https://code.visualstudio.com/>    |

### 1.3 账号准备

- **DeepSeek API Key**：<https://platform.deepseek.com/> （对话要用的**大模型**，`deepseek-chat` 模型）
  - 本项目对话层用的是 OpenAI 兼容接口，除了 DeepSeek，**智谱 / 通义 / OpenAI / 本地 ollama 也都支持**，换一家只需改 `--api-base` 和 `--model` 两个参数。

---

## 二、获取项目源码

### 2.1 克隆仓库

```powershell
git clone https://github.com/<你的用户名>/<仓库名>.git
cd <仓库名>
```

### 2.2 验证目录结构

```powershell
dir
```

应看到：

```
├── pc_brain/          ← PC 端大脑（核心，本文的主角）
│   ├── moss_brain.py  ← 总入口
│   ├── requirements.txt
│   ├── DEMO.md        ← 演示手册
│   └── ...
├── src/  include/     ← 固件（ESP32，软件端演示用不到）
├── platformio.ini
└── ...
```

> 软件端演示**只需要 `pc_brain/` 这一个目录**，`src/` 固件部分等上真机时再管。

---

## 三、安装 Python

### 3.1 安装

1. 运行 Python 安装包，**勾选「Add Python to PATH」**（这一步很关键，漏了后面 `python` 命令会找不到）。
2. 一路下一步完成。

### 3.2 验证

在 **PowerShell** 里输入（Win 键 + 输入 `powershell` 回车）：

```powershell
python --version
```

应输出类似 `Python 3.12.x`。

> ⚠️ 若提示 `python 不是内部或外部命令`，说明 3.1 没勾「Add Python to PATH」。重装一次并勾上，或手动把 Python 安装目录加进系统环境变量 `Path`。

---

## 四、创建虚拟环境并安装依赖

> 强烈建议用虚拟环境（venv），避免污染系统 Python，也和别的项目隔离。

### 4.1 创建虚拟环境

进入 `pc_brain` 目录：

```powershell
cd pc_brain
python -m venv .venv
```

### 4.2 激活虚拟环境

```powershell
.\.venv\Scripts\activate
```

激活成功后，提示符前面会出现 `(.venv)`。

### 4.3 安装依赖

```powershell
pip install -r requirements.txt
```

> 国内网络慢可加清华镜像：
>
> ```powershell
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

### 4.4 验证依赖

```powershell
python -c "import cv2, numpy, serial, sounddevice, faster_whisper, zhconv; print('依赖 OK')"
```

输出 `依赖 OK` 即成功。

### 4.5 依赖说明（哪些可选）

| 包                                                      | 什么时候需要                                |
| ------------------------------------------------------ | ------------------------------------- |
| `opencv-python` / `numpy` / `pyserial` / `sounddevice` | 核心，必装                                 |
| `faster-whisper`                                       | 只有用本地语音识别（`--asr whisper`）才需要         |
| `edge-tts`                                             | 只有用在线音色（`--tts edge`）才需要；离线用 SAPI 不需要 |
| `zhconv`                                               | whisper 输出繁体转简体，很小                    |

> 本项目刻意保持「零重型依赖」：目标检测用 `onnxruntime`（faster-whisper 自带）跑 `.onnx`，**不装 PyTorch/ultralytics**；回声消除、记忆、变化感知都是纯 Python/标准库。所以装依赖很快。

---

## 五、下载模型

模型文件**不走 pip**，要单独下（因为托管在模型仓库，不在 PyPI）。

### 5.1 语音识别模型（whisper）

看有哪些可选：

```powershell
python asr.py --list-models
```

| 模型        | 大小     | 说明             |
| --------- | ------ | -------------- |
| tiny      | 75 MB  | 最快，精度一般        |
| **small** | 484 MB | **默认**，速度/精度均衡 |
| medium    | 1.5 GB | 更准，更慢          |
| large-v3  | 3.0 GB | 最准，很慢          |

下载默认的 small：

```powershell
python asr.py --get-model small
```

> ⚠️ **也可以不下**：首次用 `--asr whisper` 跑的时候会自动下载。但**建议提前下好**，避免第一次启动卡在下载上。

### 5.2 目标检测模型（yolov8n）

```powershell
python vision.py --list-models
python vision.py --download
```

> yolov8n.onnx 只有 **12.2 MB**，几秒下完，落在 `~/.cache/moss_vision/`。

> ⚠️ **不下也能跑**：`--vision-model auto` 会自动降级——没有 yolo 模型就退回 haar（只认人脸），再不行退回 null（只测运动），并在启动横幅的「识别」那行写清原因。视觉是锦上添花，不会拖垮听觉和对话。

---

## 六、配置大模型（DeepSeek API Key）

对话层需要一个大模型。**密钥不随仓库分发**，通过环境变量注入。

### 6.1 设置 API Key（临时，仅当前窗口）

```powershell
$env:MOSS_API_KEY = "你的DeepSeek-Key"
```

> 关掉这个 PowerShell 窗口就失效，适合临时测试。

### 6.2 设置 API Key（永久，推荐）

```powershell
[Environment]::SetEnvironmentVariable("MOSS_API_KEY", "你的DeepSeek-Key", "User")
```

> 设完后**重开一个 PowerShell 窗口**才生效。之后所有窗口都自动有，不用每次设。

### 6.3 验证 Key 可用（不启动整个大脑）

```powershell
python mind.py --llm api --api-base "https://api.deepseek.com/v1" --api-key "你的Key" --model deepseek-chat --tools
```

预期看到：

```
[后端] API https://api.deepseek.com/v1 / deepseek-chat
...
[2] 有人说话 → 回答问题
  听到 '你是谁' → say='我是550W，行星发动机的控制中枢。' ...
[3] 工具编排 ...
  听到 '现在几点了' → 申请 get_time({}) ...
[OK] 语言层自检通过。
```

> 若这里报 `401`（Key 错）、`402`（余额不足）、`429`（限流），见第九章排错。

---

## 七、首次启动与验证

### 7.1 快速自检（可选但推荐）

每个模块自带自检，先把链路各段单独验一遍：

```powershell
python wake.py                 # 唤醒门
python mind.py --llm api --api-base "https://api.deepseek.com/v1" --api-key "你的Key" --model deepseek-chat --tools   # 语言层
python vision.py --selftest    # 视觉
python moss_tools.py --selftest  # 工具
python aec.py --selftest       # 回声消除
```

### 7.2 完整启动（真实摄像头 + 麦克风 + 自由对话）

```powershell
python moss_brain.py --llm api --api-base "https://api.deepseek.com/v1" --model deepseek-chat --no-echo-guard
```

> 已固化的默认值（不用写）：`--tts sapi`（Windows 离线语音）、`--asr whisper`（本地识别）、唤醒词 `boss`、喊名字先应「在」、开机只说一次不再话痨、`--vision-source 0`、`--audio-source mic`。

启动后你会看到一屏横幅，重点确认这几行：

```
听懂：本地 whisper-small
语言：API https://api.deepseek.com/v1 / deepseek-chat
语音：Windows SAPI（离线）
唤醒：boss（...）
```

### 7.3 对话测试

对着麦克风清楚地说：

```
BOSS，现在几点了
```

预期链路：**唤醒（应一声「在」）→ 语音识别 → 意图/工具（查时间）→ 大模型回答 → 语音念出来**。

### 7.4 无麦克风也能演示「耳朵」（用音频文件）

```powershell
# 先造一句话（Windows SAPI 合成）
python tools/make_speech.py --text "现在几点了" --out tools/assets/my.wav

# 把 WAV 当麦克风喂进去
python moss_brain.py --demo --audio-source tools/assets/my.wav --asr whisper --llm api --api-base "https://api.deepseek.com/v1" --api-key "你的Key" --model deepseek-chat
```

### 7.5 全合成演示（连摄像头都没有也能跑）

```powershell
python moss_brain.py --demo --audio-source tools/assets/ask_time.wav --asr whisper --llm offline
```

---

## 八、常用操作速查

| 你想做什么              | 命令                                                                                                              |
| ------------------ | --------------------------------------------------------------------------------------------------------------- |
| 真实对话（摄像头+麦克风+大模型）  | `python moss_brain.py --llm api --api-base "https://api.deepseek.com/v1" --model deepseek-chat --no-echo-guard` |
| 只要离线模板（不联网、不接 Key） | `python moss_brain.py --llm offline`                                                                            |
| 用音频文件代替麦克风         | 加 `--audio-source 文件.wav`                                                                                       |
| 关掉状态汇报（彻底闭嘴，只回答）   | 加 `--narrate quiet --narrate-gap 9999`                                                                          |
| 换唤醒词               | 加 `--wake-word "你的词"`                                                                                           |
| 不要唤醒门（谁说话都答）       | 加 `--no-wake`                                                                                                   |
| 列出所有参数             | `python moss_brain.py --help`                                                                                   |

---

## 九、常见问题排错

| 现象                                             | 原因                      | 处理                                                              |
| ---------------------------------------------- | ----------------------- | --------------------------------------------------------------- |
| `python 不是内部或外部命令`                             | 装 Python 没勾 Add to PATH | 重装勾上，或手动加环境变量                                                   |
| `ModuleNotFoundError: No module named 'numpy'` | 没装依赖，或没用 venv           | `.\ .venv\Scripts\activate` 后 `pip install -r requirements.txt` |
| `can't open file 'moss_brain.py'`              | 目录不对                    | 先 `cd pc_brain`                                                 |
| 启动后「语言：离线模板」                                   | 大模型没接上                  | 检查 `--llm api` 和 Key 是否传入                                       |
| 它只回「我收到了，你说的是…」                                | 同上，退回了离线模板              | 同上，确认 Key 有效                                                    |
| 画面是个来回晃的小球                                     | 开了 `--demo`             | 去掉 `--demo`，用 `--vision-source 0`                               |
| 说话它听不到                                         | `--demo` 覆盖了麦克风         | 去掉 `--demo`，用 `--audio-source mic`                              |
| `--wav` 报「unrecognized arguments」              | 没这个参数                   | 用 `--audio-source xxx.wav`                                      |
| `HTTP Error 401`                               | Key 无效/拼错               | 检查 Key 完整复制                                                     |
| `HTTP Error 402 Insufficient Balance`          | 大模型余额不足                 | 去平台充值                                                           |
| `HTTP Error 429 Too Many Requests`             | 限流                      | 等 30 秒重试，或换一家                                                   |
| 喊名字没反应                                         | whisper 转写成了别的字         | 换用默认 `boss`（已配同音别名），或加 `--wake-word` 列全写法                       |

---

## 附：一页纸上手清单（TL;DR）

```powershell
# 1. 拉源码
git clone https://github.com/<你的用户名>/<仓库名>.git
cd <仓库名>\pc_brain

# 2. Python 环境 + 依赖
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 下模型（whisper small + yolov8n）
python asr.py --get-model small
python vision.py --download

# 4. 填 DeepSeek Key（永久，设完重开窗口）
[Environment]::SetEnvironmentVariable("MOSS_API_KEY", "你的Key", "User")

# 5. 启动（真实摄像头 + 麦克风 + 自由对话）
python moss_brain.py --llm api --api-base "https://api.deepseek.com/v1" --model deepseek-chat --no-echo-guard

# 6. 说一句试试
#    「BOSS，现在几点了」
```
