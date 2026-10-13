# MOSS 演示与操作手册

> 流浪地球 550W 复刻项目 · PC 端大脑（阶段 9–19）  
> **你一块硬件都没有也能跑完整场演示。** 这份手册里每一条命令都在本机实测过。

---

## 零、先看懂一件事：两条"假腿"

这个项目从第一天起就是**无硬件复刻**，所以每个真实输入都配了一个**假的**：

| 真东西      | 假的替身                                       | 什么时候用假的                |
| -------- | ------------------------------------------ | ---------------------- |
| 摄像头      | `--vision-source demo`（合成画面：黑底 + 画出来的圆当人脸） | 想**稳定复现**同一个结果时        |
| 麦克风      | `--audio-source 某个.wav`（把 WAV 当麦克风喂进去）     | **麦克风权限被挡时——本机就是这种情况** |
| ESP32 设备 | `tools/mock_esp32.py`（一个假串口服务）             | 没有板子时                  |
| 大模型      | `--llm offline`（内置模板，零配置零联网）               | 没有 API Key 时           |
| YOLO 检测器 | `--vision-model fake`（按剧本报物体）              | 想测"下游逻辑"而不是"模型准不准"时    |

**演示的第一原则：假腿换哪条，结论的适用范围就跟着变。**  
用 `fake` 检测器验出来的是**逻辑对**，不是**模型准**。手册里每条都会标清楚。

---

## 一、一次性准备

```bash
cd C:/Users/hp/Documents/PlatformIO/Projects/550W_Sim/pc_brain

# 1. Python 必须用项目自己的 venv（托管 Python 没装 numpy，直接用会报 ModuleNotFoundError）
./.venv/Scripts/python.exe -c "import numpy, cv2, onnxruntime, faster_whisper; print('依赖 OK')"

# 2. 确认两个模型都在本地（它们不在 pip 里，是单独下的）
ls ~/.cache/moss_vision/yolov8n.onnx        # 12.2 MB，目标检测
ls ~/.cache/moss_asr/                        # faster-whisper-tiny/base/small

# 3. 检查嗓子能用（本机 edge-tts 被 403，所以默认走 sapi）
./.venv/Scripts/python.exe moss_brain.py --tts sapi --tts-test
```

**三条铁律**：

1. **永远写 `./.venv/Scripts/python.exe`**，不要写 `python`。
2. **假设备用 `--port socket://127.0.0.1:4013`**，不是 `--serial`。pyserial 天生认这个伪 URL，不用折腾虚拟串口。
3. **假设备要在另一个终端先起**（或用后台任务）。`WinError 10061 拒绝连接` = 假设备没起来。

### 涉及哪些代码

| 文件                     | 干什么                   | 演示里什么时候会碰到   |
| ---------------------- | --------------------- | ------------ |
| `moss_brain.py`        | **主程序**。所有演示都是它       | 每一条命令        |
| `vision.py`            | 目标检测（阶段 18）           | 演示 1         |
| `presence.py`          | 物体在场跟踪 / 滤抖动（阶段 19）   | 演示 5         |
| `behave.py`            | 主动行为 + 权限分级（阶段 16/19） | 演示 5         |
| `asr.py` `wake.py`     | 听觉理解 / 唤醒门（阶段 12/15）  | 演示 2、4       |
| `mind.py`              | 语言层 + 工具调度（阶段 10/13）  | 演示 4         |
| `voice.py`             | 嗓子 TTS（阶段 11）         | 演示 3         |
| `memory.py`            | 长期记忆（阶段 13）           | 演示 4         |
| `moss_tools.py`        | 13 个工具（阶段 13/14/18）   | 演示 4         |
| `aec.py`               | 回声消除（阶段 17）           | 只在真机/戴耳机时有意义 |
| `tools/mock_esp32.py`  | 假设备                   | 演示 6         |
| `tools/make_speech.py` | **造测试语音**（本手册新增）      | 演示 2 的准备步骤   |

---

## 二、演示 1：眼睛 —— 看真东西

**最直观的一条。用你笔记本的摄像头，真模型认真物体。**

```bash
./.venv/Scripts/python.exe vision.py --image tools/assets/desk_objects.png
```

实测输出：

```
[检测器] yolov8n（COCO 80 类 · ONNX · 640×640 · CPU 2 线程）
tools/assets/desk_objects.png  1024x1024  58 ms
  cup            杯子       conf 0.91  框 (695,382,113,147)
  keyboard       键盘       conf 0.84  框 (204,561,677,174)
  bottle         瓶子       conf 0.82  框 (23,326,177,305)
  一句话：1 个杯子、1 个瓶子、1 个键盘
```

换成**你的**摄像头（会弹预览窗口，绿框是人、蓝框是别的，按 q 退出）：

```bash
./.venv/Scripts/python.exe moss_brain.py --vision-source 0 --audio-source demo \
    --llm offline --tts off --autonomy off --frames 300
```

本机实测：`person x1, book x1`、`person x2, book x1`，融合状态 `VISION 14 次 / SPEAK 5 次 / IDLE 1 次`。

**说什么**：这一条不用说话，**对着镜头举一个杯子**就能看到 `cup` 跳进日志。

---

## 三、演示 2：耳朵 —— 真 whisper 认真人话（不用麦克风）

本机麦克风被 Windows 隐私权限挡着（所有设备都录到 **-96 dBFS 数字静音**）。  
但 `moss_brain.py` 支持**把 WAV 当麦克风**，所以先用 SAPI 造一段真人话的 WAV，再喂给真的 whisper。

### 第一步：造测试语音

```bash
# 一句句造（想说什么就说什么）
./.venv/Scripts/python.exe tools/make_speech.py --say "摩丝，帮我看看画面" --out tools/assets/my.wav

# 或者一次造一整组演示用的（已生成过就不用再跑）
./.venv/Scripts/python.exe tools/make_speech.py --batch
```

`--batch` 会生成这四句 —— **它们分别打到四个不同的功能上**，不是随便挑的：

| 文件             | 内容          | 打的是哪个功能                  |
| -------------- | ----------- | ------------------------ |
| `ask_time.wav` | 摩丝，现在几点了    | 唤醒词 + **工具调用**（get_time） |
| `ask_look.wav` | 摩丝，看看画面里有什么 | 工具调用 + **视觉**（look）      |
| `ask_fact.wav` | 摩丝，记住我住在杭州  | **记忆写入**（remember）       |
| `no_wake.wav`  | 现在几点了       | **不带唤醒词 —— 验证它真的不理你**    |

### 第二步：喂给真 whisper

```bash
./.venv/Scripts/python.exe moss_brain.py \
    --asr whisper --asr-model small \
    --audio-source tools/assets/ask_time.wav \
    --asr-test
```

实测输出：

```
[HEARD] 摩斯现在几点了   (音频3.14s, 1578ms, RTF 0.503, whisper, off)
```

**模型换档**：`--asr-model tiny` 最快最糙，`base` 适中，`small`（默认）最准。三个都在本地了。  
RTF 0.5 的意思是"处理 3 秒音频要 1.5 秒"——**识别比说话慢一倍**，这是 CPU 上的正常水平。

---

## 四、演示 3：嘴 —— 真的出声

```bash
./.venv/Scripts/python.exe moss_brain.py --tts sapi --tts-test
```

实测：`[说话] 系统自检完成，各单元上线。我是五百五十W。` 用了 9.1 秒。

**为什么不是 edge-tts**（音质好得多）：本机 edge-tts 被服务端 **403**，所以默认退到 Windows 内置的 SAPI。  
机械音色反而更像 MOSS——项目里就是这么写进理由的。想试 edge 用 `--tts edge`，通了就通，不通会自动退回来。

---

## 五、演示 4：全链路对话 —— 一问一答（★ 最完整的一条）

这是**把前面全串起来**的一条：WAV → whisper → 唤醒门 → 语言层 → 真工具 → 说话 → 设备收到。

### 第一步：起假设备（单独一个终端）

```bash
./.venv/Scripts/python.exe tools/mock_esp32.py --port 4013
```

### 第二步：跑主程序

```bash
./.venv/Scripts/python.exe moss_brain.py \
    --vision-source demo --vision-model fake \
    --audio-source tools/assets/ask_time.wav \
    --asr whisper --asr-model small \
    --llm offline \
    --tts sapi \
    --autonomy l1 \
    --port socket://127.0.0.1:4013 \
    --frames 250
```

> ⚠️ **别用 `--demo`**。`--demo` 是总开关，会把 `--vision-source` 和 `--audio-source`  
> **都**强制切成合成——你的 WAV 会被无声地覆盖掉。想混搭就分开写这两个参数。

实测输出（这是一次真实运行）：

```
[唤醒] 「摩斯」开头 → 剥掉称呼，只送：现在几点了
[HEARD] 现在几点了   (音频3.14s, 1402ms, RTF 0.447, whisper, wake-head)
[工具] get_time({}) → ok 0ms  现在是 2026年10月12日 星期一 18:52:04
[SAY] 现在是 2026年10月12日 星期一 18:52:04。   [OLED] time   (offline, 0ms, 工具 get_time)
   -> 出声
```

### 要说什么 → 会得到什么

改 `--audio-source` 换句子，或者自己造 WAV 说任意话：

| 你说              | 它会做什么              | 实测结果                                                    |
| --------------- | ------------------ | ------------------------------------------------------- |
| 摩丝，现在几点了        | 调 `get_time` 工具    | `[工具] get_time({}) → 现在是 2026年10月12日 星期一 18:52:04`      |
| 摩丝，看看画面里有什么     | 调 `look` 工具        | `[工具] look({}) → 我看到 1 个人，基本静止（运动量 0.0%）`               |
| 摩丝，记住我住在杭州      | 调 `remember` 写长期记忆 | `[工具] remember({"key": "用户所在城市", "value": "杭州"}) → 已记住` |
| **现在几点了**（不带名字） | **不理你**            | `唤醒 0 次 · 忽略 4 句`                                       |

**记忆是真的落盘了**（换个进程也还在）：

```bash
./.venv/Scripts/python.exe -c "
import json; d=json.load(open('data/moss_memory.json',encoding='utf-8'))
print(d['facts'])"
# {'用户所在城市': {'value': '杭州', 'at': 1791802359.9, 'hits': 0}}
```

**三个细节值得注意**：

- 唤醒门**剥掉称呼**才把话送进语言层（`只送：现在几点了`）——不然"摩斯"两个字会污染识别
- `[OLED]` 后面那个 `time` 就是**设备屏幕上真的会显示**的短标签
- 想关掉唤醒门（谁说话都应）用 `--no-wake`

---

## 六、演示 5：它自己会开口 —— 变化感知 + 主动行为

**这条最能体现"它不是个问答机器"。** 全程没人问它，它自己注意到了变化并说出来。

```bash
./.venv/Scripts/python.exe moss_brain.py \
    --demo --vision-model fake \
    --autonomy l1 --proactive-gap 0 \
    --tts off \
    --port socket://127.0.0.1:4013 \
    --frames 700
```

实测输出：

```
[主动] 听到拍手 → LED 0 255 60 500
[主动] 看到有人出现 → SPIN 1024 CW, LED 0 80 255 2500, BEEP 1500 60
[主动] 画面里多了 1 个杯子、1 台笔记本电脑、1 个键盘 → LED 0 255 160 700（要说话）
[主动] 说出口：我注意到画面里多了 1 个杯子、1 台笔记本电脑、1 个键盘。  [OLED] +cup +keyboard +laptop
[主动] 人走了，转回原位 → SPIN 1024 CCW, LED 255 120 0 800
[主动] 画面里少了 1 个杯子、1 台笔记本电脑、1 个键盘 → LED 0 255 160 700（要说话）
[主动] 说出口：我注意到画面里少了 1 个杯子、1 台笔记本电脑、1 个键盘。 [OLED] -cup -keyboard -laptop
```

收尾统计（**这行数字是整条链的收敛证据**）：

```
主动 10 次 · arrive 1 · leave 1 · clap 6 · notice 2 · 冷却拦住 117
变化感知：检测 349 次 · 新出现 5 · 消失 4
```

**349 次检测 → 2 句话**。中间那 117 次"冷却拦住"就是它没变成坏掉报警器的原因。

### 权限分级：`--autonomy` 三档

| 值                         | 允许它做什么              | 演示时看什么                   |
| ------------------------- | ------------------- | ------------------------ |
| `off`（或 `--no-proactive`） | 什么都不做               | 一条 `[主动]` 都没有            |
| `l0`                      | 转电机 + 改灯色（**不发声音**） | 有 `SPIN`/`LED`，没有 `BEEP` |
| `l1`                      | 上面 + 蜂鸣 + **说出那句话** | 出现 `[主动] 说出口：…`          |

> `--vision-model fake` 是**刻意用的**：目标检测本身有噪声（模型/光照/阈值让框闪），  
> 拿它验下游逻辑等于在沙子上盖楼。`fake` 按剧本稳定地报物体，这样"该开口的开口了吗"才测得准。

---

## 七、演示 6：设备端收到了什么（假 ESP32）

假设备会把它收到的每条命令打出来。跑任意一条上面的命令，然后看它的终端：

```
[设备] [CMD] TEXT = face x1 + cup, laptop
[设备] [CMD] LED 0 255 160 700
[设备] [CMD] SPIN 1024 CW
[设备] [CMD] TEXT = time
```

固件认这几条命令（`src/main.cpp`）：`MODE / TEXT / LED / SPIN / BEEP / STOP / PING / DUMP`。

**OLED 是 23 个字符的硬预算**（固件里 `char gLine1[24]`），所以短标签会自己截断：  
`+cup +keyboard +laptop` 正好 21 字符。超过会变成 `&N`（"还有 N 样"）——  
因为 `+3` 会被读成"又多了一个杯子"，那是撒谎。

---

## 八、验收：一次性把所有层都跑一遍

改完任何代码，跑这一组（每个模块都自带 `--selftest`）：

```bash
for m in presence behave vision moss_tools memory asr wake voice aec; do
  printf "%-12s " "$m"
  ./.venv/Scripts/python.exe $m.py --selftest 2>&1 | tail -1
done

# 端到端探针（不需要真硬件）
./.venv/Scripts/python.exe tools/objects_e2e.py      # 阶段 19 变化感知
./.venv/Scripts/python.exe tools/vision_e2e.py       # 阶段 18 视觉
./.venv/Scripts/python.exe tools/behave_e2e.py       # 阶段 16 主动行为
./.venv/Scripts/python.exe tools/fullduplex_e2e.py   # 阶段 17 全双工

# 固件
cd .. && /c/Users/hp/.platformio/penv/Scripts/pio.exe run
```

当前基线：presence 6/6 · behave 8/8 · vision 7/7 · aec 7/7 · 四个探针全绿 ·  
固件 SUCCESS（RAM 6.1% / Flash 10.0%）。

---

## 九、常见坑（照着对）

| 现象                                             | 原因                   | 怎么办                                                              |
| ---------------------------------------------- | -------------------- | ---------------------------------------------------------------- |
| `ModuleNotFoundError: No module named 'numpy'` | 用了系统 Python          | 改用 `./.venv/Scripts/python.exe`                                  |
| `WinError 10061 拒绝连接`                          | 假设备没起来               | 另开一个终端起 `tools/mock_esp32.py`，端口要对得上                             |
| 我指定的 WAV **没被用上**                              | 加了 `--demo`，它强制覆盖了音源 | 去掉 `--demo`，改用 `--vision-source demo` + `--audio-source xxx.wav` |
| 麦克风读到 `-96.0` dBFS、一片死寂                        | Windows 麦克风隐私权限关着    | 设置 → 隐私和安全性 → 麦克风 → 放行。**这是软件开关，不是设备坏了**                         |
| `edge-tts 403`                                 | 服务端拒绝                | 用 `--tts sapi`                                                   |
| 弹窗挡住/卡住                                        | OpenCV 预览窗口          | 加 `--no-window`                                                  |
| 每次都报同一屏东西                                      | 这是**变化感知**在暖机        | `--objects-warmup` 调大，或 `--no-objects` 关掉                        |
| 它一直在自言自语                                       | `--narrate` 是 `all`  | 开 `--asr` 会自动变 `quiet`（少说多听）；或 `--narrate quiet`                 |
| 日志刷得太快                                         | 每帧都输出                | 默认有变化才输出；`--verbose` 才是每帧                                        |

---

## 十、等你有了硬件（回真机）

现在的演示**没有一条需要买设备**。等板子到手，按这个顺序：

1. 照 `include/config.h` 接线（**只改这一个文件**，PC 端一行都不用动）
   - ⚠️ 步进电机**必须加 ULN2003 驱动板**，四个 GPIO 直驱线圈在仿真里可以、真机会烧
2. `pio run` 烧固件，把假设备换成真串口：`--port COM3`（假设备是 `--port socket://...`）
3. 修好麦克风权限后，把 `--audio-source xxx.wav` 去掉，改回真麦克风
4. 再跑本手册第二节到第七节——**命令只改 `--port` 和 `--audio-source` 两处**

**只有真机能验的东西**（仿真测不出来，手册里所有数字都不覆盖这些）：

- `aec.py` 的样本时钟对**真**声卡抖动的表现 → 看`时间轴跳变 N 次`、`预热 N 块`
- AEC 滤波器长度该设多少 → `python tools/aec_live.py --yes`（会出声）
- `--objects-enter/leave/count` 该设多少 → 看`闪一下没够着`（本机真摄像头实测 **11**，  
  这就是"检测器在抖"的证据；它大就先调 `--vision-conf`）
