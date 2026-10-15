# MOSS 550W · 硬件烧录与上机指南

> 本文回答四个问题：**烧什么、怎么烧、怎么改、买什么、怎么接线**。
> 项目分两层：**PC 端大脑**（电脑上跑，不用烧）+ **ESP32 固件**（烧进芯片）。

---

## 目录

1. 先搞懂：什么要烧、什么不用烧
2. 编译固件（生成 bin）
3. 怎么烧录（3 种方法）
4. 怎么修改固件（改完重新编译）
5. 购物清单
6. 接线图
7. 两个必须提前知道的坑
8. 烧录后的联调步骤

---

## 一、先搞懂：什么要烧、什么不用烧

| 层 | 运行位置 | 要不要烧录 | 说明 |
|---|---|---|---|
| **PC 端大脑**（`pc_brain/`） | 你的电脑 | ❌ 不用 | 直接 `python moss_brain.py` 跑，见 [INSTALL.md](INSTALL.md) |
| **ESP32 固件**（`src/`+`include/`） | ESP32 芯片 | ✅ 要烧录 | 烧 3 个 bin 文件（已编译好） |

> 只有 ESP32 那一层需要「烧录」。PC 端大脑永远在电脑上运行，不存在烧录。

---

## 二、编译固件（生成 bin）

> ⚠️ 固件 bin 文件**不随仓库分发**（它们是编译产物，会随代码更新而变化）。拿到源码后，用下面命令自己编译生成：

```bash
pio run          # 编译，生成下面 3 个 bin
```

编译产物在 `.pio/build/esp32-s3-devkitc-1/`：

| 文件 | 大小 | 作用 | 烧录地址 |
|---|---|---|---|
| `bootloader.bin` | 15 KB | 启动引导 | `0x0` |
| `partitions.bin` | 3 KB | 分区表 | `0x8000` |
| `firmware.bin` | 335 KB | 主程序（你的代码） | `0x10000` |

> **目标芯片**：ESP32-S3-DevKitC-1-**N8**（8MB Flash，无 PSRAM）。当前固件 RAM 6.1% / Flash 10.0%。

---

## 三、怎么烧录（3 种方法，选一种）

### 方法 A：PlatformIO 一键烧录（推荐）

1. USB 线连上 ESP32-S3 板；
2. VS Code 打开项目，底部点 **→（上传）**，或终端跑：

```bash
pio run -t upload
```

### 方法 B：用 esptool 烧录那 3 个 bin

```bash
pip install esptool
cd .pio/build/esp32-s3-devkitc-1
esptool.py --chip esp32s3 --port COM3 --baud 460800 write_flash \
  0x0 bootloader.bin \
  0x8000 partitions.bin \
  0x10000 firmware.bin
```

> `COM3` 换成你电脑识别出的串口号（设备管理器 → 端口 里看）。

### 方法 C：图形化工具（不想碰命令行）

去乐鑫官网下载 **ESP Flash Download Tool**，GUI 里选 `esp32s3`，填三个文件对应地址（0x0 / 0x8000 / 0x10000），点 START。

> 三个 bin 文件都在 `.pio/build/esp32-s3-devkitc-1/` 目录下。

---

## 四、怎么修改固件（改完重新编译）

固件的所有引脚和功能都集中在 `include/config.h`。**换引脚、改灯环数量、改唤醒行为，通常只改这一个文件。**

### 4.1 常见要改的地方

| 想改什么 | 改哪里 |
|---|---|
| 某个模块的引脚 | `include/config.h` 里对应的 `#define` |
| 灯环数量 | `RING_NUM`（默认 12） |
| 屏幕尺寸 | `OLED_W` / `OLED_H`（默认 128×64） |
| 串口波特率 | `SERIAL_BAUD`（默认 115200） |

### 4.2 改完重新编译 + 烧录

```bash
pio run              # 编译，生成新的 firmware.bin
pio run -t upload    # 编译并烧录到板子
```

> 编译产物在 `.pio/build/esp32-s3-devkitc-1/`，`firmware.bin` 就是新的主程序。

### 4.3 如果只改了 PC 端（`pc_brain/`）

**完全不用重新编译固件。** 固件和 PC 大脑是两层，PC 端代码改了直接重跑 `moss_brain.py` 就生效，固件一行不用动。

---

## 五、购物清单

### 必买（设备端一套，约 ¥65–100）

| # | 硬件 | 型号/规格 | 数量 | 对应引脚 | 参考价 |
|---|---|---|---|---|---|
| 1 | 主控板 | **ESP32-S3 DevKitC-1-N8** | 1 | — | ¥25–35 |
| 2 | 屏幕（眼睛） | **SSD1306 OLED 128×64 I2C**，地址 0x3C | 1 | SDA=8, SCL=9 | ¥8–12 |
| 3 | 灯环 | **WS2812B 灯环 12 颗** | 1 | `RING_PIN=48` ⚠️ | ¥8–15 |
| 4 | 步进电机（转头） | **28BYJ-48**（5V 减速步进） | 1 | A=10/11, B=12/13 | ¥5–8 |
| 5 | 电机驱动板 | **ULN2003**（⚠️ 必须） | 1 | 接在 GPIO 与电机之间 | ¥3–5 |
| 6 | 蜂鸣器 | **无源蜂鸣器** | 1 | `BUZZ_PIN=14` | ¥1–2 |
| 7 | 按键 | 轻触按键 + 10kΩ 电阻 | 1 | `BTN_PIN=4` | ¥0.5 |
| 8 | 电位器 | 10kΩ 旋钮电位器 | 1 | `POT_PIN=5` | ¥1–2 |
| 9 | 面包板 + 杜邦线 | 面包板 ×1 + 公母/公公线一包 | 1 套 | — | ¥10–15 |
| 10 | USB 数据线 | USB-C，**要能传数据**（非纯充电线） | 1 | — | ¥5–10 |

### 可选（进阶，第二批再加）

| # | 硬件 | 用途 | 参考价 |
|---|---|---|---|
| A | I2S 功放 + 喇叭（MAX98357） | MOSS 声音更响 | ¥15–25 |
| B | INMP441 麦克风模块 | 拾音（更贴近实体） | ¥10–15 |
| C | 电池 + 电源模块 | 脱机供电 | ¥15–30 |

---

## 六、接线图

> 引脚定义集中在 `include/config.h`，下面按模块列出。**GND 全部共地**（所有模块 GND 接到 ESP32 的 GND）。

### 6.1 OLED 屏幕（眼睛）

| OLED 引脚 | ESP32 引脚 | 说明 |
|---|---|---|
| VCC | 3V3 | 供电（3.3V） |
| GND | GND | 共地 |
| SDA | **GPIO 8** | I2C 数据 |
| SCL | **GPIO 9** | I2C 时钟 |

### 6.2 WS2812B 灯环

| 灯环引脚 | ESP32 引脚 | 说明 |
|---|---|---|
| VCC | 5V | 供电（灯环吃 5V，不要接 3V3） |
| GND | GND | 共地 |
| DIN | **GPIO 48** | 数据线 ⚠️ 见「坑 2」 |

### 6.3 步进电机 + ULN2003（转头）

> ⚠️ **必须经 ULN2003**，电机线圈绝不能直接接 GPIO。

| ULN2003 引脚 | 接什么 |
|---|---|
| IN1 | ESP32 **GPIO 10** |
| IN2 | ESP32 **GPIO 11** |
| IN3 | ESP32 **GPIO 12** |
| IN4 | ESP32 **GPIO 13** |
| GND | ESP32 GND（共地） |
| VCC | 5V（给电机供电） |
| 输出 A/B/C/D | 电机 4 根线（蓝/粉/黄/橙，按序） |
| 电机红线 | 5V（电机公共端） |

### 6.4 蜂鸣器

| 蜂鸣器引脚 | ESP32 引脚 |
|---|---|
| 正极 | **GPIO 14**（经 100Ω 电阻更稳） |
| 负极 | GND |

### 6.5 按键 + 电位器（交互）

| 元件 | 接法 |
|---|---|
| 按键一脚 | **GPIO 4**（另一脚接 GND，GPIO 内部上拉或外接 10kΩ 上拉到 3V3） |
| 电位器中间脚 | **GPIO 5**（模拟输入） |
| 电位器两边 | 分别接 3V3 和 GND |

### 6.6 供电总览

| 电源轨 | 接哪些模块 |
|---|---|
| **3V3** | OLED、按键上拉、电位器 |
| **5V** | 灯环、ULN2003 电机驱动 |
| **GND** | 所有模块共地 |

> 5V 供电建议用 USB 直供或独立 5V 电源；若灯环 + 电机同时满负荷，板载 5V 可能不够，可给电机单独供电（共地即可）。

---

## 七、两个必须提前知道的坑

### 坑 1：步进电机必须接 ULN2003

现在 `config.h` 里 `STEP_A_P/A_N/B_P/B_N` 是 4 个 GPIO「直连」——那是 **Wokwi 仿真简化**。真机若把 28BYJ-48 线圈直接接 GPIO，**电流过大会烧坏板子**。正确接法：

```
GPIO 10/11/12/13 → ULN2003 输入端 → 电机线圈
```

### 坑 2：RING_PIN 48 是板载灯引脚

GPIO48 在 ESP32-S3 DevKitC-1 上是**板载 RGB LED**。两个选择：

- **直接用板载灯**当「眼睛光环」（不用买灯环，省 ¥8–15）；
- 或改 `include/config.h` 里 `#define RING_PIN 48` 为一个空闲 GPIO（如 16/17），再外接 WS2812 灯环，然后**重新编译烧录**。

---

## 八、烧录后的联调步骤

1. 烧录完成 → ESP32 板载灯亮、OLED 显示 MOSS 开机画面；
2. 接上模块（OLED → 灯环 → 蜂鸣器 → 电机，逐个验证）；
3. PC 端启动大脑，接串口：

```bash
cd pc_brain
python moss_brain.py --llm api --api-base "https://api.deepseek.com/v1" --model deepseek-chat --port COM3 --no-echo-guard
```

> `--port COM3` 换成 ESP32 的串口号。此时 PC 大脑的命令会通过串口发给实体，OLED 显示状态、灯环变色、电机转头。

4. 对着麦克风说「BOSS，现在几点了」，观察 OLED / 灯环 / 蜂鸣器 / 电机联动。

---

## 附：一页纸上手清单

```bash
# 1. 编译固件（生成 3 个 bin）
pio run
cd .pio/build/esp32-s3-devkitc-1

# 2. 烧录（方法 B，esptool）
pip install esptool
esptool.py --chip esp32s3 --port COM3 --baud 460800 write_flash \
  0x0 bootloader.bin 0x8000 partitions.bin 0x10000 firmware.bin

# 3. 联调（PC 端）
cd pc_brain
python moss_brain.py --llm api --api-base "https://api.deepseek.com/v1" \
  --model deepseek-chat --port COM3 --no-echo-guard
```
