#pragma once

// ============================================================
//  550W（MOSS）· 引脚配置
//  ----------------------------------------------------------
//  所有引脚只在这里定义一次。真机阶段只改这一个文件，
//  下面的业务代码一行都不用动。
// ============================================================

// OLED（I2C）
#define OLED_SDA   8
#define OLED_SCL   9
#define OLED_ADDR  0x3C
#define OLED_W     128
#define OLED_H     64

// WS2812 圆环灯组
#define RING_PIN   48
#define RING_NUM   12

// 交互
#define BTN_PIN        4
#define POT_PIN        5
#define LONG_PRESS_MS  700

// 串口
#define SERIAL_BAUD 115200
// 步进电机（A+/A-/B+/B- 四个线圈脚，仿真里直连 GPIO）
#define STEP_A_P   10
#define STEP_A_N   11
#define STEP_B_P   12
#define STEP_B_N   13

// 蜂鸣器
#define BUZZ_PIN   14
