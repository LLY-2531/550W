#pragma once
#include <stdint.h>

// 设备所有状态。顺序 = 按钮循环的顺序。
enum MossState : uint8_t {
  ST_BOOT, ST_IDLE, ST_LISTEN, ST_THINK,
  ST_SPEAK, ST_VISION, ST_MOTION, ST_ERROR,
  ST_COUNT                       // 状态总数，放最后，只用来数数
};

// 灯环动画类型
enum Anim : uint8_t { A_BREATHE, A_CHASE, A_WAVE, A_BLINK, A_FLOW, A_SCAN };

// 一个状态的"全部信息"
struct StateInfo {
  const char* name;    // OLED 第一行
  const char* hint;    // OLED 第二行
  uint8_t r, g, b;     // 灯环颜色
  Anim    anim;        // 灯环动画
  uint16_t period;     // 动画周期(ms)
  uint16_t beepHz;     // 【新增】进入该状态时的提示音，0 = 不响
};


// 状态表：这里只"声明"（extern），真正的定义在 moss_state.cpp
extern const StateInfo kStates[ST_COUNT];

// 循环切状态：IDLE → … → ERROR → 回 IDLE
MossState stateCycle(MossState s);
