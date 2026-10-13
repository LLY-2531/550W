#pragma once
#include <stdint.h>
#include "moss_state.h"

namespace hal {
void motionInit();
void motionTick(MossState st);   // 每帧调用，按状态决定怎么转
int  motionSteps();              // 累计步数（调试用）

// ------------------------------------------------------------
//  【阶段 14】手动转动：让 MOSS 能"拨动"东西
// ------------------------------------------------------------
// motionSpin 排队一次转动，走完自动停（不是永久接管）。
// dir > 0 = 顺时针，dir < 0 = 逆时针。
//
// 语义上刻意选**覆盖**而不是**累加**：
//   如果模型先说"转半圈"、又说"转一圈"，人的预期是最终转一圈；
//   累加的话会变成转一圈半，非常反直觉。所以新命令直接替换旧命令。
// ------------------------------------------------------------
void motionSpin(long steps, int8_t dir);
void motionStop();               // 清掉待走步数（串口 STOP 用）
long motionQueue();              // 还剩多少步没走完（调试/回报用）
}
