#pragma once
#include <stdint.h>
#include "moss_state.h"

namespace hal {
void lightInit();
void lightRender(MossState st, uint8_t brightness);      // brightness 0..255

// ------------------------------------------------------------
//  【阶段 14】手动接管：让 MOSS 有"手"
// ------------------------------------------------------------
// 状态动画是"自动挡"：MODE 一切，灯环就跟着状态呼吸/扫描。
// 手动接管是"人工挡"：整圈铺成指定颜色，ms 毫秒之后**自动归还**给状态动画。
//
// 为什么要超时归还，而不是一直锁住？
//   因为 MOSS 是个"活的"东西。你让它变红一下，它红完就该继续跟着状态呼吸；
//   如果永久锁死，之后 MODE 切到 SPEAK 都不变色了 ——
//   那不是 MOSS，那是一台坏掉的路由器。
// 唯一的"立刻归还"是 lightRelease()（对应串口 STOP 命令）。
// ------------------------------------------------------------
void lightOverride(uint8_t r, uint8_t g, uint8_t b, uint32_t ms);
void lightRelease();                 // 立刻归还给状态动画
bool lightOverridden();              // 此刻是不是手动接管中（调试/回报用）
void lightTick();                    // 每帧调用，负责"到点归还"
}
