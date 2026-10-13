#pragma once
#include <stdint.h>
#include "moss_state.h"

namespace hal {
void audioInit();
void audioBeep(uint16_t freq, uint16_t ms);   // 非阻塞"嘀"一声
void audioOnState(MossState st);              // 按状态表播提示音
void audioSilence();                          // 【阶段14】立刻停声（串口 STOP 用）
void audioTick();                             // 每帧调用，负责"到点静音"
}
