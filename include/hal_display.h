#pragma once
#include <stdint.h>
#include "moss_state.h"

namespace hal {
void displayInit();                                      // 开机初始化一次
void displayRender(MossState st, const char* line1, int adcRaw);  // 每帧调用
}
