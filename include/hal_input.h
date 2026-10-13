#pragma once
#include <stdint.h>

namespace hal {

// 按钮事件
enum BtnEvent : uint8_t { BTN_NONE, BTN_CLICK, BTN_LONG };

void     inputInit();
BtnEvent inputTick();     // 每帧调用一次，返回本帧发生的事件
int      potRead();       // 0..100

}
