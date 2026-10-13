#include "hal_input.h"
#include "config.h"

#include <Arduino.h>

namespace {                      // 记录按钮的"上一帧状态"，藏在本文件里
bool     sPrev   = false;
uint32_t sDownAt = 0;
}  // namespace

namespace hal {

void inputInit() {
  pinMode(BTN_PIN, INPUT_PULLUP);
}

BtnEvent inputTick() {
  bool now = (digitalRead(BTN_PIN) == LOW);      // 上拉，按下为 LOW
  uint32_t t = millis();
  BtnEvent ev = BTN_NONE;

  if (now && !sPrev) {
    sDownAt = t;                                  // 按下瞬间
  } else if (!now && sPrev) {                     // 松开瞬间算按住多久
    ev = (t - sDownAt >= LONG_PRESS_MS) ? BTN_LONG : BTN_CLICK;
  }
  sPrev = now;
  return ev;
}

int potRead() {
  int raw = analogRead(POT_PIN);                  // ESP32-S3: 0..4095
  return (raw * 100) / 4095;                      // 换算成 0..100
}

}  // namespace hal
