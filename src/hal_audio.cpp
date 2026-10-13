#include "hal_audio.h"
#include "config.h"

#include <Arduino.h>

namespace {
bool     sOn    = false;
uint32_t sOffAt = 0;          // 该静音的时刻
}  // namespace

namespace hal {

void audioInit() {
  pinMode(BUZZ_PIN, OUTPUT);
  noTone(BUZZ_PIN);
}

void audioBeep(uint16_t freq, uint16_t ms) {
  if (freq == 0) return;
  tone(BUZZ_PIN, freq);       // 开始发声（不阻塞）
  sOn    = true;
  sOffAt = millis() + ms;     // 记下"什么时候该停"
}

void audioOnState(MossState st) {
  audioBeep(kStates[st].beepHz, 110);
}

// 【阶段 14】立刻停声。和 audioTick() 的"到点静音"是两条路：
// 一条是"时间到了自己停"，一条是"人喊停就停"。
void audioSilence() {
  noTone(BUZZ_PIN);
  sOn = false;
}

void audioTick() {
  if (sOn && millis() >= sOffAt) {
    noTone(BUZZ_PIN);         // 时间到 → 停
    sOn = false;
  }
}

}  // namespace hal
