#include "hal_light.h"
#include "config.h"

#include <Arduino.h>
#include <Adafruit_NeoPixel.h>
#include <math.h>

namespace {
Adafruit_NeoPixel sRing(RING_NUM, RING_PIN, NEO_GRB + NEO_KHZ800);

// 【阶段 14】手动接管状态。sOvOn = true 时，状态动画整段让位。
bool     sOvOn    = false;
uint8_t  sOvR     = 0;
uint8_t  sOvG     = 0;
uint8_t  sOvB     = 0;
uint32_t sOvUntil = 0;      // 到这个时刻自动归还
}  // namespace

namespace hal {

void lightInit() {
  sRing.begin();
  sRing.clear();
  sRing.show();
}

void lightOverride(uint8_t r, uint8_t g, uint8_t b, uint32_t ms) {
  sOvOn    = true;
  sOvR     = r;
  sOvG     = g;
  sOvB     = b;
  sOvUntil = millis() + (ms ? ms : 3000);
}

void lightRelease() { sOvOn = false; }

bool lightOverridden() { return sOvOn; }

void lightTick() {
  // 用"有符号差值"判断到期，这样即使 millis() 跨过 49.7 天回绕也不会误判。
  if (sOvOn && (int32_t)(millis() - sOvUntil) >= 0) {
    sOvOn = false;
  }
}

void lightRender(MossState st, uint8_t brightness) {
  // 亮度仍然受电位器控制 —— 手动接管只是"换颜色"，不该绕过整机的亮度旋钮。
  const float dim = brightness / 255.0f;

  // ---------- 手动接管：整圈纯色，压掉状态动画 ----------
  if (sOvOn) {
    for (int i = 0; i < RING_NUM; i++) {
      sRing.setPixelColor(i, sRing.Color((uint8_t)(sOvR * dim),
                                         (uint8_t)(sOvG * dim),
                                         (uint8_t)(sOvB * dim)));
    }
    sRing.show();
    return;
  }

  // ---------- 自动挡：跟着状态走（阶段 3 的逻辑，一行没改）----------
  const StateInfo& info = kStates[st];
  const uint32_t t  = millis();
  const float phase = (float)(t % info.period) / (float)info.period;

  for (int i = 0; i < RING_NUM; i++) {
    float k = 1.0f;
    switch (info.anim) {
      case A_BREATHE:
        k = 0.10f + 0.90f * (0.5f - 0.5f * cosf(phase * 6.2831853f));
        break;
      case A_CHASE: {
        int head = (int)(phase * RING_NUM);
        int d = (i - head + RING_NUM) % RING_NUM;
        k = (d == 0) ? 1.00f : (d == 1 ? 0.45f : (d == 2 ? 0.15f : 0.04f));
        break;
      }
      case A_WAVE: {
        float c = (RING_NUM - 1) / 2.0f;
        float d = fabsf(i - c) / c;
        k = 0.10f + 0.90f * fabsf(1.0f - 2.0f * fmodf(phase + d * 0.5f, 1.0f));
        break;
      }
      case A_BLINK:
        k = (t % 260 < 130) ? 1.0f : 0.04f;
        break;
      case A_FLOW: {
        int head = (int)(phase * RING_NUM);
        int d = (i - head + RING_NUM) % RING_NUM;
        k = (d < 4) ? (1.0f - d * 0.22f) : 0.05f;
        break;
      }
      case A_SCAN: {
        float s = (phase < 0.5f) ? phase * 2.0f : (1.0f - phase) * 2.0f;
        int p = (int)(s * (RING_NUM - 1));
        int d = abs(i - p);
        k = (d == 0) ? 1.0f : (d == 1 ? 0.35f : 0.05f);
        break;
      }
    }
    sRing.setPixelColor(i, sRing.Color((uint8_t)(info.r * k * dim),
                                       (uint8_t)(info.g * k * dim),
                                       (uint8_t)(info.b * k * dim)));
  }
  sRing.show();
}

}  // namespace hal
