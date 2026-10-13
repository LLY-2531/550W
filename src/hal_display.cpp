#include "hal_display.h"
#include "config.h"

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <math.h>

namespace {                      // 匿名 namespace：对象只在本文件可见，外部碰不到
Adafruit_SSD1306 sOled(OLED_W, OLED_H, &Wire, -1);

void drawBars(uint32_t t, int adc) {
  const int n = 12, w = 4, gap = 2, x0 = 52, y0 = 46;
  int active = adc * n / 100 / 41;                 // 0..12 段
  for (int i = 0; i < n; i++) {
    int x = x0 + i * (w + gap);
    if (i < active) {
      int h = 6 + ((i + (int)(t / 120)) % 3) * 4;
      sOled.fillRect(x, y0 + (16 - h), w, h, SSD1306_WHITE);
    } else {
      sOled.drawRect(x, y0 + 13, w, 3, SSD1306_WHITE);
    }
  }
}
}  // namespace

namespace hal {

void displayInit() {
  Wire.begin(OLED_SDA, OLED_SCL);
  Wire.setClock(400000);                          // I2C 提速，动画才跟得上
  if (!sOled.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("[ERR] OLED init failed");
    while (true) delay(1000);
  }
  sOled.clearDisplay();
  sOled.setTextColor(SSD1306_WHITE);
  sOled.display();
}

void displayRender(MossState st, const char* line1, int adcRaw) {
  const StateInfo& info = kStates[st];
  const uint32_t t = millis();

  sOled.clearDisplay();

  // ---------- MOSS 的眼睛 ----------
  const int cx = 26, cy = 30, r = 15;
  bool blink = ((t % 3400) < 110) && (st == ST_IDLE || st == ST_BOOT);

  if (blink) {
    sOled.fillRect(cx - r + 3, cy - 1, 2 * (r - 3), 3, SSD1306_WHITE);
  } else {
    sOled.drawCircle(cx, cy, r, SSD1306_WHITE);
    sOled.drawCircle(cx, cy, r - 1, SSD1306_WHITE);

    int dx = 0;
    switch (st) {
      case ST_IDLE:   dx = (int)(8  * sinf(t / 700.0f)); break;
      case ST_LISTEN: dx = (int)(11 * sinf(t / 260.0f)); break;
      case ST_THINK:  dx = (int)(6  * sinf(t / 400.0f)); break;
      case ST_SPEAK:  dx = (int)(13 * sinf(t / 180.0f)); break;
      case ST_VISION: dx = -9; break;
      case ST_MOTION: dx =  9; break;
      default:        dx =  0;
    }
    sOled.fillCircle(cx + dx, cy, 6, SSD1306_WHITE);

    if (st == ST_THINK) {                        // 想问题时，外面绕一个点
      float a = (t % 1200) / 1200.0f * 6.2831853f;
      sOled.fillCircle(cx + (int)((r + 5) * cosf(a)),
                       cy + (int)((r + 5) * sinf(a)), 2, SSD1306_WHITE);
    }
    if (st == ST_ERROR) {                        // 报错时半闭眼
      sOled.fillRect(cx - r, cy - r - 1, 2 * r + 2, 9, SSD1306_BLACK);
      sOled.drawLine(cx - r + 2, cy - 3, cx + r - 2, cy - 3, SSD1306_WHITE);
    }
  }

  // ---------- 右侧文字 ----------
  sOled.setTextSize(1);
  sOled.setCursor(50, 6);
  sOled.print(info.name);
  sOled.setCursor(50, 18);
  sOled.print(info.hint);
  if (line1 && line1[0]) {
    sOled.setCursor(50, 30);
    sOled.print(line1);
  }

  drawBars(t, adcRaw);
  sOled.display();
}

}  // namespace hal
