#include "hal_motion.h"
#include "config.h"

#include <Arduino.h>

namespace {

const uint8_t kPins[4] = { STEP_A_P, STEP_A_N, STEP_B_P, STEP_B_N };

// 四相全步励磁序列。每一行是一"步"，4 列对应 A+ / A- / B+ / B-
// 每步只改变一个线圈的电流方向，磁场就转 1/4 圈，转子被"拽"着走一小格。
const uint8_t kSeq[4][4] = {
  {1, 0, 1, 0},
  {0, 1, 1, 0},
  {0, 1, 0, 1},
  {1, 0, 0, 1},
};

uint8_t  sPhase      = 0;     // 当前在第几步（0..3）
int      sPos        = 0;     // 累计步数，正=顺时针
uint32_t sLastStepAt = 0;     // 上次迈步的时刻
uint32_t sFlipAt     = 0;     // 上次换方向的时刻
int8_t   sDir        = 1;     // 扫描方向

// 【阶段 14】手动待走步数（恒为 >= 0，只减不增；走完归零）
long     sQueue      = 0;
int8_t   sQueueDir   = 1;

void applyPhase() {
  for (int i = 0; i < 4; i++) digitalWrite(kPins[i], kSeq[sPhase][i]);
}

void stepOnce(int8_t dir) {
  sPhase = (uint8_t)((sPhase + (dir > 0 ? 1 : 3)) & 3);   // +1 或 -1（模 4）
  sPos  += dir;
  applyPhase();
}

void release() {                                            // 断电，电机不发热、可自由转
  for (int i = 0; i < 4; i++) digitalWrite(kPins[i], LOW);
}

}  // namespace

namespace hal {

void motionInit() {
  for (int i = 0; i < 4; i++) {
    pinMode(kPins[i], OUTPUT);
    digitalWrite(kPins[i], LOW);
  }
}

void motionSpin(long steps, int8_t dir) {
  sQueue    = (steps < 0) ? -steps : steps;
  sQueueDir = (dir >= 0) ? 1 : -1;
}

void motionStop() { sQueue = 0; release(); }

long motionQueue() { return sQueue; }

void motionTick(MossState st) {
  const uint32_t t = millis();

  // ---------- 【阶段 14】手动步数优先 ----------
  // 只要队列里还有步数，就先把人工指令走完，期间不做任何状态动画。
  // 速度与下面 ST_MOTION 一致（4ms/步），保持手感统一。
  if (sQueue > 0) {
    if (t - sLastStepAt >= 4) {
      stepOnce(sQueueDir);
      sLastStepAt = t;
      sQueue--;
      if (sQueue == 0) release();   // 走完就断电，别让线圈干耗发热
    }
    return;
  }

  // ---------- 自动挡：跟着状态走（阶段 5 的逻辑）----------
  switch (st) {
    case ST_MOTION:                                   // 连续正转（机械臂/眼睛旋转）
      if (t - sLastStepAt >= 4) { stepOnce(+1); sLastStepAt = t; }
      break;

    case ST_VISION:                                   // 左右扫描，像在找人
      if (t - sLastStepAt >= 6) {
        stepOnce(sDir);
        sLastStepAt = t;
        if (t - sFlipAt > 1200) { sDir = -sDir; sFlipAt = t; }
      }
      break;

    case ST_THINK:                                    // 小幅度抖动，像在"想事情"
      if (t - sLastStepAt >= 90) {
        stepOnce(sDir);
        sLastStepAt = t;
        if (t - sFlipAt > 400) { sDir = -sDir; sFlipAt = t; }
      }
      break;

    default:                                          // 其他状态：停住
      release();
      break;
  }
}

int motionSteps() { return sPos; }

}  // namespace hal
