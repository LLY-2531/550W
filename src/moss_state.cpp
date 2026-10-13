#include "moss_state.h"

const StateInfo kStates[ST_COUNT] = {
  {"BOOT",   "self test",   255, 255, 255, A_BREATHE, 2400,  880},
  {"IDLE",   "hi, MOSS",      0,  90, 255, A_BREATHE, 3200,  440},
  {"LISTEN", "listening",     0, 220, 170, A_WAVE,     900,  660},
  {"THINK",  "thinking",    255, 150,   0, A_CHASE,   1400,  520},
  {"SPEAK",  "speaking",    150,  60, 255, A_WAVE,     900,  740},
  {"VISION", "camera on",     0, 200, 255, A_FLOW,    1600,  620},
  {"MOTION", "arm moving",  255, 100,   0, A_SCAN,    1800,  300},
  {"ERROR",  "fault",       255,  30,  30, A_BLINK,    500,  160},
};


MossState stateCycle(MossState s) {
  if (s < ST_IDLE || s >= ST_ERROR) return ST_IDLE;
  return (MossState)(s + 1);
}
