#include <Arduino.h>
#include "config.h"
#include "moss_state.h"
#include "hal_display.h"
#include "hal_motion.h"
#include "hal_audio.h"
#include "hal_light.h"
#include "hal_input.h"

// 业务状态（设备"现在在想什么"）
MossState gState = ST_BOOT;
char gLine1[24] = ""; // OLED 第三行（TEXT 命令）
bool gDemo = false;   // 自动轮播
uint32_t gDemoAt = 0;

// 串口收命令用的缓冲区
char gCmd[40];
uint8_t gCmdLen = 0;
// ---------- 前置声明 ----------
void printHelp();
void handleCommand(char *cmd);
void pollSerial();
void pollDemo();

// ---------- 状态切换（唯一改状态的地方）----------
void setState(MossState s)
{
  gState = s;
  gDemoAt = millis();
  hal::audioOnState(s);
  Serial.printf("[STATE] %s\n", kStates[s].name);
}

// ---------- 串口 ----------
void pollSerial()
{
  while (Serial.available())
  {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r')
    {
      if (gCmdLen > 0)
      {
        gCmd[gCmdLen] = '\0';
        handleCommand(gCmd);
        gCmdLen = 0;
      }
    }
    else if (gCmdLen < sizeof(gCmd) - 1)
    {
      gCmd[gCmdLen++] = c;
    }
  }
}

// 【阶段 14】把 "255 40 0 5000" 这种参数串拆成整数。
// 遇到不是数字的地方就停（容错：模型可能把参数写成 "255,40,0" 或带单位）。
// 返回实际解析到的个数。
uint8_t parseInts(char *s, long *out, uint8_t maxn)
{
  uint8_t n = 0;
  char *p = s;
  while (p && *p && n < maxn)
  {
    while (*p == ' ' || *p == ',' || *p == ';')
      p++;
    bool numeric = (*p == '-' || *p == '+' || (*p >= '0' && *p <= '9'));
    if (!numeric)
      break;
    char *end = nullptr;
    long v = strtol(p, &end, 10);
    if (end == p)
      break; // 一个数字都没前进 → 防止死循环
    out[n++] = v;
    p = end;
  }
  return n;
}

void handleCommand(char *cmd)
{
  char *arg = strchr(cmd, ' ');
  if (arg)
  {
    *arg = '\0';
    arg++;
    while (*arg == ' ')
      arg++;
  }
  else
    arg = nullptr;

  for (char *p = cmd; *p; p++)
    *p = toupper(*p);

  if (!strcmp(cmd, "MODE") && arg)
  {
    for (char *p = arg; *p; p++)
      *p = toupper(*p);
    for (uint8_t i = 0; i < ST_COUNT; i++)
    {
      if (!strcmp(arg, kStates[i].name))
      {
        setState((MossState)i);
        return;
      }
    }
    Serial.printf("[CMD] unknown mode: %s\n", arg);
  }
  else if (!strcmp(cmd, "TEXT") && arg)
  {
    strncpy(gLine1, arg, sizeof(gLine1) - 1);
    gLine1[sizeof(gLine1) - 1] = '\0';
    Serial.printf("[CMD] TEXT = %s\n", gLine1);
  }
  else if (!strcmp(cmd, "DEMO"))
  {
    gDemo = (arg && (arg[0] == 'O' || arg[0] == 'o'));
    gDemoAt = millis();
    Serial.printf("[CMD] DEMO %s\n", gDemo ? "ON" : "OFF");
  }
  // ============================================================
  //  【阶段 14】执行器命令 —— 让上位机（PC 端 MOSS 大脑）能"动手"
  //  设计原则：所有手动动作都是**有时限**的，到点自动归还给状态动画。
  //  否则一条命令下去，之后 MODE 切状态都不生效了，设备就"死"在那儿。
  // ============================================================
  else if (!strcmp(cmd, "LED") && arg)
  {
    long v[4];
    uint8_t n = parseInts(arg, v, 4);
    if (n < 3)
    {
      Serial.println("[CMD] LED 用法: LED <r> <g> <b> [ms]   (0-255, ms 默认 3000)");
      return;
    }
    uint32_t ms = (n >= 4 && v[3] > 0) ? (uint32_t)v[3] : 3000;
    if (ms > 60000)
      ms = 60000;
    uint8_t r = (uint8_t)constrain(v[0], 0, 255);
    uint8_t g = (uint8_t)constrain(v[1], 0, 255);
    uint8_t b = (uint8_t)constrain(v[2], 0, 255);
    hal::lightOverride(r, g, b, ms);
    Serial.printf("[LED] %u %u %u hold %lums\n", r, g, b, (unsigned long)ms);
  }
  else if (!strcmp(cmd, "SPIN") && arg)
  {
    long v[2];
    uint8_t n = parseInts(arg, v, 2);
    if (n < 1)
    {
      Serial.println("[CMD] SPIN 用法: SPIN <steps> [CW|CCW]   (4096 步 ≈ 一圈)");
      return;
    }
    // 方向大小写都认：先把方向串复制出来统一转大写再找
    char dirbuf[24];
    strncpy(dirbuf, arg, sizeof(dirbuf) - 1);
    dirbuf[sizeof(dirbuf) - 1] = '\0';
    for (char *p = dirbuf; *p; p++)
      *p = toupper(*p);
    int8_t dir = strstr(dirbuf, "CCW") ? -1 : 1;

    long steps = (v[0] < 0) ? -v[0] : v[0];
    if (steps > 200000)
      steps = 200000; // 上限保护：别让模型把电机转上一整天
    hal::motionSpin(steps, dir);
    Serial.printf("[SPIN] %ld steps %s\n", steps, dir > 0 ? "CW" : "CCW");
  }
  else if (!strcmp(cmd, "BEEP") && arg)
  {
    long v[2];
    uint8_t n = parseInts(arg, v, 2);
    uint16_t f = (n >= 1) ? (uint16_t)constrain(v[0], 50, 8000) : 880;
    uint16_t ms = (n >= 2) ? (uint16_t)constrain(v[1], 10, 3000) : 200;
    hal::audioBeep(f, ms);
    Serial.printf("[BEEP] %uHz %ums\n", f, ms);
  }
  else if (!strcmp(cmd, "STOP"))
  {
    hal::motionStop();
    hal::audioSilence();
    hal::lightRelease();
    Serial.println("[STOP] motor off / buzzer off / light released");
  }
  else if (!strcmp(cmd, "PING"))
  {
    Serial.printf("[PONG] %s\n", kStates[gState].name);
  }
  else if (!strcmp(cmd, "HELP"))
  {
    printHelp();
  }
  else
  {
    printHelp();
  }
}

void printHelp()
{
  Serial.println("---- commands ----");
  Serial.println("MODE IDLE|LISTEN|THINK|SPEAK|VISION|MOTION|ERROR");
  Serial.println("TEXT <text>       OLED third line");
  Serial.println("DEMO ON|OFF       auto cycle states");
  Serial.println("-- actuators (stage 14) --");
  Serial.println("LED  <r> <g> <b> [ms]   ring solid color, auto-restore");
  Serial.println("SPIN <steps> [CW|CCW]   turn motor N steps (4096 ~= 1 turn)");
  Serial.println("BEEP <freq> <ms>        buzzer tone");
  Serial.println("STOP                    stop motor + buzzer + light override");
  Serial.println("PING                    -> [PONG] <state>");
  Serial.println("HELP              show this");
  Serial.println("------------------");
}

void pollDemo()
{
  if (!gDemo)
    return;
  if (millis() - gDemoAt < 3000)
    return;
  setState(stateCycle(gState));
}

// ---------- 开机 ----------
void setup()
{
  Serial.begin(SERIAL_BAUD);
  delay(200);

  hal::inputInit();
  hal::displayInit();
  hal::lightInit();
  hal::motionInit();
  hal::audioInit();

  Serial.println("[OK] MOSS ready");
  printHelp();

  setState(ST_BOOT);
  delay(1200);
  setState(ST_IDLE);
}

// ---------- 主循环：只做"读 → 决策 → 画" ----------
void loop()
{
  // 1. 读输入
  hal::BtnEvent ev = hal::inputTick();
  int pot = hal::potRead(); // 0..100

  // 2. 决策
  if (ev == hal::BTN_LONG)
  {
    setState(ST_IDLE);
    Serial.println("[BTN] long -> IDLE");
  }
  else if (ev == hal::BTN_CLICK)
  {
    setState(stateCycle(gState));
  }
  pollSerial();
  pollDemo();

  // 3. 输出
  uint8_t bright = (uint8_t)map(pot, 0, 100, 20, 200);
  hal::displayRender(gState, gLine1, pot);
  hal::lightTick();                 // 【阶段14】先处理"接管到期归还"，再渲染
  hal::lightRender(gState, bright);
  hal::motionTick(gState);
  hal::audioTick();
}
