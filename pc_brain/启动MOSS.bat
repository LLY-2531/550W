@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   MOSS 550W 复刻  ·  一键启动
echo   真实摄像头 + 真实麦克风 + 语音对话
echo   喊「MOSS」唤醒，它会先应一声「在」
echo   按 Ctrl+C 或按 q 退出
echo ============================================
echo.

rem 你的 DeepSeek API Key 填在这里（替换下面这行的 your-deepseek-key）
set MOSS_API_KEY=your-deepseek-key

rem 下面这些默认值都已经固化进代码了，无需重复指定：
rem   --tts sapi  --asr whisper  --narrate quiet  --narrate-gap 9999
rem   --wake-ack voice  --aec off  --vision-source 0  --audio-source mic
rem 只需要指定：接入 DeepSeek 大模型（自由对话）
".venv\Scripts\python.exe" moss_brain.py --llm api --api-base "https://api.deepseek.com/v1" --model deepseek-chat --no-echo-guard

pause
