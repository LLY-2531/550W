#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
假 LLM 服务端（离线验证工具）  ·  流浪地球 550W 复刻项目 · 阶段 10（阶段 13 扩展）
============================================================
作用：扮演一个「OpenAI 兼容的 chat/completions 接口」，让 moss_brain.py 的
--llm api 这条路**不需要真 API Key、不需要联网**就能验证。

它能做到：
    1. 按 OpenAI 的格式回一个正常的响应
    2. 把收到的 system / user 提示词打印出来（用来核对"状态有没有传进模型"）
    3. 从提示词里认出当前状态，回一句对应的中文台词 + ASCII 标签
    4. 故意把 JSON 包在 ```json 代码块里 —— 用来验证客户端的容错解析
    5. **阶段 13**：认得出提示词里的「★ 用户刚刚对你说」并决定**要不要调工具**；
       第二轮（带着工具结果回来的那次）它会读结果、据此回答。
       这样才能在完全离线的条件下验证"工具调用"的完整两轮链路。

用法：
    # 终端 1
    python tools/mock_llm_server.py --port 8000 --verbose
    # 终端 2
    python moss_brain.py --demo --llm api \
        --api-base http://localhost:8000/v1 --api-key fake --model mock-1

    # 也可以单独测语言层
    python mind.py --llm api --api-base http://localhost:8000/v1 --api-key fake
"""

import argparse
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPLIES = {
    "BOOT":   ("MOSS 已上线，各单元自检通过。", "boot ok"),
    "IDLE":   ("我在待机，随时听候指令。", "standing by"),
    "LISTEN": ("我听到声音了，正在采集。", "listening"),
    "THINK":  ("让我算一下。", "computing"),
    "SPEAK":  ("你在对我说话，我在听。", "you talking"),
    "VISION": ("画面里有一个人，我看见你了。", "i see you"),
    "MOTION": ("检测到画面在动。", "motion"),
    "ERROR":  ("出现异常，请检查硬件。", "fault"),
}

# 阶段 13：假服务端的"意图识别"。命中就真的返回一个 tool 请求，
# 让 moss_brain 去执行真工具、再带着结果回来问第二轮。
# 每条 = (正则, 工具名, 参数构造函数(match) -> dict)
_TURNS = {"半圈": 0.5, "一圈": 1.0, "一转": 1.0,
          "两圈": 2.0, "三圈": 3.0, "几圈": 1.0}

TOOL_HINTS = [
    # ---------- 阶段 14：动作类（"动手"）----------
    # 放在最前面：动作意图比查询意图更"字面"，先匹配不容易被别的规则抢走。
    (r"(?:灯|灯环|环形灯)[^，,。！？]{0,6}?([红橙黄绿青蓝紫白灭])",
     "set_led", lambda m: {"color": m.group(1)}),
    (r"(?:变|亮|调|换)(?:成|为)?\s*([红橙黄绿青蓝紫白灭])色?",
     "set_led", lambda m: {"color": m.group(1)}),
    (r"(?:电机|马达|转)[^，,。！？]{0,4}?(半圈|一圈|一转|两圈|三圈|几圈)",
     "turn_motor", lambda m: {"turns": _TURNS.get(m.group(1), 0.5)}),
    (r"(转一下|转一转|转动|旋转)", "turn_motor", lambda m: {"turns": 0.5}),
    (r"(响一声|响一下|嘀一声|嘀一下|哔一声|啾一声|提示音|叫一声)",
     "beep", lambda m: {}),
    (r"(急停|停下来|停下|别动|别转|住手|安静)", "stop_all", lambda m: {}),
    (r"(?:屏幕|OLED|oled)[^，,。！？]{0,6}?(?:写|显示|打)(?:上|出)?\s*([A-Za-z0-9 _.\-]{1,13})",
     "show_text", lambda m: {"text": m.group(1)}),
    (r"(?:设备|板子|单片机)[^，,。！？]{0,4}?(?:状态|在线|连上|怎么样|好了吗)",
     "device_status", lambda m: {}),

    # ---------- 查询类（阶段 13）----------
    (r"(几点|时间|日期|几号|星期几|今天是)", "get_time", lambda m: {}),
    (r"(?:我叫|我的名字是|我的名字叫)([\u4e00-\u9fffA-Za-z0-9]{1,12})",
     "remember", lambda m: {"key": "用户的名字", "value": m.group(1)}),
    (r"(我是谁|我叫什么|还记得我|你记得我)", "recall", lambda m: {"query": ""}),
    (r"([\u4e00-\u9fff]{2,8}?)(?:市|的)?(?:的)?(?:天气|气温)",
     "get_weather", lambda m: {"city": m.group(1) or ""}),
    (r"(看看我|画面里有什么|看得见我吗|有人吗)", "look", lambda m: {}),
]

VERBOSE = False
WRAP_FENCE = True
_REQUEST_N = 0


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass                                       # 关掉默认的访问日志，输出自己控制

    def _send_json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        global _REQUEST_N
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n).decode("utf-8", "replace")
        try:
            req = json.loads(raw)
        except Exception:
            req = {}

        msgs = req.get("messages", [])
        user_text = ""
        for m in msgs:
            if m.get("role") == "user":
                user_text = m.get("content", "")
        sys_text = next((m.get("content", "") for m in msgs if m.get("role") == "system"), "")
        _REQUEST_N += 1

        has_tools = "可用工具：" in sys_text
        has_facts = "你知道的关于用户的事" in user_text
        is_round2 = "【你刚才申请调用的工具已经执行完毕】" in user_text

        if VERBOSE:
            print("-" * 70)
            print(f"[收到请求 #{_REQUEST_N}] {self.path}  model={req.get('model')}  "
                  f"messages={len(msgs)}  auth={self.headers.get('Authorization', '(无)')}"
                  f"  回合={'第二轮(带工具结果)' if is_round2 else '第一轮'}")
            print(f"  system 提示词长度：{len(sys_text)} 字符"
                  f"（含 ASCII 约束：{'是' if 'ASCII' in sys_text else '否'}）"
                  f"（含工具清单：{'是' if has_tools else '否'}）")
            if has_tools:
                names = re.findall(r"^- (\w+)：", sys_text, re.M)
                print(f"  工具清单：{names}")
            print("  user 提示词：")
            for line in user_text.splitlines():
                print(f"    | {line}")

        # ---------- 第二轮：已经带着工具结果回来了 ----------
        if is_round2:
            tn = re.search(r"工具：(\S+)", user_text)
            tv = re.search(r"结果：(.+)", user_text)
            name = tn.group(1) if tn else "?"
            val = tv.group(1).strip() if tv else "（没读到结果）"
            say = {"get_time": f"{val}。",
                   "get_weather": f"查到了：{val}",
                   "remember": "好，我记下来了。",
                   "recall": f"我记得：{val}",
                   "look": f"我看了一眼：{val}",
                   # 阶段 14：动作类工具的结果本身就是完整的一句话，直接照说
                   "set_led": val, "turn_motor": val, "beep": val,
                   "show_text": val, "stop_all": val, "device_status": val,
                   }.get(name, f"查询完成：{val}")
            content = json.dumps({"say": say, "tag": "tool done"}, ensure_ascii=False)
            if VERBOSE:
                print(f"[第二轮] 读到工具 {name} 的结果 → {content!r}")
            return self._reply(req, content)

        # ---------- 第一轮 ----------
        q = re.search(r"★\s*用户刚刚对你说：\s*(.+)", user_text)
        if q:
            heard = q.group(1).strip()
            if VERBOSE:
                has_dialog = "最近的对话" in user_text
                print(f"[识别到提问] 用户说：{heard}")
                print(f"             提示词里带历史对话：{'是' if has_dialog else '否'}"
                      f"｜带长期事实：{'是' if has_facts else '否'}")

            # 开了工具 → 看看该不该调
            if has_tools:
                # 只申请**清单里真的有**的工具。否则没接串口时（没有 set_led），
                # 假服务端会去调一个不存在的工具，第二轮拿到"失败" —— 那是
                # 测试脚本的错，不是被测代码的错，很容易误导人。
                avail = set(re.findall(r"^- (\w+)：", sys_text, re.M))
                for pat, name, mk in TOOL_HINTS:
                    if name not in avail:
                        continue
                    m = re.search(pat, heard)
                    if m:
                        tool = {"name": name, "args": mk(m)}
                        content = json.dumps(
                            {"say": "", "tag": "", "tool": tool}, ensure_ascii=False)
                        if VERBOSE:
                            print(f"[决定调工具] {tool}")
                        return self._reply(req, content)

            say = f"收到，你说的是{heard}。"
            content = json.dumps({"say": say, "tag": "heard"}, ensure_ascii=False)
        else:
            mode = "IDLE"
            m = re.search(r"当前状态：\s*([A-Z]+)", user_text)
            if m:
                mode = m.group(1)
            say, tag = REPLIES.get(mode, ("……", "online"))
            content = json.dumps({"say": say, "tag": tag}, ensure_ascii=False)

        if VERBOSE:
            print(f"[回复] {'问答' if q else '状态汇报'}  →  {content!r}")
        return self._reply(req, content)

    def _reply(self, req, content):
        if WRAP_FENCE:                             # 故意加个 ```json 外壳，考验客户端解析
            content = "```json\n" + content + "\n```"
        if VERBOSE:
            print(f"[发出] {content!r}")
        self._send_json(200, {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.get("model", "mock-1"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })


def main():
    global VERBOSE, WRAP_FENCE
    p = argparse.ArgumentParser(description="假 OpenAI 兼容服务端（离线验证用）")
    p.add_argument("--port", type=int, default=8000, help="监听端口，默认 8000")
    p.add_argument("--host", default="127.0.0.1", help="监听地址")
    p.add_argument("--verbose", action="store_true", help="打印收到的提示词和回复")
    p.add_argument("--no-fence", action="store_true",
                   help="回复不加 ```json 外壳（默认加，用来测客户端容错）")
    p.add_argument("--once", type=float, default=0,
                   help="跑多少秒后自动退出，0=一直跑")
    a = p.parse_args()
    VERBOSE = a.verbose
    WRAP_FENCE = not a.no_fence

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print("=" * 70)
    print(f"  假 LLM 服务已启动：http://{a.host}:{a.port}/v1/chat/completions")
    print("  用法示例：")
    print(f"    python moss_brain.py --demo --llm api \\")
    print(f"        --api-base http://{a.host}:{a.port}/v1 --api-key fake --model mock-1")
    print("=" * 70, flush=True)

    if a.once > 0:
        threading.Thread(target=lambda: (time.sleep(a.once), srv.shutdown()),
                         daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    print("\n[假 LLM] 已退出。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
