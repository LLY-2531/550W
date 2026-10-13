#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
假 ASR 服务端  ·  流浪地球 550W 复刻项目 · 阶段 12
============================================================
一个 OpenAI 兼容的 `/v1/audio/transcriptions` 假服务端。

用途和 `tools/mock_llm_server.py` 一样：让 `--asr api` 这条路
**不需要真 Key、不需要联网、不需要下模型**就能验证。

它不只是"随便回一句"，还会**真的把收到的 WAV 拆开量一下时长**
（WAV 头里的采样率 × 数据字节数），这样你能一眼看出：
    - 音频真的传过来了吗
    - 传过来的时长对不对（是不是把整句话都切进来了）

用法：
    # 终端 1
    python tools/mock_asr_server.py --port 8010 --verbose

    # 终端 2（用假 ASR 跑完整大脑）
    python moss_brain.py --audio-source 录音.wav --asr api ^
        --asr-api-base http://localhost:8010/v1 --asr-api-key fake

固定回话用 `--script`（用 | 分隔），不指定就用内置的几句。
"""

import argparse
import json
import re
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_SCRIPT = [
    "现在几点了",
    "帮我看看画面里有什么",
    "你是谁",
    "打开舱门",
    "谢谢大家",
]

STATE = {"n": 0, "lock": threading.Lock()}


def wav_duration(buf):
    """从 WAV 字节里读出时长（秒）。读不出来就返回 None。"""
    try:
        if buf[:4] != b"RIFF" or buf[8:12] != b"WAVE":
            return None
        pos = 12
        fmt = None
        data_len = None
        while pos + 8 <= len(buf):
            cid = buf[pos:pos + 4]
            size = struct.unpack("<I", buf[pos + 4:pos + 8])[0]
            body = buf[pos + 8:pos + 8 + size]
            if cid == b"fmt ":
                fmt = struct.unpack("<HHIIHH", body[:16])
            elif cid == b"data":
                data_len = size
            pos += 8 + size + (size & 1)
        if not fmt or data_len is None:
            return None
        _tag, channels, rate, _br, _ba, bits = fmt
        bytes_per_sec = rate * channels * bits // 8
        return data_len / float(bytes_per_sec) if bytes_per_sec else None
    except Exception:
        return None


def parse_multipart(body, boundary):
    """极简 multipart 解析：返回 {name: bytes}。"""
    out = {}
    sep = b"--" + boundary
    for part in body.split(sep):
        if not part or part in (b"--\r\n", b"--", b"\r\n"):
            continue
        head, _, data = part.partition(b"\r\n\r\n")
        m = re.search(rb'name="([^"]+)"', head)
        if not m:
            continue
        name = m.group(1).decode("utf-8", "replace")
        out[name] = data.rstrip(b"\r\n")
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):        # 关掉默认的访问日志，太吵
        pass

    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        ctype = self.headers.get("Content-Type") or ""

        if "boundary=" not in ctype:
            self._send(400, {"error": {"message": "缺少 multipart boundary"}})
            return
        boundary = ctype.split("boundary=", 1)[1].strip().strip('"').encode()

        fields = parse_multipart(body, boundary)
        audio = fields.get("file", b"")
        dur = wav_duration(audio)

        with STATE["lock"]:
            idx = STATE["n"]
            STATE["n"] += 1
        text = self.server.script[idx % len(self.server.script)]

        print("[假ASR] #%d 收到 %d 字节音频%s → 回话 %r"
              % (idx + 1, len(audio),
                 ("，时长 %.2fs" % dur) if dur else "（不是WAV？）", text),
              file=sys.stderr, flush=True)

        if self.server.verbose:
            print("        表单字段：%s" % {
                k: (v[:60].decode("utf-8", "replace") if k != "file" else "<%d字节>" % len(v))
                for k, v in fields.items()}, file=sys.stderr, flush=True)
            if dur is None:
                print("        ⚠️ 没解析出时长 —— 检查是不是真的发了 WAV", file=sys.stderr)
            else:
                print("        ✓ WAV 合法，时长 %.2f 秒（说明整句话都切进来了）" % dur,
                      file=sys.stderr)

        time.sleep(0.15)                       # 假装有点延迟，像真的
        self._send(200, {"text": text})

    def do_GET(self):
        self._send(200, {"ok": True, "served": STATE["n"]})


def main():
    p = argparse.ArgumentParser(description="假 OpenAI 兼容 ASR 服务端")
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--script", default=None,
                   help="固定回话，用 | 分隔。例：\"几点了|打开舱门\"")
    p.add_argument("--verbose", action="store_true", help="打印收到的表单字段")
    a = p.parse_args()

    script = a.script.split("|") if a.script else list(DEFAULT_SCRIPT)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.script = script
    srv.verbose = a.verbose

    print("=" * 66)
    print("  假 ASR 服务端（OpenAI 兼容）")
    print("=" * 66)
    print("  地址：http://%s:%d/v1/audio/transcriptions" % (a.host, a.port))
    print("  回话：%s" % script)
    print("  真实接口调用方式：")
    print("     python moss_brain.py --asr api --asr-api-base http://%s:%d/v1 \\"
          % (a.host, a.port))
    print("         --asr-api-key fake --audio-source 录音.wav")
    print("=" * 66)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[OK] 退出。共服务 %d 次请求。" % STATE["n"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
