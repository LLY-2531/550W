#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOSS 长期记忆  ·  流浪地球 550W 复刻项目 · 阶段 13
============================================================
阶段 12 结束时 MOSS 已经能听懂你说话了，但它有**失忆症**：
对话历史只存在一个内存里的 deque 里，程序一关就全没了；
而且它也只记得最近 3 轮 —— 你上周告诉过它你叫什么，它不知道。

这一步给它两样东西：

    1. 对话历史落盘  →  关掉再开，它还接着上一回聊
    2. 长期事实表    →  "我叫张三""我住在深圳""别叫我小爱" 这类
                         跨会话都该记住的事，单独存成 key-value

为什么不用数据库
------------------------------------------------------------
要存的数据量是"几十轮对话 + 十几个事实"，总量几十 KB。
上 SQLite 属于杀鸡用牛刀，而且多一个文件格式要维护。
一个 JSON 文件、人类可读、出问题直接打开看 —— 这才符合本项目的调性。

一个容易被忽略的问题：写盘要原子
------------------------------------------------------------
如果直接 open(path,'w') 然后写，写到一半断电/被杀，文件就半截了，
下次启动直接解析失败、**记忆全丢**。
所以这里走"写临时文件 → os.replace 覆盖"：
在同一分区内 rename 是原子操作，要么是旧内容，要么是完整新内容，没有中间态。

线程安全
------------------------------------------------------------
主循环（20Hz）、ASR 线程、语言层线程都可能碰记忆，
所以内部一把 RLock 保护全部状态，对外只暴露方法、不暴露字段。

自检：
    python memory.py            # 在临时目录里跑一遍增删改查，不碰真实记忆
"""

import json
import os
import threading
import time

# 默认存到工程内的 data/ 目录：跟着项目走，拷走工程 = 拷走记忆
DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "moss_memory.json")

# 对话最多留多少轮（一轮 = 你说一句 + MOSS 答一句）
MAX_DIALOG = 60
# 长期事实最多存多少条（超了淘汰最久没被用到的）
MAX_FACTS = 200
# 两次落盘之间至少间隔多少秒（避免每轮对话都写盘）
SAVE_INTERVAL = 2.0

# 这些 key 太笼统，存了会污染提示词，直接拒掉
BAD_KEYS = {"", "-", "无", "none", "null", "null_value", "unknown"}


def _now():
    return time.time()


def _clip(s, n):
    """把任意值洗成一个不超过 n 字的单行字符串。"""
    if s is None:
        return ""
    s = str(s).replace("\r", " ").replace("\n", " ").strip()
    s = " ".join(s.split())
    return s[:n]


class MemoryStore:
    """跨会话的长期记忆。

    两个独立的部分：
      - `dialog`：最近 N 轮对话，滚动窗口，喂给模型当上下文
      - `facts` ：长期事实，key-value，永远保留（除非被淘汰/主动忘掉）

    为什么要分开：对话是"流水"（多了要截断），事实是"沉淀"（多了才是价值）。
    混在一起存的话，一截断就把事实也截没了。
    """

    VERSION = 1

    def __init__(self, path=None, max_dialog=MAX_DIALOG, max_facts=MAX_FACTS,
                 save_interval=SAVE_INTERVAL, autosave=True, quiet=False):
        self.path = path or DEFAULT_PATH
        self.max_dialog = max(0, int(max_dialog))
        self.max_facts = max(1, int(max_facts))
        self.save_interval = max(0.0, float(save_interval))
        self.autosave = autosave
        self.quiet = quiet

        self.lock = threading.RLock()
        self.dialog = []          # [[user, moss], ...]  moss 可能先是空串
        self.facts = {}           # {key: {"value": str, "at": ts, "hits": int}}
        self.created = _now()
        self.updated = self.created
        self.n_sessions = 0
        self.last_save = 0.0
        self.dirty = False
        self.stats = {"turns": 0, "remembers": 0, "recalls": 0,
                      "forgets": 0, "evicted": 0, "load_errors": 0}

        self.load()
        self.n_sessions += 1          # 本次启动也算一次会话
        self.dirty = True

    # ================================================================
    #  落盘 / 读取
    # ================================================================
    def load(self):
        """从磁盘读回来。文件不存在 = 全新记忆；损坏 = 备份后从空开始。"""
        if not os.path.exists(self.path):
            return False
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            # 不能直接崩：宁可丢记忆也要让 MOSS 能启动。
            # 把坏文件改名留档，方便事后看是哪儿写坏的。
            self.stats["load_errors"] += 1
            bad = self.path + ".bad"
            try:
                os.replace(self.path, bad)
            except OSError:
                pass
            if not self.quiet:
                print(f"[记忆] 读不了 {self.path}（{e}），已备份为 {bad}，从空记忆开始。")
            return False

        dlg = data.get("dialog") or []
        clean = []
        for item in dlg:
            if isinstance(item, (list, tuple)) and item:
                u = _clip(item[0], 200)
                m = _clip(item[1], 200) if len(item) > 1 else ""
                if u or m:
                    clean.append([u, m])
        self.dialog = clean[-self.max_dialog:] if self.max_dialog else []

        facts = data.get("facts") or {}
        if isinstance(facts, dict):
            for k, v in facts.items():
                k = _clip(k, 40)
                if not k:
                    continue
                if isinstance(v, dict):
                    val = _clip(v.get("value"), 200)
                    at = float(v.get("at") or 0.0)
                    hits = int(v.get("hits") or 0)
                else:
                    val, at, hits = _clip(v, 200), 0.0, 0
                if val:
                    self.facts[k] = {"value": val, "at": at, "hits": hits}

        self.created = float(data.get("created") or self.created)
        self.n_sessions = int(data.get("n_sessions") or 0)
        return True

    def save(self, force=False):
        """原子写盘。force=False 时会按 save_interval 节流。"""
        with self.lock:
            if not self.dirty and not force:
                return False
            now = _now()
            if not force and (now - self.last_save) < self.save_interval:
                return False                    # 攒一攒再写，别每轮都碰磁盘
            payload = {
                "version": self.VERSION,
                "created": self.created,
                "updated": now,
                "n_sessions": self.n_sessions,
                "dialog": self.dialog[-self.max_dialog:] if self.max_dialog else [],
                "facts": self.facts,
            }
            self.updated = now
            self.last_save = now
            self.dirty = False

        d = os.path.dirname(os.path.abspath(self.path))
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=1)
                f.flush()
                os.fsync(f.fileno())        # 真正落到盘上再 rename
            os.replace(tmp, self.path)      # ← 原子操作，不会出现半截文件
            return True
        except Exception as e:
            try:
                os.remove(tmp)
            except OSError:
                pass
            if not self.quiet:
                print(f"[记忆] 存盘失败：{e}")
            return False

    def close(self):
        """退出前一定要调，把没落盘的那部分强制写下去。"""
        return self.save(force=True)

    def _touch(self):
        with self.lock:
            self.dirty = True
        if self.autosave:
            self.save()

    # ================================================================
    #  对话历史
    # ================================================================
    def add_turn(self, user, moss=""):
        """记一轮对话。MOSS 的回答可能要几秒后才知道，所以先留空。"""
        user = _clip(user, 200)
        moss = _clip(moss, 200)
        if not user and not moss:
            return
        with self.lock:
            self.dialog.append([user, moss])
            if self.max_dialog and len(self.dialog) > self.max_dialog:
                # 超出窗口就从头砍掉最老的
                self.dialog = self.dialog[-self.max_dialog:]
            self.stats["turns"] += 1
        self._touch()

    def fill_last_answer(self, say):
        """把刚想出来的回答补进最后那一轮（阶段 12 的老逻辑，搬到这里）。"""
        say = _clip(say, 200)
        if not say:
            return False
        with self.lock:
            if self.dialog and not self.dialog[-1][1]:
                self.dialog[-1][1] = say
                self._touch()
                return True
        return False

    def recent(self, n=None):
        """最近 n 轮（返回副本，别让外面改到内部状态）。"""
        with self.lock:
            d = self.dialog[-(n if n is not None else self.max_dialog):] \
                if (n if n is not None else self.max_dialog) else []
            return [[u, m] for u, m in d]

    def dialog_text(self, n=None):
        """压成纯文本给提示词用。

        为什么不用标准的多轮 messages：我们每轮都要求模型返回 JSON，
        把历史回答（JSON）放进 assistant 角色会让它开始"续写 JSON"。
         """
        rows = self.recent(n)
        if not rows:
            return ""
        lines = []
        for u, m in rows:
            if u:
                lines.append(f"  用户：{u}")
            if m:
                lines.append(f"  MOSS：{m}")
        return "\n".join(lines)

    # ================================================================
    #  长期事实
    # ================================================================
    def remember(self, key, value):
        """记一条事实。同一个 key 再记就是覆盖（并累加 hits）。"""
        key = _clip(key, 40)
        value = _clip(value, 200)
        if key.lower() in BAD_KEYS or not value:
            return False
        with self.lock:
            old = self.facts.get(key)
            hits = (old or {}).get("hits", 0)
            self.facts[key] = {"value": value, "at": _now(), "hits": hits}
            self.stats["remembers"] += 1
            self._evict_locked()
        self._touch()
        return True

    def _evict_locked(self):
        """超过上限时，淘汰最"没用"的一条：先看 hits，再看新旧。"""
        while len(self.facts) > self.max_facts:
            victim = min(self.facts.items(),
                         key=lambda kv: (kv[1].get("hits", 0), kv[1].get("at", 0)))
            self.facts.pop(victim[0], None)
            self.stats["evicted"] += 1

    def recall(self, query, limit=5):
        """按字面命中找事实。没传 query 就返回全部（按新旧排）。"""
        with self.lock:
            items = list(self.facts.items())
        q = _clip(query, 100)
        hits = []
        for k, v in items:
            if q and (q in k or k in q or q in v["value"] or v["value"] in q):
                hits.append((k, v))
        if not hits:
            hits = sorted(items, key=lambda kv: kv[1].get("at", 0), reverse=True)
        with self.lock:
            self.stats["recalls"] += 1
        # 命中一次就记一下，让淘汰算法知道哪条有用
        with self.lock:
            for k, _ in hits[:limit]:
                if k in self.facts:
                    self.facts[k]["hits"] = self.facts[k].get("hits", 0) + 1
        return [(k, v["value"]) for k, v in hits[:limit]]

    def forget(self, key):
        key = _clip(key, 40)
        with self.lock:
            got = self.facts.pop(key, None)
            if got:
                self.stats["forgets"] += 1
        if got:
            self._touch()
        return bool(got)

    def all_facts(self):
        with self.lock:
            return {k: dict(v) for k, v in self.facts.items()}

    def facts_text(self, limit=12):
        """把长期事实压成一段文字塞进提示词。最新的排前面。"""
        with self.lock:
            items = sorted(self.facts.items(),
                           key=lambda kv: kv[1].get("at", 0), reverse=True)[:limit]
        if not items:
            return ""
        return "\n".join(f"  {k}：{v['value']}" for k, v in items)

    # ================================================================
    #  杂项
    # ================================================================
    def summary(self):
        with self.lock:
            return {
                "path": self.path,
                "turns": len(self.dialog),
                "facts": len(self.facts),
                "sessions": self.n_sessions,
                "created": self.created,
                "updated": self.updated,
            }

    def describe(self):
        with self.lock:
            age = (_now() - self.created) / 86400.0
        s = self.summary()
        return (f"记忆文件 {s['path']}｜本次是第 {s['sessions']} 次启动｜"
                f"对话 {s['turns']} 轮｜长期事实 {s['facts']} 条｜"
                f"已存在 {age:.1f} 天")


# ================================================================
#  自检：python memory.py
# ================================================================
def _selftest():
    import tempfile

    d = tempfile.mkdtemp(prefix="moss_mem_")
    path = os.path.join(d, "mem.json")

    print("=" * 66)
    print("  MOSS 长期记忆自检")
    print("=" * 66)

    print("\n[1/5] 全新记忆")
    m = MemoryStore(path, quiet=True)
    print("      " + m.describe())
    assert m.recent() == [] and m.all_facts() == {}, "新记忆应该是空的"

    print("\n[2/5] 记对话 + 记事实")
    m.add_turn("你叫什么名字")
    m.fill_last_answer("我是550W。")
    m.add_turn("现在几点了")
    m.fill_last_answer("下午三点。")
    m.remember("用户的名字", "张三")
    m.remember("用户所在城市", "深圳")
    m.remember("用户的喜好", "喜欢安静")
    m.save(force=True)
    print("      对话 %d 轮，事实 %d 条" % (len(m.recent()), len(m.all_facts())))
    assert len(m.recent()) == 2 and len(m.all_facts()) == 3
    assert m.recent()[-1] == ["现在几点了", "下午三点。"]

    print("\n[3/5] 跨会话：重新打开同一个文件")
    m2 = MemoryStore(path, quiet=True)
    print("      " + m2.describe())
    assert len(m2.recent()) == 2, "对话历史应该还在"
    assert m2.all_facts().get("用户的名字", {}).get("value") == "张三", "事实应该还在"
    assert m2.n_sessions == 2, "启动次数应该累加到 2"
    print("      ✓ 对话和事实都活过了重启（第 %d 次启动）" % m2.n_sessions)

    print("\n[4/5] 覆盖 / 检索 / 遗忘")
    m2.remember("用户的名字", "李四")           # 同一个 key 覆盖
    assert m2.all_facts()["用户的名字"]["value"] == "李四"
    got = m2.recall("城市")
    print("      检索「城市」→ %s" % got)
    assert ("用户所在城市", "深圳") in got
    assert m2.forget("用户的喜好") is True
    assert m2.forget("不存在的key") is False
    m2.save(force=True)

    print("\n[5/5] 坏文件不能让它崩")
    with open(path, "w", encoding="utf-8") as f:
        f.write("{这不是合法 json")
    m3 = MemoryStore(path, quiet=True)
    assert m3.recent() == [] and m3.all_facts() == {}
    assert os.path.exists(path + ".bad"), "坏文件应该被备份成 .bad"
    print("      ✓ 坏文件已备份为 .bad，程序照常启动（丢记忆但不崩）")
    print("      stats = %s" % m3.stats)

    print("\n[OK] 长期记忆自检全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
