# -*- coding: utf-8 -*-
"""
text_cleaner.py — Kotlin TextCleaner 的 Python 移植 (与 Android 端行为对齐)

包含 2026-10-10 全部文本优化:
  - 语种分开: isCjkOrKana 智能空格 (CJK/假名间删空格, 英文/韩文/数字间保留)
  - 标点过滤: 只保留 。 ， ？ ； 、 5个安全标点, !/！→。(MOSS乱音修复)
  - emoji/符号: 逐字符转逗号或删除, markdown/URL/HTML清理
  - Fix D: 首尾孤悬"，"修剪 + 无实质内容(纯标点/emoji)返回空串防 runaway 怪音
  - 小数点保护: 3.14→3点14 (数字.数字), 数字.→数字，
  - compressGaps: 停顿压缩语速(只剪静音段, 发音本体零改动, 不升调)
"""
from __future__ import annotations
import re
import numpy as np

SAFE_PUNCT = {"。", "，", "？", "；", "、"}


def is_cjk_or_kana(ch: str) -> bool:
    code = ord(ch)
    return (0x4E00 <= code <= 0x9FFF   # CJK统一汉字
            or 0x3400 <= code <= 0x4DBF  # CJK扩展A
            or 0x3040 <= code <= 0x309F  # 平假名
            or 0x30A0 <= code <= 0x30FF  # 片假名
            or 0x3000 <= code <= 0x303F  # CJK标点
            or 0xFF00 <= code <= 0xFFEF)  # 全角


def clean(text: str) -> str:
    s = str(text or "")

    # === 1. 基础清理 ===
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    s = re.sub(r"```[\s\S]*?```", "", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"[\u2600-\u27BF\u2B50\u2B55\u23CF\u23E9-\u23F3\u231A\u231B"
               r"\u25AA-\u25C0\u25FB-\u25FE\u2648-\u2653\u267F\u2934\u2935\u2B05-\u2B07]+", "", s)

    # === 2. 引号全部删除 ===
    s = re.sub(r"[\"'\u201C\u201D\u2018\u2019]", "", s)

    # === 2.5. v+数字 → 去掉v前缀 (MOSS读v+数字会跳过) ===
    s = re.sub(r"([vV])(\d)", r"\2", s)

    # === 3. 数字+点号 (必须在半角.→。之前!) ===
    s = re.sub(r"(\d+)\.(\d)", r"\1点\2", s)
    s = re.sub(r"(\d+)\.", r"\1，", s)

    # === 4. 括号全部删除 ===
    s = re.sub(r"[()\[\]{}（）【】「」『』《》]", "", s)

    # === 5. 长符号 → 句号 ===
    s = re.sub(r"[—–―]{2,}", "。", s)
    s = re.sub(r"-{3,}", "。", s)
    s = re.sub(r"…{1,}|\.{3,}", "。", s)

    # === 6. 箭头 → 逗号 ===
    s = re.sub(r"[→←⇒⇐⟶⟼⟷⟹⟺↦↪↩]|<-+|-+>|<=+|=+>", "，", s)

    # === 7. 半角→中文标点 (!→。 因为！会导致MOSS乱音) ===
    s = s.replace(",", "，")
    s = s.replace("!", "。")
    s = s.replace("?", "？")
    s = s.replace(";", "；")
    s = s.replace(":", "：")
    s = s.replace(".", "。")

    # === 7.4. 数学符号(上下文感知: 数字间的才转) ===
    s = re.sub(r"(\d)\s*-\s*(\d)", r"\1减\2", s)
    s = re.sub(r"(\d)\s*\*\s*(\d)", r"\1乘\2", s)
    s = s.replace("÷", "除").replace("×", "乘")

    # === 7.5. 常见符号→文字 (MOSS读不出来就乱音) ===
    s = s.replace("=", "等于").replace("+", "加").replace("/", "或")
    s = s.replace("%", "百分之")
    for ch in "$€£¥#@~\\^_`":
        s = s.replace(ch, "")
    s = s.replace("℃", "度").replace("℉", "度").replace("°", "度")
    s = s.replace("&", "和").replace("|", "，")

    # === 8. 其他符号→逗号 (只保留SAFE标点+字母数字+空白) ===
    out = []
    for ch in s:
        if ch in SAFE_PUNCT:
            out.append(ch)
        elif ch == "！":
            out.append("。")
        elif ch.isalnum():
            out.append(ch)
        elif ch.isspace():
            out.append(ch)
        else:
            out.append("，")
    s = "".join(out)

    # === 9. 连续标点收敛 ===
    s = re.sub(r"。{2,}", "。", s)
    s = re.sub(r"，{2,}", "，", s)
    s = re.sub(r"？{2,}", "？", s)

    # === 10. 换行→句号 ===
    s = s.replace("\n", "。")

    # === 11. 智能空格处理 (CJK/假名间删空格, 英文/韩文/数字间保留) ===
    out = []
    for i, ch in enumerate(s):
        if ch.isspace():
            prev = s[i - 1] if i > 0 else None
            nxt = s[i + 1] if i + 1 < len(s) else None
            prev_cjk = prev is not None and is_cjk_or_kana(prev)
            next_cjk = nxt is not None and is_cjk_or_kana(nxt)
            if not prev_cjk and not next_cjk:
                out.append(ch)  # 英文/韩文/数字之间的空格保留
        else:
            out.append(ch)
    s = "".join(out)

    # === 11.5. Fix D: 首尾孤悬"，"修剪 ===
    s = s.replace("，。", "。").replace("，？", "？").replace("，；", "；")
    s = s.strip().strip("，、；").strip()

    # === 12. 句末补标点 ===
    s = s.strip()
    if s and s[-1] not in "。？；，、":
        s += "。"

    # === 13. Fix D: 实质内容检查 (纯标点/emoji句丢弃, 防runaway怪音) ===
    if not any(ch.isalnum() for ch in s):
        return ""
    return s


def split_sentences(text: str, max_len: int = 20) -> list[str]:
    """拆分TTS短句 (与Kotlin TextCleaner.splitSentences一致)"""
    parts = re.split(r"([。？；\n])", text)
    raw = []
    i = 0
    while i < len(parts):
        seg = parts[i]
        punct = ""
        if i + 1 < len(parts) and re.fullmatch(r"[。？；\n]", parts[i + 1] or ""):
            i += 1
            punct = parts[i]
        i += 1
        combined = (seg + punct).strip()
        if combined:
            raw.append(combined)
    result = []
    for seg in raw:
        if len(seg) <= max_len:
            if len(seg) > 1:
                result.append(seg)
        else:
            buf = ""
            for p in re.split(r"[，,\s]", seg):
                if buf and len(buf) + len(p) > max_len:
                    result.append(buf.strip())
                    buf = ""
                buf += p
            if buf and len(buf) > 1:
                result.append(buf.strip())
    return result


def compress_gaps(pcm: np.ndarray, speed: float, sample_rate: int) -> np.ndarray:
    """
    Fix F: 停顿压缩(剪空白) — 只缩短静音段, 发音本体零改动, 音色/音高不变。
    keep = clamp(min(段长,800ms)/speed^5, 15ms, 段长)
      speed=1.0 → 正常停顿保留, >800ms怪静音截到800ms
      speed=1.3 → 停顿压约1/3.7 (中文-10%, 日韩英-20%)
      speed<1.0 → 停顿不变 (语速只影响紧凑度, 不拉长)
    pcm: int16 一维数组
    """
    if pcm.size == 0:
        return pcm
    threshold = 300
    gap_cap_ms = 800.0
    gap_floor_ms = 15.0
    keep_ratio = 1.0 / (max(0.5, min(3.0, float(speed))) ** 5)  # Fix②: 上限2.0→3.0(2026-10-10)
    out = np.empty_like(pcm)
    w = 0
    i = 0
    n = pcm.size
    abs_pcm = np.abs(pcm.astype(np.int32))
    while i < n:
        if abs_pcm[i] < threshold:
            j = i
            while j < n and abs_pcm[j] < threshold:
                j += 1
            gap_len = j - i
            gap_ms = gap_len * 1000.0 / sample_rate
            keep_ms = max(min(min(gap_ms, gap_cap_ms) * keep_ratio, gap_ms), gap_floor_ms)
            keep = int(round(keep_ms * sample_rate / 1000.0))
            keep = max(1, min(keep, gap_len))
            out[w:w + keep] = pcm[i:i + keep]
            w += keep
            i = j
        else:
            out[w] = pcm[i]
            w += 1
            i += 1
    return pcm if w == n else out[:w].copy()
