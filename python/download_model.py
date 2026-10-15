#!/usr/bin/env python3
"""
download_model.py — 模型下载/检测 (16文件平铺到 model/MOSS-TTS-Nano-100M-ONNX/)
  逐文件从魔搭下载 + md5 校验, 缺哪个补哪个(可续传补齐)
  python download_model.py              # 无模型/缺文件则补齐
  python download_model.py --check-only # 只检测, 有模型 exit 0, 无模型 exit 1
"""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model" / "MOSS-TTS-Nano-100M-ONNX"

MODEL_BASE_URL = "https://www.modelscope.cn/models/ArchitectLi/MOSS-TTS-Nano-int8/resolve/master"

# 文件名 -> md5 (与魔搭 md5sums.txt 一致)
REQUIRED_FILES = {
    "browser_poc_manifest.json": "c19a8289fee80697fa871e924ff45e52",
    "codec_browser_onnx_meta.json": "3261f9c906dd72e86450dc0b07c8ca47",
    "config.json": "620f6e74bbd0e94aaaba9c062d9fcb29",
    "global_shared.data": "6c7114481aa1b4a33e1fed3884162bba",
    "local_shared.data": "649ba6194564242239786e937f21ddc6",
    "moss_audio_tokenizer_decode_full_int8.onnx": "adac133cd7bde48ebf06835a3dd3fa77",
    "moss_tts_decode_step.onnx": "c1fa623f337471759500e44260413058",
    "moss_tts_local_cached_step.onnx": "990be57ee616dc9dd10f13b0af1b2107",
    "moss_tts_local_decoder.onnx": "0621f01b4724a67a88a4e068d1a4dbb5",
    "moss_tts_local_fixed_sampled_frame.onnx": "31eaa1c6a635282f7e9d73a84ec44b55",
    "moss_tts_local_greedy_frame.onnx": "4d479d5ce528d1b2656b3f7e389a1672",
    "moss_tts_local_sampled_frame.onnx": "5905aaf3248907c12336015afde4a12c",
    "moss_tts_prefill.onnx": "e6a6e83f7da1a787b4abeab7e3f7f1f4",
    "quantization_config.json": "75a27ac4f10853d5ec231627a3b67393",
    "tokenizer.model": "13276bc226e31b5cce9fef78d6e4e270",
    "tts_browser_onnx_meta.json": "d494d9ca9921adc8e6001ae5277c864d",
}


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_ready() -> bool:
    return all((MODEL_DIR / n).is_file() and (MODEL_DIR / n).stat().st_size > 0
               for n in REQUIRED_FILES)


def download() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    names = list(REQUIRED_FILES)
    for i, name in enumerate(names, 1):
        want = REQUIRED_FILES[name]
        target = MODEL_DIR / name
        if target.is_file() and md5_of(target) == want:
            continue
        url = f"{MODEL_BASE_URL}/{name}"
        print(f"[download] ({i}/{len(names)}) {name}", flush=True)
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                # 带超时的流式下载(挂死连接会在 120s 内报错进重试)
                with urllib.request.urlopen(url, timeout=60) as resp, open(target, "wb") as f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                if md5_of(target) == want:
                    last_err = None
                    break
                last_err = RuntimeError(f"md5 不一致: {name}")
            except Exception as e:  # 网络抖动/超时重试
                last_err = e
        if last_err is not None:
            target.unlink(missing_ok=True)
            raise SystemExit(f"[download] 失败: {name}: {last_err}")
    if not is_ready():
        missing = [n for n in REQUIRED_FILES if not (MODEL_DIR / n).exists()]
        raise SystemExit(f"[download] 模型文件不全, 缺: {missing}")
    print(f"[download] 完成, {len(REQUIRED_FILES)} 个文件 (md5 全部校验通过)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    if is_ready():
        print(f"[check] 模型已就绪: {MODEL_DIR}")
        return 0
    if args.check_only:
        print(f"[check] 模型缺失: {MODEL_DIR}")
        return 1
    download()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
