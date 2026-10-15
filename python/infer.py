#!/usr/bin/env python3
"""
infer.py — MOSS-TTS-Nano CPU 推理入口
  命令行:  python infer.py --text "今天天气真好" --voice 4 --out output/a.wav
  模块组装: from infer import MossTts; tts = MossTts(); tts.synthesize("你好")

与 Android 端行为对齐: TextCleaner 文本清理 / 固定种子确定性(1234) /
帧数封顶防 runaway(≤5 token→40帧, ≤15→80, ≤40→200, 其余400) / --speed 停顿压缩 /
--stretch WSOLA整句压伸(音高不变)
"""
from __future__ import annotations

import argparse
import sys
import time
import wave  # noqa: E402
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moss_runtime.onnx_tts_runtime import OnnxTtsRuntime  # noqa: E402
from moss_runtime.text_cleaner import clean, compress_gaps  # noqa: E402
from moss_runtime.audio_ops import wsola_stretch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "model" / "MOSS-TTS-Nano-100M-ONNX"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
SAMPLE_MODES = ("full", "fixed", "greedy")
DEFAULT_SEED = 1234  # Fix E: 与Android端一致, 默认确定性输出


def estimate_max_frames(token_count: int) -> int:
    """帧数封顶 (与Android MossTtsEngine.estimateMaxFrames一致, 防退化输入runaway)"""
    if token_count <= 5:
        return 40
    if token_count <= 15:
        return 80
    if token_count <= 40:
        return 200
    return 400


class MossTts:
    """模块化 API — 可直接组装到其他 Python 项目"""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        voice: int | str = 4,
        speed: float = 1.0,
        stretch: float = 1.0,
        sample_mode: str = "full",
        seed: int | None = DEFAULT_SEED,
        threads: int = 4,
        output_dir: str | Path | None = None,
    ) -> None:
        self.rt = OnnxTtsRuntime(
            model_dir=str(model_dir or DEFAULT_MODEL_DIR),
            thread_count=int(threads),
            sample_mode=str(sample_mode),
            output_dir=str(output_dir or DEFAULT_OUTPUT_DIR),
        )
        self.voice = self._resolve_voice(voice)
        self.speed = float(speed)
        self.stretch = float(stretch)  # WSOLA 压伸倍数 1.0~1.5
        self.seed = seed
        self.sample_mode = str(sample_mode)

    def _resolve_voice(self, voice: int | str) -> str:
        names = [row["voice"] for row in self.rt.list_builtin_voices()]
        if isinstance(voice, int) or (isinstance(voice, str) and voice.isdigit()):
            idx = int(voice)
            if not 0 <= idx < len(names):
                raise ValueError(f"voiceId 超出范围 0~{len(names) - 1}: {idx}")
            return names[idx]
        if str(voice) not in names:
            raise ValueError(f"未知音色: {voice}, 可用: {names}")
        return str(voice)

    def list_voices(self) -> list[dict]:
        return [{"id": i, "name": row["voice"]}
                for i, row in enumerate(self.rt.list_builtin_voices())]

    def synthesize(self, text: str, out_path: str | Path | None = None) -> Path:
        """文本 → wav。返回输出路径。"""
        cleaned = clean(text)
        if not cleaned:
            raise ValueError("文本清理后为空(纯标点/emoji会被丢弃)")
        out = Path(out_path) if out_path else DEFAULT_OUTPUT_DIR / "output.wav"
        out.parent.mkdir(parents=True, exist_ok=True)

        token_count = self.rt.count_text_tokens(cleaned)
        t0 = time.perf_counter()
        result = self.rt.synthesize(
            text=cleaned,
            voice=self.voice,
            output_audio_path=str(out),
            sample_mode=self.sample_mode,
            seed=self.seed,
            max_new_frames=estimate_max_frames(token_count),
        )
        gen_ms = int((time.perf_counter() - t0) * 1000)

        # Fix F: 停顿压缩 — 只剪静音段, 发音本体零改动 (speed=1.0也截>800ms怪静音)
        waveform = np.asarray(result["waveform"], dtype=np.float32).reshape(-1)
        pcm16 = np.round(np.clip(waveform, -1.0, 1.0) * 32767.0).astype(np.int16)
        pcm16 = compress_gaps(pcm16, self.speed, int(result["sample_rate"]))
        if self.stretch > 1.01:  # Fix③: WSOLA 压伸, 音高不变
            pcm16 = wsola_stretch(pcm16, self.stretch)
        with wave.open(str(out), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(result["sample_rate"]))
            wf.writeframes(pcm16.tobytes())
        audio_ms = pcm16.size * 1000 // int(result["sample_rate"])
        print(f"[infer] voice={self.voice} tokens={token_count} "
              f"gen={gen_ms}ms audio={audio_ms}ms -> {out}")
        return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MOSS-TTS-Nano CPU 推理")
    p.add_argument("--text", help="要合成的文本")
    p.add_argument("--file", help="从文本文件读取")
    p.add_argument("--voice", default="4", help="音色: id(0-17)或名字, 默认4悦文")
    p.add_argument("--speed", type=float, default=1.0, help="停顿压缩强度 1.0~3.0 (默认1.0)")
    p.add_argument("--stretch", type=float, default=1.0, help="WSOLA整句压伸 1.0~1.5, 音高不变 (默认1.0关闭)")
    p.add_argument("--sample-mode", default="full", choices=SAMPLE_MODES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="采样种子, -1=随机")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--out", default=None, help="输出wav路径, 默认 output/output.wav")
    p.add_argument("--list-voices", action="store_true")
    args = p.parse_args(argv)

    tts = MossTts(
        model_dir=args.model_dir, voice=args.voice, speed=args.speed,
        stretch=args.stretch,
        sample_mode=args.sample_mode,
        seed=None if args.seed < 0 else args.seed,
        threads=args.threads,
    )
    if args.list_voices:
        for row in tts.list_voices():
            print(f"{row['id']:2d}  {row['name']}")
        return 0
    text = args.text or (Path(args.file).read_text(encoding="utf-8") if args.file else "")
    if not text.strip():
        p.error("需要 --text 或 --file")
    tts.synthesize(text, out_path=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
