# moss_runtime — MOSS-TTS-Nano CPU 运行时 (精简版)
# 来源: OpenMOSS-Team/MOSS-TTS-Nano (Apache-2.0) ONNX CPU 版
# 精简: 删除语音克隆(torchaudio)/wetext文本规范化/自动下载, 接入Android同款TextCleaner
from .onnx_tts_runtime import OnnxTtsRuntime
from .text_cleaner import clean, split_sentences, compress_gaps

__all__ = ["OnnxTtsRuntime", "clean", "split_sentences", "compress_gaps"]
