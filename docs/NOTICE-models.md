# Third-Party Model Notices

## MOSS-Audio-Tokenizer-Nano (TTS 音频解码器)

- 文件: `moss_audio_tokenizer_decode_full_int8.onnx` (随 app 部署)
- 来源: OpenMOSS-Team / MOSS-Audio-Tokenizer-Nano-ONNX (经 REALBITS/MOSS-TTS-Nano-100M-ONNX-int8 公开发布渠道)
- 许可: Apache License 2.0 (全文见 LICENSE-MOSS-Audio-Tokenizer.txt)
- 改编说明: 本文件由上述 Apache-2.0 授权的 FP32 ONNX 解码器
  (moss_audio_tokenizer_decode_full.onnx) 经 ONNX Runtime 动态 int8
  量化 (MatMul/Attention/Gemm 权重量化) 派生, 网络结构与输入输出不变,
  仅权重精度压缩。属于 Apache-2.0 授权作品的衍生作品, 依原许可分发。

## MOSS-TTS-Nano-100M (TTS 声学模型)

- 文件: `moss_tts_*.onnx` + `global_shared.data` / `local_shared.data`
- 来源: linlinkj/MOSS-TTS-Nano-100M-INT8 (Apache-2.0) 的 MatMul 动态 int8
  权重 + wte/lm_head u8 动态量化 (与 REALBITS 公开 int8 版逐字节同源)
- 许可: Apache License 2.0
