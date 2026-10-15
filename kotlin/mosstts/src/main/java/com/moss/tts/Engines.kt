package com.moss.tts

/**
 * TTS语音合成引擎接口
 * 实现: MossTtsEngine
 * (原AsrEngine死接口已删 2026-10-10: 本模块纯TTS, ASR在宿主项目用sherpa-onnx)
 */
interface TtsEngine {
    suspend fun load(modelPath: String, voiceId: Int = 1, speed: Float = 1.0f): Boolean
    fun unload()
    fun isLoaded(): Boolean
    /** 切换音色/语速 (不重新加载模型) */
    fun setVoice(voiceId: Int, speed: Float = 1.0f) {}
    /** WSOLA 整句压伸 1.0~1.5 (音高不变) — 默认无操作 (Fix③ 2026-10-10) */
    fun setStretch(ratio: Float) {}
    /** 文本转PCM音频，返回16bit PCM数据 */
    suspend fun generate(text: String): TtsResult?
    /** 引擎级预热(有界, 禁止用generate跑预热文本) — 默认无操作 (Fix B 2026-10-10) */
    fun prewarm() {}
}

data class TtsResult(
    val pcmData: ShortArray,  // 16bit PCM
    val sampleRate: Int,      // 采样率
)
