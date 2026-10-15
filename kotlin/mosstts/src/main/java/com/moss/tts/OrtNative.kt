package com.moss.tts

/**
 * JNI桥接：通过dlopen调sherpa-onnx已有的libonnxruntime.so的C API
 * 避免ORT Java API的版本符号冲突
 */
object OrtNative {
    init {
        // libonnxruntime.so已由sherpa-onnx加载，无需额外loadLibrary
        System.loadLibrary("ort_jni")
    }

    /**
     * 创建ORT session (纯CPU — NNAPI已删除: 8Gen3负优化, 真NPU走QNN HTP)
     * @param forceCpu 已废弃, 仅为接口兼容保留, 不再使用
     * @return session handle, 0表示失败
     */
    external fun nativeCreateSession(modelPath: String, numThreads: Int, forceCpu: Boolean): Long

    /**
     * 运行推理
     * @param handle session handle
     * @param inputNames 输入名数组, e.g. ["face", "audio"]
     * @param inputShapes 输入形状数组, e.g. [[1,6,96,96], [1,1,80,16]]
     * @param inputDatas 输入数据数组
     * @param outputNames 输出名数组, e.g. ["output"]
     * @return 输出数据数组, null表示失败
     */
    external fun nativeRun(
        handle: Long,
        inputNames: Array<String>,
        inputShapes: Array<IntArray>,
        inputDatas: Array<FloatArray>,
        outputNames: Array<String>
    ): Array<FloatArray>?

    /** 释放session */
    external fun nativeRelease(handle: Long)

    /** 获取session的输入名列表 */
    external fun nativeGetInputNames(handle: Long): Array<String>?

    /** 支持混合tensor类型的Run (float32/int32/int64) */
    external fun nativeRunTyped(
        handle: Long,
        inputNames: Array<String>,
        inputShapes: Array<IntArray>,
        inputDatas: Array<ByteArray>,  // 统一用ByteArray传，内部按type解释
        inputTypes: IntArray,           // 0=float32, 1=int32, 2=int64
        outputNames: Array<String>
    ): Array<ByteArray>?

    /** Benchmark: 加载模型运行N次，返回每次耗时(ms) */
    external fun nativeBenchmark(modelPath: String, numThreads: Int, numRuns: Int): IntArray?

    // --- MOSS TTS: C++层管理past cache ---

    /** 初始化MOSS上下文（运行prefill，存储past cache在C++层） */
    external fun nativeMossInit(
        prefillHandle: Long, decodeHandle: Long,
        localFixedHandle: Long, codecDecodeHandle: Long,
        inputIds: ByteArray, inputIdsShape: IntArray,
        attnMask: ByteArray, attnMaskShape: IntArray,
        nVq: Int, rowWidth: Int, padTokenId: Int, assistantSlotTokenId: Int
    ): Long

    /** 生成一帧+decode step（past cache在C++内管理）。返回 [shouldContinue, token0..token15] */
    external fun nativeMossNextFrame(
        contextHandle: Long,
        repMask: ByteArray, repMaskShape: IntArray,
        assistantRandomU: ByteArray,
        audioRandomU: ByteArray, audioRandomUShape: IntArray
    ): IntArray?

    /** Codec decode: tokens -> float32 audio (前8字节=audio_length, 后面=float32 PCM) */
    external fun nativeMossCodecDecode(
        codecHandle: Long,
        codes: ByteArray, codesShape: IntArray,
        codesLengths: ByteArray, codesLengthsShape: IntArray
    ): ByteArray?

    /** 释放MOSS上下文 */
    external fun nativeMossCleanup(contextHandle: Long)

    /** 一次JNI调用生成全部音频。返回: 前4字节sampleCount(int32), 后面int16 PCM */
    external fun nativeMossGenerateAudio(contextHandle: Long, maxFrames: Int, audioCodebookSize: Int): ByteArray?

    // --- Kotlin友好封装 ---

    class Session(val handle: Long) {
        val isValid get() = handle != 0L

        fun run(
            inputs: Map<String, Pair<IntArray, FloatArray>>,
            outputNames: Array<String>
        ): Array<FloatArray>? {
            if (!isValid) return null
            val names = inputs.keys.toTypedArray()
            val shapes = inputs.values.map { it.first }.toTypedArray()
            val datas = inputs.values.map { it.second }.toTypedArray()
            return nativeRun(handle, names, shapes, datas, outputNames)
        }

        fun release() {
            if (isValid) nativeRelease(handle)
        }
    }

    fun createSession(modelPath: String, numThreads: Int = 4, forceCpu: Boolean = false): Session {
        return Session(nativeCreateSession(modelPath, numThreads, forceCpu))
    }
}
