package com.moss.tts

import android.util.Log
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.random.Random

/**
 * MOSS TTS引擎 — 纯Kotlin实现
 * 44.1kHz输出，16通道VQ编码
 *
 * 架构:
 *   text → sentencepiece tokenize → prefill → generate frames → codec decode → audio
 */
class MossTtsEngine : TtsEngine {
    companion object {
        private const val TAG = "MossTts"
        // Fix①(2026-10-10): 48000=codec_meta官方sample_rate(3840样本/帧=80ms)。
        // 旧44100=错误标注, 导致所有输出慢放8.8%/低8.8%音高。
        private const val SAMPLE_RATE = 48000
        private const val N_VQ = 16
        private const val MAX_NEW_FRAMES = 400  // 15句长文本需要~350帧, should_continue会提前终止
        // 根据文本长度估算帧数
        private fun estimateMaxFrames(tokenCount: Int): Int = when {
            tokenCount <= 5 -> 40    // 1-2句短话
            tokenCount <= 15 -> 80   // 一段话
            tokenCount <= 40 -> 200  // 5-10句
            else -> MAX_NEW_FRAMES   // 15句+
        }
        private const val THREADS = 2
    }

    private var loaded = false
    private var modelDir: String? = null
    private var manifest: JSONObject? = null
    private var currentVoiceIdx: Int = 5  // 默认Lingyu深夜电台(女声)
    private var currentSpeed: Float = 1.0f
    private var currentStretch: Float = 1.0f  // Fix③: WSOLA压伸倍数 1.0~1.5
    private var ttsMeta: JSONObject? = null
    private var codecMeta: JSONObject? = null

    // ONNX sessions (lazy loaded)
    private var prefillSession: Long = 0
    private var decodeSession: Long = 0
    private var localFixedSampledFrameSession: Long = 0
    private var codecDecodeSession: Long = 0

    // Fix A(2026-10-10): generate/unload/load互斥锁。
    // 之前mossContext是类成员字段+finally清理, 两个并发generate互删对方context(UAF, 静默丢句)。
    // 现context全程为generate()局部变量; 锁保证unload不会在生成中途释放session。
    // 用synchronized而非Mutex: unload()是非suspend接口, 且锁内无挂起点, monitor等效且可重入(load调unload)。
    private val genLock = Any()

    // Tokenizer
    private var tokenizer: MossTokenizer? = null

    // Random generator for sampling
    private val rng = Random(1234)

    override suspend fun load(modelPath: String, voiceId: Int, speed: Float): Boolean {
        return try {
            unload()
            modelDir = modelPath

            // 加载manifest
            val manifestFile = findFile(modelPath, "browser_poc_manifest.json",
                "MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json")
            if (manifestFile == null) {
                Log.e(TAG, "manifest.json不存在")
                return false
            }
            manifest = JSONObject(manifestFile.readText())

            // 加载tts_meta
            val ttsMetaPath = resolvePath(manifest!!.getJSONObject("model_files").getString("tts_meta"))
            ttsMeta = JSONObject(ttsMetaPath.readText())

            // 加载codec_meta
            val codecMetaPath = resolvePath(manifest!!.getJSONObject("model_files").getString("codec_meta"))
            codecMeta = JSONObject(codecMetaPath.readText())

            // 加载tokenizer
            val tokenizerPath = resolvePath(manifest!!.getJSONObject("model_files").optString("tokenizer_model", "tokenizer.model"))
            tokenizer = MossTokenizer(tokenizerPath.absolutePath)

            loaded = true
            currentVoiceIdx = voiceId  // 应用传入的音色
            currentSpeed = speed.coerceIn(0.5f, 3.0f)  // Fix F: load()的speed参数之前被忽略(只有setVoice会设)

            // 对齐Python: 强制设置采样配置 (tts_moss.py generate_moss)
            try {
                val genDefaults = manifest!!.optJSONObject("generation_defaults")
                if (genDefaults != null) {
                    genDefaults.put("do_sample", true)
                    genDefaults.put("sample_mode", "fixed")
                    genDefaults.put("text_temperature", 0.8)
                    genDefaults.put("audio_repetition_penalty", 1.2)
                    Log.i(TAG, "已设置采样配置: do_sample=true, sample_mode=fixed, text_temperature=0.8, audio_repetition_penalty=1.2")
                }
            } catch (e: Exception) {
                Log.w(TAG, "设置采样配置失败", e)
            }

            Log.i(TAG, "MOSS TTS加载成功: $modelPath (voice=$voiceId)")
            true
        } catch (e: Exception) {
            Log.e(TAG, "MOSS TTS加载失败", e)
            false
        }
    }

    override fun unload() = synchronized(genLock) {
        // Fix A: 与generate互斥, 不会在生成中途释放session/context
        if (prefillSession != 0L) OrtNative.nativeRelease(prefillSession)
        if (decodeSession != 0L) OrtNative.nativeRelease(decodeSession)
        if (localFixedSampledFrameSession != 0L) OrtNative.nativeRelease(localFixedSampledFrameSession)
        if (codecDecodeSession != 0L) OrtNative.nativeRelease(codecDecodeSession)
        prefillSession = 0; decodeSession = 0; localFixedSampledFrameSession = 0
        codecDecodeSession = 0
        loaded = false; modelDir = null; manifest = null; ttsMeta = null; codecMeta = null
        tokenizer = null
    }

    override fun isLoaded(): Boolean = loaded
    override fun setVoice(voiceId: Int, speed: Float) {
        // voiceId直接对应builtin_voices数组索引
        // 0=Junhao男,1=Zhiming男,2=Weiguo男,3=Xiaoyu女,4=Yuewen女,5=Lingyu女,6=Trump男...
        currentVoiceIdx = voiceId
        currentSpeed = speed.coerceIn(0.5f, 3.0f)
        Log.i(TAG, "Voice changed: idx=$voiceId, speed=$currentSpeed")
    }

    override fun setStretch(ratio: Float) {
        currentStretch = ratio.coerceIn(1.0f, 1.5f)
        Log.i(TAG, "Stretch changed: x$currentStretch")
    }

    override suspend fun generate(text: String): TtsResult? {
        if (text.isBlank()) return null

        // Fix A(2026-10-10): 全程持锁 + ctx局部化。
        // 锁内无挂起点(JNI全程阻塞), synchronized安全; 与其他generate/unload/load互斥。
        synchronized(genLock) {
            if (!loaded || manifest == null) return null

            var ctx = 0L  // Fix A: 局部context, 只属于本次generate, finally自清理
            return try {
            val tTotal = System.currentTimeMillis()

            ensureSessions()
            val tSessions = System.currentTimeMillis() - tTotal

            val m = manifest!!
            val textCleaned = text.trim()

            // 1. Tokenize
            val t1 = System.currentTimeMillis()
            val textTokenIds = tokenizer!!.encode(textCleaned)
            val tTokenize = System.currentTimeMillis() - t1
            Log.i(TAG, "Tokenized: ${textTokenIds.size} tokens (${tTokenize}ms) ids=${textTokenIds.take(40)}")

            // 2. Build request rows
            val t2 = System.currentTimeMillis()
            val requestRows = buildVoiceCloneRequestRows(textTokenIds)
            val tBuildRows = System.currentTimeMillis() - t2

            // 3. Init MOSS context (prefill, C++持有past cache)
            val t3 = System.currentTimeMillis()
            ctx = initMossContext(requestRows)
            val tPrefill = System.currentTimeMillis() - t3
            Log.i(TAG, "MossContext创建: ctx=0x${java.lang.Long.toHexString(ctx)}")
            if (ctx == 0L) {
                Log.e(TAG, "MossContext初始化失败")
                return null
            }

            // 4. 一次JNI调用生成全部音频（C++内部循环+codec解码）
            val ttsConfig = m.getJSONObject("tts_config")
            val genDefaults = m.getJSONObject("generation_defaults")
            val audioCodebookSize = ttsMeta!!.getJSONObject("model_config")
                .getJSONArray("audio_codebook_sizes").getInt(0)
            // Fix B(2026-10-10): estimateMaxFrames接上——短文本按token数封顶(≤5 token→40帧≈3.5s),
            // 退化输入(纯标点/无句末标点)不再跑飞到400帧; 400仍是绝对上限(else分支)。
            val maxFrames = estimateMaxFrames(textTokenIds.size)
            Log.i(TAG, "maxFrames上限: tokens=${textTokenIds.size} → $maxFrames")

            val t4 = System.currentTimeMillis()
            val pcmBytes = OrtNative.nativeMossGenerateAudio(ctx, maxFrames, audioCodebookSize)
            val tGenerate = System.currentTimeMillis() - t4

            if (pcmBytes == null || pcmBytes.size < 6) {
                Log.e(TAG, "MossGenerateAudio返回null或空")
                return null
            }

            // 解析: 前4字节=sampleCount, 后面=int16 PCM
            val sampleCount = java.nio.ByteBuffer.wrap(pcmBytes, 0, 4).order(java.nio.ByteOrder.LITTLE_ENDIAN).int
            if (sampleCount <= 0) {
                Log.w(TAG, "MossGenerateAudio: 0 samples")
                return null
            }

            val audioData = ShortArray(sampleCount)
            val buf = java.nio.ByteBuffer.wrap(pcmBytes, 4, sampleCount * 2).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            for (i in 0 until sampleCount) audioData[i] = buf.short

            // P1#10: 裁剪尾部静音 (对齐Python _trim_trailing_silence)
            val trimmedLen = trimTrailingSilence(audioData, sampleCount)
            val finalAudio = if (trimmedLen < sampleCount) {
                audioData.copyOf(trimmedLen)
            } else audioData

            // Fix F(2026-10-10): 语速=字间停顿压缩(剪空白)。旧重采样按speed丢样本=升调
            // (用户反馈1.2/1.3音色尖锐), 已废弃。现在只剪静音段, 发音本体零改动。
            var resultAudio = compressGaps(finalAudio, currentSpeed)
            // Fix③(2026-10-10): WSOLA整句压伸(音高/音色不变), stretch=1.0时零开销
            if (currentStretch > 1.01f) {
                val tWs = System.currentTimeMillis()
                val wsInLen = resultAudio.size
                resultAudio = Wsola.stretch(resultAudio, currentStretch)
                Log.i(TAG, "WSOLA压伸: x$currentStretch, ${wsInLen}→${resultAudio.size} samples, ${System.currentTimeMillis() - tWs}ms")
            }

            val tTotalElapsed = System.currentTimeMillis() - tTotal
            val audioMs = resultAudio.size * 1000L / SAMPLE_RATE
            Log.i(TAG, "[TTS-BENCH Kotlin] sessions=${tSessions}ms, tokenize=${tTokenize}ms, buildRows=${tBuildRows}ms, prefill=${tPrefill}ms, generate=${tGenerate}ms, total=${tTotalElapsed}ms")
            Log.i(TAG, "[TTS-BENCH Kotlin] 音频: ${resultAudio.size} samples = ${audioMs}ms, RTF=${if(tTotalElapsed>0) String.format("%.2f", tTotalElapsed.toFloat()/audioMs) else "N/A"}")

            TtsResult(pcmData = resultAudio, sampleRate = SAMPLE_RATE)
        } catch (e: Exception) {
            Log.e(TAG, "MOSS生成失败", e)
            null
        } finally {
            // Fix A: 只清理本次generate自己的ctx(指针日志自证无互删)
            if (ctx != 0L) {
                Log.i(TAG, "MossContext清理: ctx=0x${java.lang.Long.toHexString(ctx)}")
                OrtNative.nativeMossCleanup(ctx)
                ctx = 0
            }
        }
        }
    }

    /**
     * Fix B(2026-10-10): 引擎级预热 — 有界帧数(maxFrames=2), 不走generate("预热")。
     * 旧warmup用裸generate("预热")跑出344帧30秒垃圾音频还与对话generate竞态, 已废弃。
     */
    override fun prewarm() {
        synchronized(genLock) {
            if (!loaded) return
            try {
                ensureSessions()  // 内含maxFrames=2的有界warmup
                Log.i(TAG, "TTS引擎预热完成(有界)")
            } catch (e: Exception) {
                Log.w(TAG, "TTS引擎预热失败(不影响使用)", e)
            }
        }
    }

    // ============ Session管理 ============

    private fun ensureSessions() {
        if (prefillSession != 0L) return

        val ttsDir = ttsMeta!!.getJSONObject("files")
        val codecDir = codecMeta!!.getJSONObject("files")

        prefillSession = loadSession(resolvePath(ttsDir.getString("prefill")).absolutePath)
        decodeSession = loadSession(resolvePath(ttsDir.getString("decode_step")).absolutePath)
        localFixedSampledFrameSession = loadSession(resolvePath(ttsDir.getString("local_fixed_sampled_frame")).absolutePath)
        codecDecodeSession = loadSession(resolvePath(codecDir.getString("decode_full")).absolutePath)

        Log.i(TAG, "Sessions loaded: prefill=$prefillSession decode=$decodeSession codec=$codecDecodeSession")

        // P2#14: warmup — 预热ONNX runtime，首次generate不冷启动
        try {
            val warmupTokens = tokenizer!!.encode("你好")
            val warmupRows = buildVoiceCloneRequestRows(warmupTokens)
            val warmupCtx = initMossContext(warmupRows)
            if (warmupCtx != 0L) {
                val ttsConfig = ttsMeta!!.getJSONObject("model_config")
                val codebookSize = ttsConfig.getJSONArray("audio_codebook_sizes").getInt(0)
                OrtNative.nativeMossGenerateAudio(warmupCtx, 2, codebookSize)
                OrtNative.nativeMossCleanup(warmupCtx)
                Log.i(TAG, "Warmup完成")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Warmup失败(不影响使用)", e)
        }
    }

    private fun loadSession(path: String): Long {
        return OrtNative.nativeCreateSession(path, THREADS, true)
    }

    // ============ Text构建 ============

    private fun buildVoiceCloneRequestRows(textTokenIds: List<Int>): Map<String, Any> {
        val m = manifest!!
        val ttsConfig = m.getJSONObject("tts_config")
        val templates = m.getJSONObject("prompt_templates")

        val voices = m.getJSONArray("builtin_voices")
        val voiceIdx = currentVoiceIdx.coerceIn(0, voices.length() - 1)
        val voice = voices.getJSONObject(voiceIdx)
        val codesArr = voice.getJSONArray("prompt_audio_codes")
        val promptCodes = mutableListOf<List<Int>>()
        for (i in 0 until codesArr.length()) {
            val row = mutableListOf<Int>()
            val rowArr = codesArr.getJSONArray(i)
            for (j in 0 until rowArr.length()) row.add(rowArr.getInt(j))
            promptCodes.add(row)
        }

        val prefixTokens = mutableListOf<Int>()
        val prefixArr = templates.getJSONArray("user_prompt_prefix_token_ids")
        for (i in 0 until prefixArr.length()) prefixTokens.add(prefixArr.getInt(i))
        prefixTokens.add(ttsConfig.getInt("audio_start_token_id"))

        val suffixTokens = mutableListOf<Int>()
        suffixTokens.add(ttsConfig.getInt("audio_end_token_id"))
        val afterRefArr = templates.getJSONArray("user_prompt_after_reference_token_ids")
        for (i in 0 until afterRefArr.length()) suffixTokens.add(afterRefArr.getInt(i))
        suffixTokens.addAll(textTokenIds)
        val assistPrefixArr = templates.getJSONArray("assistant_prompt_prefix_token_ids")
        for (i in 0 until assistPrefixArr.length()) suffixTokens.add(assistPrefixArr.getInt(i))
        suffixTokens.add(ttsConfig.getInt("audio_start_token_id"))

        val rows = mutableListOf<List<Int>>()
        rows.addAll(buildTextRows(prefixTokens))
        rows.addAll(buildAudioPrefixRows(promptCodes))
        rows.addAll(buildTextRows(suffixTokens))

        return mapOf(
            "inputIds" to rows,
            "attentionMask" to listOf(rows.map { 1 })
        )
    }

    private fun buildTextRows(tokenIds: List<Int>): List<List<Int>> {
        val ttsConfig = manifest!!.getJSONObject("tts_config")
        val rowWidth = ttsConfig.getInt("n_vq") + 1
        val padId = ttsConfig.getInt("audio_pad_token_id")
        return tokenIds.map { id ->
            MutableList(rowWidth) { padId }.apply { this[0] = id }
        }
    }

    private fun buildAudioPrefixRows(promptCodes: List<List<Int>>): List<List<Int>> {
        val ttsConfig = manifest!!.getJSONObject("tts_config")
        val rowWidth = ttsConfig.getInt("n_vq") + 1
        val padId = ttsConfig.getInt("audio_pad_token_id")
        val slotId = ttsConfig.getInt("audio_user_slot_token_id")
        val nVq = ttsConfig.getInt("n_vq")

        return promptCodes.map { codeRow ->
            MutableList(rowWidth) { padId }.apply {
                this[0] = slotId
                for (i in 0 until minOf(codeRow.size, nVq)) {
                    this[i + 1] = codeRow[i]
                }
            }
        }
    }

    // ============ 音频帧生成 (新C++架构) ============

    /** 初始化MOSS上下文: prefill + 存储past cache到C++层 */
    private fun initMossContext(requestRows: Map<String, Any>): Long {
        val m = manifest!!
        val ttsConfig = m.getJSONObject("tts_config")
        val nVq = ttsConfig.getInt("n_vq")
        val rowWidth = nVq + 1

        @Suppress("UNCHECKED_CAST")
        val inputIds = requestRows["inputIds"] as List<List<Int>>
        @Suppress("UNCHECKED_CAST")
        val attentionMask = requestRows["attentionMask"] as List<List<Int>>

        val prefillIds = flatten3d(listOf(inputIds))
        val prefillMask = flatten2d(attentionMask)

        // 转ByteArray
        val idsBytes = ByteArray(prefillIds.first.size * 4)
        ByteBuffer.wrap(idsBytes).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().put(prefillIds.first)
        val idsShape = prefillIds.second

        val maskBytes = ByteArray(prefillMask.first.size * 4)
        ByteBuffer.wrap(maskBytes).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().put(prefillMask.first)
        val maskShape = prefillMask.second

        return OrtNative.nativeMossInit(
            prefillSession, decodeSession,
            localFixedSampledFrameSession, codecDecodeSession,
            idsBytes, idsShape,
            maskBytes, maskShape,
            nVq, rowWidth,
            ttsConfig.getInt("audio_pad_token_id"),
            ttsConfig.getInt("audio_assistant_slot_token_id")
        )
    }

    /** 解码一帧chunk: frames -> PCM (使用C++ codec decode) */
    private fun decodeFrameChunk(frames: List<List<Int>>, nVq: Int): ShortArray? {
        if (frames.isEmpty()) return null
        val frameCount = frames.size

        // 转置: frames[frame][channel] -> codes[channel][frame], int64
        val codesBytes = ByteArray(nVq * frameCount * 8)
        val codesBuf = ByteBuffer.wrap(codesBytes).order(ByteOrder.LITTLE_ENDIAN)
        for (ch in 0 until nVq) {
            for (f in 0 until frameCount) {
                codesBuf.putLong(frames[f][ch].toLong())
            }
        }

        val codesLengthsBytes = ByteArray(8)
        ByteBuffer.wrap(codesLengthsBytes).order(ByteOrder.LITTLE_ENDIAN).putLong(frameCount.toLong())

        val result = OrtNative.nativeMossCodecDecode(
            codecDecodeSession,
            codesBytes, intArrayOf(nVq, 1, frameCount),
            codesLengthsBytes, intArrayOf(1)
        ) ?: return null

        // 解析: 前8字节=audio_length(int64), 后面=float32 PCM
        val audioLength = ByteBuffer.wrap(result, 0, 8).order(ByteOrder.LITTLE_ENDIAN).long.toInt()
        val audioBuf = ByteBuffer.wrap(result, 8, result.size - 8).order(ByteOrder.LITTLE_ENDIAN)
        val pcm = ShortArray(audioLength)
        for (i in 0 until audioLength) {
            val sample = audioBuf.getFloat(i * 4)
            pcm[i] = (sample * 32767f).coerceIn(-32768f, 32767f).toInt().toShort()
        }
        return pcm
    }

    // ============ 音频解码(旧版，保留兼容) ============

    private fun decodeFullAudio(frames: List<List<Int>>): ShortArray? {
        return decodeFrameChunk(frames, frames[0].size)
    }

    // ============ 辅助函数 ============

    private fun flatten3d(data: List<List<List<Int>>>): Pair<IntArray, IntArray> {
        val d0 = data.size; val d1 = data[0].size; val d2 = data[0][0].size
        val flat = IntArray(d0 * d1 * d2)
        var offset = 0
        for (i in data) for (j in i) for (k in j) flat[offset++] = k
        return flat to intArrayOf(d0, d1, d2)
    }

    private fun flatten2d(data: List<List<Int>>): Pair<IntArray, IntArray> {
        val d0 = data.size; val d1 = data[0].size
        val flat = IntArray(d0 * d1)
        var offset = 0
        for (i in data) for (j in i) flat[offset++] = j
        return flat to intArrayOf(d0, d1)
    }

    private fun findFile(base: String, vararg candidates: String): File? {
        for (c in candidates) {
            val f = File(base, c)
            if (f.exists()) return f
        }
        return null
    }

    private fun resolvePath(relative: String): File {
        val base = File(modelDir!!)
        val candidates = listOf(
            File(base, relative),
            File(base, "MOSS-TTS-Nano-100M-ONNX/$relative"),
            File(base, "MOSS-Audio-Tokenizer-INT8/$relative"),
            File(base.parentFile ?: base, relative),
            File(base, "../MOSS-Audio-Tokenizer-INT8/$relative"),
        )
        for (c in candidates) if (c.exists()) return c.canonicalFile
        return File(base, relative)
    }

    /** 裁剪尾部静音: 从末尾扫描，找到最后一个幅度>阈值的样本 */
    private fun trimTrailingSilence(pcm: ShortArray, length: Int): Int {
        val threshold = 100  // 幅度阈值
        var end = length - 1
        // 跳过末尾连续静音
        while (end > SAMPLE_RATE / 10 && kotlin.math.abs(pcm[end].toInt()) < threshold) {
            end--
        }
        // 保留一点尾部余韵(50ms)
        return (end + SAMPLE_RATE / 20).coerceAtMost(length)
    }

    /**
     * Fix F(2026-10-10): 停顿压缩(剪空白) — 只缩短静音段, 发音本体零改动, 音色/音高不变。
     * 每段静音保留 keep = clamp(min(段长, 800ms)/speed^5, 15ms, 段长):
     *   speed=1.0 → 正常停顿原样保留, 只把>800ms的怪静音截到800ms(治日韩1~2秒怪静音)
     *   speed=1.3 → 停顿压到约1/3.7(中文总时长-10%级, 日韩英-20%级)
     *   speed<1.0 → 停顿不变(不拉长, 语义: 语速只影响紧凑度)
     * 实测依据: 中文停顿占~15%(多为25-80ms字间微停顿), 日韩英22~26%(含0.5~2s怪静音)。
     * 旧实现按speed丢样本重采样=升调(音色尖锐), 已废弃。
     */
    private fun compressGaps(pcm: ShortArray, speed: Float): ShortArray {
        if (pcm.isEmpty()) return pcm
        val threshold = 300                      // 幅度<300(约-40dB)视为静音
        val gapCapMs = 800.0                     // 单段静音保留上限
        val gapFloorMs = 15.0                    // 每段至少保留(防字词粘连)
        val keepRatio = 1.0 / Math.pow(speed.toDouble().coerceIn(0.5, 3.0), 5.0)
        val out = ShortArray(pcm.size)
        var w = 0
        var i = 0
        var removedMs = 0L
        while (i < pcm.size) {
            if (Math.abs(pcm[i].toInt()) < threshold) {
                var j = i
                while (j < pcm.size && Math.abs(pcm[j].toInt()) < threshold) j++
                val gapLen = j - i
                val gapMs = gapLen * 1000.0 / SAMPLE_RATE
                // 先下限再截回段长: 短于15ms的段(<gapFloorMs)原样保留, 不能coerceIn(空范围会抛异常)
                val keepMs = (Math.min(gapMs, gapCapMs) * keepRatio).coerceAtLeast(gapFloorMs).coerceAtMost(gapMs)
                val keep = Math.round(keepMs * SAMPLE_RATE / 1000.0).toInt().coerceIn(1, gapLen)
                System.arraycopy(pcm, i, out, w, keep)
                w += keep
                removedMs += (gapLen - keep) * 1000L / SAMPLE_RATE
                i = j
            } else {
                out[w++] = pcm[i++]
            }
        }
        Log.i(TAG, "剪空白: speed=$speed, ${pcm.size}→$w samples (${pcm.size * 1000L / SAMPLE_RATE}→${w * 1000L / SAMPLE_RATE}ms), 剪掉静音${removedMs}ms")
        return if (w == pcm.size) pcm else out.copyOf(w)
    }
}
