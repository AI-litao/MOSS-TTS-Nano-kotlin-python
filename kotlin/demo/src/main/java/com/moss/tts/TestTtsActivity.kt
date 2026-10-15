package com.moss.tts

import android.app.Activity
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import com.moss.tts.MossTtsEngine
import kotlinx.coroutines.*
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

class TestTtsActivity : Activity() {
    companion object {
        private const val TAG = "TestTts"
    }

    private lateinit var logView: TextView
    private lateinit var scrollView: ScrollView
    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    private val engine = MossTtsEngine()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val layout = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        scrollView = ScrollView(this)
        logView = TextView(this).apply {
            textSize = 13f
            setPadding(16, 16, 16, 16)
        }
        scrollView.addView(logView)
        layout.addView(scrollView, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))

        fun addButton(label: String, action: suspend () -> Unit) {
            layout.addView(Button(this).apply {
                text = label
                setOnClickListener { scope.launch { action() } }
            })
        }

        setContentView(layout)

        log("=== MOSS TTS 独立测试 ===")
        log("不依赖LLM/Wav2Lip/Pipeline")

        // 支持adb intent传文本: am start -n com.moss.tts/.TestTtsActivity --es text "测试文本" --es tag "name"
        // 也支持从文件读取: --es file "/sdcard/Download/test.txt"
        val intentText = intent.getStringExtra("text")
            ?: intent.getStringExtra("file")?.let { filePath ->
                try { java.io.File(filePath).readText().trim() } catch (e: Exception) { log("读取文件失败: $filePath"); null }
            }
        val intentTag = intent.getStringExtra("tag") ?: "test"

        if (intentText != null) {
            val voiceId = intent.getIntExtra("voice", 4) // 默认悦文
            val raw = intent.getBooleanExtra("raw", false) // true=跳过TextCleaner
            val speed = intent.getFloatExtra("speed", 1.0f) // Fix F: 剪空白语速(停顿压缩强度)
            val stretch = intent.getFloatExtra("stretch", 1.0f) // Fix③: WSOLA压伸 1.0~1.5(音高不变)
            log("Intent模式: tag=$intentTag, voice=$voiceId, raw=$raw, speed=$speed, stretch=$stretch")
            log("文本: $intentText")
            scope.launch {
                loadModel(voiceId, speed)
                if (engine.isLoaded()) {
                    engine.setStretch(stretch)
                    testAndSave(intentText, intentTag, skipCleaner = raw)
                    engine.unload()
                    log("DONE")
                }
            }
        } else {
            addButton("1. 加载模型") { loadModel(4) }
            addButton("2. 短句(1句)") { testTts("你好呀宝贝") }
            addButton("3. 中句(3句)") { testTts("今天天气真好，我们出去玩吧，公园里有很多漂亮的花") }
            addButton("4. 长句(5句)") { testTts("早上好呀，你想吃点什么呀，我刚学会了做蛋糕，要不要尝一尝呢，保证很好吃的哦") }
            addButton("5. 超长(10句)") { testTts("我今天去了超市，买了很多好吃的东西，有水果有零食，还给你买了你最喜欢的巧克力，回来的路上看到了一只很可爱的小猫，它毛茸茸的特别乖，我好想把它带回家呀，你说我们养一只猫好不好，它一定很听话的，你觉得怎么样呢") }
            addButton("6. 极长(15句)") { testTts("昨天晚上我做了一个很奇怪的梦，梦见自己飞到了天上，云朵软绵绵的像棉花糖一样，我伸手去抓结果真的抓到了一块，放进嘴里居然是草莓味的，然后我继续往前飞，看到了一座彩虹桥，桥的那边有一座小城堡，城堡里面住着一只会说话的兔子，它请我喝了一杯热可可，味道特别香浓，我们聊了很久很久，它告诉我很多有趣的故事，后来闹钟响了我就醒了，醒来嘴角还带着笑呢，下次还想做这样的梦") }
            addButton("7. 释放模型") { engine.unload(); log("已释放") }
        }
    }

    private suspend fun loadModel(voiceId: Int = 4, speed: Float = 1.0f) = withContext(Dispatchers.IO) {
        log("正在加载MOSS TTS...")
        val t0 = System.currentTimeMillis()
        val modelPath = filesDir.absolutePath + "/moss_tts"
        log("模型路径: $modelPath")
        val ok = engine.load(modelPath, voiceId = voiceId, speed = speed)
        val elapsed = System.currentTimeMillis() - t0
        if (ok) {
            log("加载成功! (${elapsed}ms) voice=$voiceId")
        } else {
            log("加载失败! (${elapsed}ms)")
        }
    }

    /** adb intent模式: 生成音频+保存WAV+播放 */
    private suspend fun testAndSave(text: String, tag: String, skipCleaner: Boolean = false) = withContext(Dispatchers.IO) {
        log("")
        log("=== 测试: tag=$tag ===")
        log("输入: '$text'")

        // 应用TextCleaner（或跳过）
        val cleaned = if (skipCleaner) {
            log("跳过TextCleaner (raw模式)")
            text
        } else {
            val c = com.moss.tts.TextCleaner.clean(text)
            log("清理后: '$c'")
            c
        }
        // hex dump输入前20字符
        val inHex = text.take(20).map { "U+${it.code.toString(16).uppercase().padStart(4,'0')}" }.joinToString(" ")
        log("输入hex: $inHex")
        // hex dump输出前20字符
        val outHex = cleaned.take(20).map { "U+${it.code.toString(16).uppercase().padStart(4,'0')}" }.joinToString(" ")
        log("输出hex: $outHex")

        val t0 = System.currentTimeMillis()
        val result = engine.generate(cleaned)
        val tGen = System.currentTimeMillis() - t0

        if (result == null) {
            log("生成失败!")
            return@withContext
        }

        val audioMs = result.pcmData.size * 1000L / result.sampleRate
        log("生成: ${tGen}ms, 音频: ${audioMs}ms, RTF=${String.format("%.2f", tGen.toFloat() / audioMs)}")
        log("samples: ${result.pcmData.size}, sr: ${result.sampleRate}")

        // 保存WAV到filesDir/tts_tests/
        val outDir = File(filesDir, "tts_tests")
        outDir.mkdirs()
        val outFile = File(outDir, "${tag}.wav")
        saveWav(outFile, result.pcmData, result.sampleRate)
        log("WAV已保存: ${outFile.absolutePath}")
        log("文件大小: ${outFile.length()} bytes")

        // 播放 (intent --ez play false 可跳过)
        if (intent.getBooleanExtra("play", true)) {
            playPcm(result.pcmData, result.sampleRate)
            log("播放完成")
        } else {
            log("跳过播放(noplay模式)")
        }
    }

    private suspend fun testTts(text: String) = withContext(Dispatchers.IO) {
        if (!engine.isLoaded()) {
            log("错误: 先加载模型")
            return@withContext
        }
        log("")
        log("=== 测试: \"${text.take(30)}...\" ===")
        log("文本长度: ${text.length}字")

        val sentences = splitSentences(text)
        log("分句: ${sentences.size}句")

        val totalStart = System.currentTimeMillis()
        var totalAudioMs = 0L
        var totalGenMs = 0L

        for ((idx, sent) in sentences.withIndex()) {
            log("")
            log("--- 句${idx + 1}/${sentences.size}: \"${sent.take(20)}\" ---")
            val t0 = System.currentTimeMillis()
            val result = engine.generate(sent)
            val tGen = System.currentTimeMillis() - t0
            totalGenMs += tGen

            if (result == null) {
                log("生成失败!")
                continue
            }

            val audioMs = result.pcmData.size * 1000L / result.sampleRate
            totalAudioMs += audioMs
            log("生成: ${tGen}ms, 音频: ${audioMs}ms, RTF=${String.format("%.2f", tGen.toFloat() / audioMs)}")

            playPcm(result.pcmData, result.sampleRate)
            log("播放完成")
        }

        val totalTime = System.currentTimeMillis() - totalStart
        log("")
        log("=== 汇总 ===")
        log("总句数: ${sentences.size}")
        log("总生成: ${totalGenMs}ms")
        log("总音频: ${totalAudioMs}ms")
        log("总耗时: ${totalTime}ms")
        log("整体RTF=${String.format("%.2f", totalGenMs.toFloat() / totalAudioMs)}")
        log("=== 完成 ===")
    }

    /** 保存PCM为WAV文件 */
    private fun saveWav(file: File, pcm: ShortArray, sampleRate: Int) {
        val dataSize = pcm.size * 2
        val fileSize = 36 + dataSize

        RandomAccessFile(file, "rw").use { raf ->
            // RIFF header
            raf.writeBytes("RIFF")
            raf.write(intToLittleEndian(fileSize))
            raf.writeBytes("WAVE")

            // fmt chunk
            raf.writeBytes("fmt ")
            raf.write(intToLittleEndian(16))        // chunk size
            raf.write(shortToLittleEndian(1))        // PCM format
            raf.write(shortToLittleEndian(1))        // mono
            raf.write(intToLittleEndian(sampleRate)) // sample rate
            raf.write(intToLittleEndian(sampleRate * 2)) // byte rate
            raf.write(shortToLittleEndian(2))        // block align
            raf.write(shortToLittleEndian(16))       // bits per sample

            // data chunk
            raf.writeBytes("data")
            raf.write(intToLittleEndian(dataSize))

            // PCM data
            val buf = ByteBuffer.allocate(dataSize).order(ByteOrder.LITTLE_ENDIAN)
            for (sample in pcm) buf.putShort(sample)
            raf.write(buf.array())
        }
    }

    private fun intToLittleEndian(value: Int): ByteArray {
        return ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(value).array()
    }

    private fun shortToLittleEndian(value: Short): ByteArray {
        return ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort(value).array()
    }

    /** 按标点分句，每句≤30字。含英文". "硬切句(前面非数字，避免拆小数) */
    private fun splitSentences(text: String): List<String> {
        val result = mutableListOf<String>()
        val buf = StringBuilder()
        var i = 0
        while (i < text.length) {
            val ch = text[i]
            buf.append(ch)
            // 英文句号+空格 → 硬切 (句号保留在前句; 前面是数字则视为小数点不切)
            val isEnglishPeriod = ch == '.' && i + 1 < text.length && text[i + 1] == ' '
                && buf.length >= 2 && !buf[buf.length - 2].isDigit()
            if ((ch in "。！？!?；\n" || isEnglishPeriod) && buf.length >= 2) {
                result.add(buf.toString().trim())
                buf.clear()
            } else if (buf.length >= 30 && ch in "，,、；;") {
                result.add(buf.toString().trim())
                buf.clear()
            }
            i++
        }
        if (buf.isNotBlank()) result.add(buf.toString().trim())
        return result.filter { it.length > 1 }
    }

    private fun playPcm(pcm: ShortArray, sampleRate: Int) {
        val bufSize = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        val track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(maxOf(bufSize, pcm.size * 2))
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()

        track.write(pcm, 0, pcm.size)
        track.play()

        val durationMs = pcm.size * 1000L / sampleRate
        Thread.sleep(durationMs + 100)

        track.stop()
        track.release()
    }

    private fun log(msg: String) {
        Log.i(TAG, msg)
        runOnUiThread {
            logView.append(msg + "\n")
            scrollView.post { scrollView.fullScroll(ScrollView.FOCUS_DOWN) }
        }
    }

    override fun onDestroy() {
        engine.unload()
        scope.cancel()
        super.onDestroy()
    }
}