package com.moss.tts.demo

import android.app.Activity
import android.content.ContentValues
import android.content.Intent
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.media.MediaPlayer
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import com.moss.tts.MossTtsEngine
import com.moss.tts.TextCleaner
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 极简 demo: 文本输入 + 12音色下拉 + 生成(保存output/ + 播放)
 * 首启自动检测/下载模型 (ModelManager)
 */
class MainActivity : Activity() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private val engine = MossTtsEngine()
    private lateinit var statusText: TextView
    private lateinit var input: EditText
    private lateinit var voiceSpinner: Spinner
    private lateinit var speedInput: EditText
    private lateinit var stretchInput: EditText
    private lateinit var genBtn: Button
    private lateinit var lastLabel: TextView
    private lateinit var replayBtn: Button
    private lateinit var shareBtn: Button
    private lateinit var delBtn: Button
    private var lastUri: Uri? = null

    /** 12个推荐音色 (18个builtin中已筛掉6个弱音色, 详见README) */
    private val voices = listOf(
        4 to "4 悦文 Yuewen (中/女) 默认",
        0 to "0 Junhao (中/男)",
        1 to "1 Zhiming (中/男)",
        2 to "2 Weiguo (中/男)",
        5 to "5 Lingyu (中/女)",
        6 to "6 Trump (英/男)",
        7 to "7 Ava (英/女)",
        8 to "8 Bella (英/女)",
        9 to "9 Adam (英/男)",
        12 to "12 Saki (日/女)",
        14 to "14 Umiri (日/女)",
        15 to "15 Mei (日/女)",
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 48, 24, 24)
        }

        statusText = TextView(this).apply { text = "初始化…" }
        root.addView(statusText)

        input = EditText(this).apply {
            hint = "输入要合成的文本…"
            minLines = 3
            setText("今天天气真好，我们出去玩吧。")
        }
        root.addView(input)

        voiceSpinner = Spinner(this)
        voiceSpinner.adapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item, voices.map { it.second })
        root.addView(voiceSpinner)

        // 语速(停顿压缩 0.5~3.0) + 压伸(WSOLA 1.0~1.5, 音高不变)
        val optsRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        speedInput = EditText(this).apply {
            hint = "1.0"; setText("1.0"); width = 180
            inputType = android.text.InputType.TYPE_CLASS_NUMBER or android.text.InputType.TYPE_NUMBER_FLAG_DECIMAL
        }
        stretchInput = EditText(this).apply {
            hint = "1.0"; setText("1.0"); width = 180
            inputType = android.text.InputType.TYPE_CLASS_NUMBER or android.text.InputType.TYPE_NUMBER_FLAG_DECIMAL
        }
        optsRow.addView(TextView(this).apply { text = "语速(停顿压缩): " })
        optsRow.addView(speedInput)
        optsRow.addView(TextView(this).apply { text = "  压伸(音高不变): " })
        optsRow.addView(stretchInput)
        root.addView(optsRow)

        genBtn = Button(this).apply { text = "生成 (保存到 Download/MOSS-TTS 并播放)" }
        root.addView(genBtn)
        genBtn.setOnClickListener { doGenerate() }

        // 最近一次产物: 重播/分享/删除
        val fileRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        lastLabel = TextView(this).apply { text = "最近: (无)" }
        val lp = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        lastLabel.layoutParams = lp
        replayBtn = Button(this).apply { text = "重播"; isEnabled = false; setOnClickListener { onReplay() } }
        shareBtn = Button(this).apply { text = "分享"; isEnabled = false; setOnClickListener { onShare() } }
        delBtn = Button(this).apply { text = "删除"; isEnabled = false; setOnClickListener { onDelete() } }
        fileRow.addView(lastLabel)
        fileRow.addView(replayBtn)
        fileRow.addView(shareBtn)
        fileRow.addView(delBtn)
        root.addView(fileRow)

        setContentView(ScrollView(this).apply { addView(root) })

        // 首启: 检测模型, 无则自动下载
        genBtn.isEnabled = false
        ModelManager.ensureModel(this,
            onStatus = { runOnUiThread { statusText.text = it } },
            onDone = { ok -> runOnUiThread {
                genBtn.isEnabled = ok
                statusText.text = if (ok) "就绪" else "模型未就绪"
            } })
    }

    private fun doGenerate() {
        val text = input.text.toString()
        val cleaned = TextCleaner.clean(text)
        if (cleaned.isBlank()) {
            statusText.text = "无有效文本(纯标点/emoji会被过滤)"
            return
        }
        val voiceId = voices[voiceSpinner.selectedItemPosition].first
        val speed = speedInput.text.toString().toFloatOrNull()?.coerceIn(0.5f, 3.0f) ?: 1.0f
        val stretch = stretchInput.text.toString().toFloatOrNull()?.coerceIn(1.0f, 1.5f) ?: 1.0f
        genBtn.isEnabled = false
        statusText.text = "生成中…"
        scope.launch {
            try {
                val savedName = withContext(Dispatchers.IO) {
                    if (!engine.isLoaded()) {
                        engine.load(ModelManager.modelDir(this@MainActivity).absolutePath, voiceId, speed)
                    } else {
                        engine.setVoice(voiceId, speed)
                    }
                    engine.setStretch(stretch)
                    val r = engine.generate(cleaned) ?: return@withContext null
                    val stamp = java.text.SimpleDateFormat("yyyyMMdd_HHmmss", java.util.Locale.US)
                        .format(java.util.Date())
                    val name = "tts_${stamp}_v${voiceId}.wav"
                    val uri = saveToDownloads(r.pcmData, r.sampleRate, name)
                    playPcm(r.pcmData, r.sampleRate)
                    if (uri != null) lastUri = uri
                    if (uri != null) name else null
                }
                if (savedName != null) {
                    statusText.text = "完成: Download/MOSS-TTS/$savedName"
                    lastLabel.text = "最近: $savedName"
                    replayBtn.isEnabled = true
                    shareBtn.isEnabled = true
                    delBtn.isEnabled = true
                } else {
                    statusText.text = "生成失败或保存失败"
                }
            } catch (e: Exception) {
                statusText.text = "异常: ${e.message}"
            } finally {
                genBtn.isEnabled = true
            }
        }
    }

    private fun onReplay() {
        val uri = lastUri ?: return
        try {
            MediaPlayer().apply {
                setDataSource(this@MainActivity, uri)
                setOnCompletionListener { it.release() }
                prepare()
                start()
            }
            statusText.text = "重播中…"
        } catch (e: Exception) {
            statusText.text = "重播失败: ${e.message}"
        }
    }

    private fun onShare() {
        val uri = lastUri ?: return
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "audio/wav"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(Intent.createChooser(intent, "分享音频"))
    }

    private fun onDelete() {
        val uri = lastUri ?: return
        try {
            if (Build.VERSION.SDK_INT >= 29) {
                contentResolver.delete(uri, null, null)
            } else {
                uri.path?.let { java.io.File(it).delete() }
            }
        } catch (_: Exception) { }
        lastUri = null
        lastLabel.text = "最近: (已删除)"
        replayBtn.isEnabled = false
        shareBtn.isEnabled = false
        delBtn.isEnabled = false
        statusText.text = "已删除"
    }

    /** 保存到公共 Download/MOSS-TTS/ — API29+ MediaStore免权限, 27~28直接文件(WRITE_EXTERNAL_STORAGE) */
    private fun saveToDownloads(pcm: ShortArray, sr: Int, name: String): Uri? {
        return try {
            if (Build.VERSION.SDK_INT >= 29) {
                val values = ContentValues().apply {
                    put(MediaStore.Downloads.DISPLAY_NAME, name)
                    put(MediaStore.Downloads.MIME_TYPE, "audio/wav")
                    put(MediaStore.Downloads.RELATIVE_PATH,
                        android.os.Environment.DIRECTORY_DOWNLOADS + "/MOSS-TTS")
                }
                val uri = contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                    ?: return null
                contentResolver.openOutputStream(uri)?.use { writeWavStream(it, pcm, sr) }
                uri
            } else {
                val dir = java.io.File(
                    android.os.Environment.getExternalStoragePublicDirectory(
                        android.os.Environment.DIRECTORY_DOWNLOADS), "MOSS-TTS")
                dir.mkdirs()
                val f = java.io.File(dir, name)
                java.io.FileOutputStream(f).use { writeWavStream(it, pcm, sr) }
                Uri.fromFile(f)
            }
        } catch (e: Exception) {
            null
        }
    }

    /** WAV(16bit单声道)写入任意流 */
    private fun writeWavStream(out: java.io.OutputStream, pcm: ShortArray, sr: Int) {
        val dataSize = pcm.size * 2
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        header.put("RIFF".toByteArray()); header.putInt(36 + dataSize); header.put("WAVE".toByteArray())
        header.put("fmt ".toByteArray()); header.putInt(16); header.putShort(1); header.putShort(1)
        header.putInt(sr); header.putInt(sr * 2); header.putShort(2); header.putShort(16)
        header.put("data".toByteArray()); header.putInt(dataSize)
        out.write(header.array())
        val buf = ByteBuffer.allocate(dataSize).order(ByteOrder.LITTLE_ENDIAN)
        for (s in pcm) buf.putShort(s)
        out.write(buf.array())
    }

    private fun playPcm(pcm: ShortArray, sampleRate: Int) {
        val bufSize = AudioTrack.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val track = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
            .setAudioFormat(AudioFormat.Builder()
                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                .setSampleRate(sampleRate)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
            .setBufferSizeInBytes(maxOf(bufSize, pcm.size * 2))
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()
        track.write(pcm, 0, pcm.size)
        track.play()
        Thread.sleep(pcm.size * 1000L / sampleRate + 100)
        track.stop()
        track.release()
    }

    override fun onDestroy() {
        engine.unload()
        scope.cancel()
        super.onDestroy()
    }
}
