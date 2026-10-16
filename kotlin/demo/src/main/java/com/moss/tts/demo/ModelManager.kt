package com.moss.tts.demo

import android.content.Context
import android.net.wifi.WifiManager
import android.os.PowerManager
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * 模型管理: 首启检测 -> 无模型逐文件下载(16文件, 每个 md5 校验) -> files/moss_tts
 * 优化: 前台服务+WakeLock+WifiLock(防熄屏/后台被掐网络); 断连自动 Range 断点续传
 * 手动方式(可选): adb push 模型文件到 /data/local/tmp 再 run-as 拷入 files/moss_tts,
 *   或把整个模型目录放到 getExternalFilesDir(null)/moss_tts（见README）
 */
object ModelManager {
    private const val TAG = "MossModel"

    /** 模型文件直链前缀 (魔搭 resolve 直链) */
    const val MODEL_BASE_URL = "https://www.modelscope.cn/models/ArchitectLi/MOSS-TTS-Nano-int8/resolve/master"

    /** 模型平铺清单(16文件, 与md5sums.txt一致): 文件名 to md5 */
    val MODEL_FILES = listOf(
        "browser_poc_manifest.json" to "c19a8289fee80697fa871e924ff45e52",
        "codec_browser_onnx_meta.json" to "3261f9c906dd72e86450dc0b07c8ca47",
        "config.json" to "620f6e74bbd0e94aaaba9c062d9fcb29",
        "global_shared.data" to "6c7114481aa1b4a33e1fed3884162bba",
        "local_shared.data" to "649ba6194564242239786e937f21ddc6",
        "moss_audio_tokenizer_decode_full_int8.onnx" to "adac133cd7bde48ebf06835a3dd3fa77",
        "moss_tts_decode_step.onnx" to "c1fa623f337471759500e44260413058",
        "moss_tts_local_cached_step.onnx" to "990be57ee616dc9dd10f13b0af1b2107",
        "moss_tts_local_decoder.onnx" to "0621f01b4724a67a88a4e068d1a4dbb5",
        "moss_tts_local_fixed_sampled_frame.onnx" to "31eaa1c6a635282f7e9d73a84ec44b55",
        "moss_tts_local_greedy_frame.onnx" to "4d479d5ce528d1b2656b3f7e389a1672",
        "moss_tts_local_sampled_frame.onnx" to "5905aaf3248907c12336015afde4a12c",
        "moss_tts_prefill.onnx" to "e6a6e83f7da1a787b4abeab7e3f7f1f4",
        "quantization_config.json" to "75a27ac4f10853d5ec231627a3b67393",
        "tokenizer.model" to "13276bc226e31b5cce9fef78d6e4e270",
        "tts_browser_onnx_meta.json" to "d494d9ca9921adc8e6001ae5277c864d",
    )

    fun modelDir(ctx: Context): File = File(ctx.filesDir, "moss_tts")

    /** 全部16文件 md5 验过后才写的完成标记; 无标记=上次下载未完成, 走校验/续传 */
    private fun stampFile(ctx: Context): File = File(modelDir(ctx), ".download_ok")

    fun isReady(ctx: Context): Boolean {
        val dir = modelDir(ctx)
        return stampFile(ctx).exists() && MODEL_FILES.all { File(dir, it.first).length() > 0 }
    }

    /** 已有模型 -> 直接回调; 否则逐文件下载+校验(缺哪个补哪个, 断连自动续传) */
    fun ensureModel(ctx: Context, onStatus: (String) -> Unit, onDone: (Boolean) -> Unit) {
        if (isReady(ctx)) { onStatus("模型已就绪"); onDone(true); return }
        ModelDownloadService.start(ctx) // 前台服务: 熄屏/锁屏/后台下载不被掐网络
        Thread {
            fun status(msg: String) {
                onStatus(msg)
                ModelDownloadService.status(ctx, msg) // 通知同步进度
            }
            val pm = ctx.getSystemService(Context.POWER_SERVICE) as PowerManager
            val wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "com.moss.tts:download")
            val wifiMgr = ctx.applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
            @Suppress("DEPRECATION")
            val wifiLock = wifiMgr.createWifiLock(WifiManager.WIFI_MODE_FULL_HIGH_PERF, "com.moss.tts:download")
            wakeLock.acquire(30 * 60 * 1000L) // 30min 保险上限, 防泄漏
            wifiLock.acquire()
            try {
                val dir = modelDir(ctx)
                dir.mkdirs()
                val total = MODEL_FILES.size
                MODEL_FILES.forEachIndexed { idx, (name, wantMd5) ->
                    val out = File(dir, name)
                    if (out.length() > 0 && md5Of(out) == wantMd5) {
                        Log.d(TAG, "skip(已有且md5对): $name")
                        return@forEachIndexed
                    }
                    status("下载模型 (${idx + 1}/$total) $name")
                    var ok = false
                    var lastErr: Exception? = null
                    for (attempt in 1..3) {
                        try {
                            download("$MODEL_BASE_URL/$name", out)
                            val m = md5Of(out)
                            Log.d(TAG, "$name 尝试$attempt: 下载完 ${out.length()}B md5符合=${m == wantMd5}")
                            if (m == wantMd5) { ok = true; break }
                            lastErr = RuntimeException("md5 不一致: $name")
                            out.delete() // 内容损坏不续传, 下一轮从零
                            Log.d(TAG, "$name md5不符, 删掉从零重下")
                        } catch (e: Exception) {
                            lastErr = e // 断连: 保留部分文件, 下一轮 Range 续传
                            Log.d(TAG, "$name 尝试${attempt}异常: ${e.message} (留半截${out.length()}B)")
                        }
                    }
                    if (!ok) throw lastErr ?: RuntimeException("下载失败: $name")
                    // 失败时保留半截文件: 下次启动按 Range 续传(md5 校验兜底)
                }
                stampFile(ctx).writeText("ok") // 到这里 16 文件全部 md5 验过
                if (isReady(ctx)) {
                    status("模型就绪(${total}文件)")
                    onDone(true)
                } else {
                    status("模型文件不全，请检查下载")
                    onDone(false)
                }
            } catch (e: Exception) {
                status("模型下载失败: ${e.message}。可手动放置模型(见README)")
                onDone(false)
            } finally {
                if (wakeLock.isHeld) wakeLock.release()
                if (wifiLock.isHeld) wifiLock.release()
                ModelDownloadService.stop(ctx)
            }
        }.start()
    }

    /** 下载单文件; 已有部分文件时 Range 续传(206), 服务端忽略 Range(200)则从零覆盖 */
    private fun download(urlStr: String, dst: File) {
        val have = dst.length()
        val conn = URL(urlStr).openConnection() as HttpURLConnection
        conn.instanceFollowRedirects = true
        conn.connectTimeout = 15000
        conn.readTimeout = 60000
        if (have > 0) conn.setRequestProperty("Range", "bytes=$have-")
        conn.connect()
        Log.d(TAG, "GET ${dst.name} Range=${if (have > 0) "bytes=$have-" else "无"} -> HTTP ${conn.responseCode}")
        when (conn.responseCode) {
            200 -> conn.inputStream.use { writeAll(it, dst, append = false) }
            206 -> conn.inputStream.use { writeAll(it, dst, append = true) }
            416 -> Log.d(TAG, "${dst.name} HTTP 416(已到末尾), 交 md5 裁决") // 文件可能已完整, 不删
            else -> throw RuntimeException("HTTP ${conn.responseCode}")
        }
    }

    private fun writeAll(input: InputStream, dst: File, append: Boolean) {
        FileOutputStream(dst, append).use { out ->
            val buf = ByteArray(256 * 1024)
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                out.write(buf, 0, n)
            }
        }
    }

    private fun md5Of(f: File): String {
        val md = MessageDigest.getInstance("MD5")
        f.inputStream().use { input ->
            val buf = ByteArray(256 * 1024)
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }
}
