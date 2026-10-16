package com.moss.tts.demo

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.IBinder

/**
 * 模型下载前台服务: 下载期把 App 提到前台态, 熄屏/锁屏/后台不被 OEM 掐网络
 * (实测 realme/ColorOS 在 App 不可见时按 UID 断网, WakeLock+WifiLock 无效)
 * 通知文案随 ModelManager 进度更新, 下载结束(成功/失败)自动停止
 */
class ModelDownloadService : Service() {

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "模型下载", NotificationManager.IMPORTANCE_LOW)
        )
        startForeground(NOTIF_ID, build("模型下载中…"))
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        intent?.getStringExtra(EXTRA_STATUS)?.let {
            val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            nm.notify(NOTIF_ID, build(it))
        }
        return START_NOT_STICKY
    }

    private fun build(text: String): Notification =
        Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle("MOSS TTS Nano")
            .setContentText(text)
            .setOngoing(true)
            .build()

    companion object {
        private const val CHANNEL_ID = "model_download"
        private const val NOTIF_ID = 1
        private const val EXTRA_STATUS = "status"

        fun start(ctx: Context) {
            ctx.startForegroundService(Intent(ctx, ModelDownloadService::class.java))
        }

        /** 更新通知文案(服务已前台, 走 onStartCommand) */
        fun status(ctx: Context, text: String) {
            ctx.startService(Intent(ctx, ModelDownloadService::class.java).putExtra(EXTRA_STATUS, text))
        }

        fun stop(ctx: Context) {
            ctx.stopService(Intent(ctx, ModelDownloadService::class.java))
        }
    }
}
