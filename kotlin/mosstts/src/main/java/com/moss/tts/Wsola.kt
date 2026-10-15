package com.moss.tts

/**
 * WSOLA 时域压伸 — 语速加快但音高/音色不变（区别于重采样丢样本=升调，用户否决）。
 * ratio = 速度倍数（1.3 = 快 30%，输出时长 ≈ 输入/1.3）。ratio <= 1.01 原样返回。
 *
 * 算法: 波形相似度重叠相加（WSOLA）
 *   - 帧长 N=2048(@48k≈42.7ms), 分析hop=1024, 合成hop=1024/ratio
 *   - 每帧在 ±256 样本内搜与"上一帧自然后继"互相关最大的对齐位置（保持波形连续）
 *   - Hann 窗重叠相加 + 窗和归一化（消 OLA 幅度起伏/边缘效应）
 * 双端一致性: python/moss_runtime/audio_ops.py 同参数同算法。
 */
object Wsola {
    private const val N = 2048          // 帧长
    private const val HA = 1024         // 分析 hop
    private const val TOL = 256         // 对齐搜索范围 ±256 样本

    fun stretch(pcm: ShortArray, ratio: Float): ShortArray {
        if (ratio <= 1.01f || pcm.size < N * 2) return pcm
        val hs = Math.round(HA / ratio).toInt().coerceIn(128, HA)
        val x = FloatArray(pcm.size) { pcm[it] / 32768f }
        val win = FloatArray(N) { (0.5 - 0.5 * Math.cos(2.0 * Math.PI * it / (N - 1))).toFloat() }
        val outLen = pcm.size / HA * hs + N + hs
        val out = FloatArray(outLen)
        val wsum = FloatArray(outLen)

        var prevPos = 0
        var k = 0
        while (true) {
            val ideal = k * HA
            if (ideal + N + TOL >= x.size) break
            val o = k * hs
            if (o + N > outLen) break
            var pos = ideal
            if (k > 0) {
                // 自然延续对齐: 新帧头 L 段应与"上一帧按自然hop的后继"最相似
                val L = N - hs
                val target = (prevPos + HA).coerceAtMost(x.size - L)
                var best = Double.NEGATIVE_INFINITY
                for (d in -TOL..TOL) {
                    val c = ideal + d
                    if (c < 0 || c + N > x.size) continue
                    var corr = 0.0
                    for (n in 0 until L) corr += x[target + n].toDouble() * x[c + n]
                    if (corr > best) { best = corr; pos = c }
                }
            }
            for (n in 0 until N) {
                out[o + n] += x[pos + n] * win[n]
                wsum[o + n] += win[n]
            }
            prevPos = pos
            k++
        }
        // 归一化 + 裁到目标长度
        val targetLen = Math.round(pcm.size / ratio).toInt()
        val res = ShortArray(targetLen)
        // RMS增益补偿: OLA拼接相位抵消会掉能量, 按输入/输出RMS比补回(限1.0~1.5)
        var sumSq = 0.0
        for (i in 0 until targetLen) {
            val v = if (i < out.size && wsum[i] > 1e-3f) out[i] / wsum[i] else 0f
            sumSq += v * v
        }
        val outRms = Math.sqrt(sumSq / targetLen.coerceAtLeast(1))
        var inSq = 0.0
        for (s in pcm) { val v = s / 32768f; inSq += v * v }
        val inRms = Math.sqrt(inSq / pcm.size.coerceAtLeast(1))
        val gain = if (outRms > 1e-6) (inRms / outRms).coerceIn(1.0, 1.5) else 1.0
        for (i in 0 until targetLen) {
            val v = (if (i < out.size && wsum[i] > 1e-3f) out[i] / wsum[i] else 0f) * gain
            val s = Math.round(v * 32767.0).toInt()
            res[i] = Math.max(-32768, Math.min(32767, s)).toShort()
        }
        return res
    }
}
