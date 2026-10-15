"""WSOLA 时域压伸 — 语速加快但音高/音色不变（不做重采样变速=不升调）。

ratio = 速度倍数（1.3 = 快 30%，输出时长 ≈ 输入/1.3）。ratio <= 1.01 原样返回。
算法: 波形相似度重叠相加 — 帧长 2048(@48k≈42.7ms), 分析hop=1024, 合成hop=1024/ratio,
每帧在 ±256 样本内搜与"上一帧自然后继"互相关最大的对齐位置, Hann 窗 OLA + 窗和归一化。
双端一致性: kotlin/mosstts/src/main/java/com/moss/tts/Wsola.kt 同参数同算法。
"""
import numpy as np

N = 2048       # 帧长
HA = 1024      # 分析 hop
TOL = 256      # 对齐搜索范围 ±256 样本


def wsola_stretch(pcm: np.ndarray, ratio: float) -> np.ndarray:
    """pcm: int16 单声道。返回 int16 单声道, 时长≈pcm.size/ratio。"""
    if ratio <= 1.01 or pcm.size < N * 2:
        return pcm
    hs = int(round(HA / ratio))
    hs = max(128, min(HA, hs))
    x = pcm.astype(np.float32) / 32768.0
    win = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N) / (N - 1))).astype(np.float32)

    out_len = pcm.size // HA * hs + N + hs
    out = np.zeros(out_len, np.float32)
    wsum = np.zeros(out_len, np.float32)

    prev_pos = 0
    k = 0
    while True:
        ideal = k * HA
        if ideal + N + TOL >= x.size:
            break
        o = k * hs
        if o + N > out_len:
            break
        pos = ideal
        if k > 0:
            # 自然延续对齐: 新帧头 L 段与"上一帧按自然hop的后继"互相关最大化
            L = N - hs
            target = min(prev_pos + HA, x.size - L)
            seg = x[target: target + L]
            best = -1e30
            for d in range(-TOL, TOL + 1):
                c = ideal + d
                if c < 0 or c + N > x.size:
                    continue
                corr = float(np.dot(seg, x[c: c + L]))
                if corr > best:
                    best = corr
                    pos = c
        out[o: o + N] += x[pos: pos + N] * win
        wsum[o: o + N] += win
        prev_pos = pos
        k += 1

    # 归一化 + 裁到目标长度
    target_len = int(round(pcm.size / ratio))
    np.maximum(wsum, 1e-3, out=wsum)
    y = out / wsum
    y = y[:target_len] if target_len <= y.size else np.pad(y, (0, target_len - y.size))
    # RMS增益补偿: OLA拼接相位抵消会掉能量, 按输入/输出RMS比补回(限1.0~1.5)
    out_rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
    in_rms = float(np.sqrt(np.mean((pcm.astype(np.float32) / 32768.0) ** 2))) if pcm.size else 0.0
    gain = min(1.5, max(1.0, in_rms / out_rms)) if out_rms > 1e-6 else 1.0
    y = y * gain
    return np.clip(np.round(y * 32767.0), -32768, 32767).astype(np.int16)
