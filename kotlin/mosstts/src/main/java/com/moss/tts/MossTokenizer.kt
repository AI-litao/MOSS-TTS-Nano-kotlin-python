package com.moss.tts

import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale

/**
 * SentencePiece Tokenizer — 纯Kotlin实现
 * 解析protobuf格式的tokenizer.model
 *
 * Protobuf ModelProto结构:
 *   repeated ModelProto.SentencePiece pieces = 1;
 *   每个piece: string piece=1, float score=2, Type type=3
 */
class MossTokenizer(private val modelPath: String) {
    companion object {
        private const val TAG = "MossTokenizer"
    }

    private val pieces: List<Piece>
    private val pieceToId: Map<String, Int>

    data class Piece(val text: String, val score: Float, val type: Int)

    init {
        pieces = loadModel(modelPath)
        pieceToId = pieces.withIndex().associate { (i, p) -> p.text to i }
        Log.i(TAG, "Loaded ${pieces.size} pieces from $modelPath")
        if (pieces.isNotEmpty()) {
            Log.i(TAG, "First 5 pieces: ${pieces.take(5).map { it.text }}")
        }
    }

    /**
     * SentencePiece BPE 编码 — 与官方 sentencepiece 逐 token 对齐 (2026-10-10 重写)。
     * 旧实现=贪心最长匹配+静默丢字符, 不做归一化/▁前缀, 产出的 token id 全是错的
     * (数字/标点被丢, 词边界件错位 → 模型"跳读")。
     * 真算法: NFKC归一化 → 裁首尾空白 → 空白串→'▁' → 前置'▁' →
     * BPE按merge秩(分数=-rank越大越先合)链表合并 → 未知字符UTF-8字节件兜底。
     */
    fun encode(text: String): List<Int> {
        if (pieces.isEmpty()) return emptyList()
        val norm = normalize(text)
        if (norm.isEmpty()) return emptyList()

        // 初始符号: 逐codepoint; 不在词表 → UTF-8字节 <0xXX>
        val syms = ArrayList<String>()
        norm.codePoints().forEach { cp ->
            val s = String(Character.toChars(cp))
            if (pieceToId.containsKey(s)) {
                syms.add(s)
            } else {
                for (b in s.toByteArray(Charsets.UTF_8)) {
                    val hex = String.format(Locale.US, "<0x%02X>", b.toInt() and 0xFF)
                    if (pieceToId.containsKey(hex)) syms.add(hex)
                }
            }
        }
        val n = syms.size
        if (n == 0) return emptyList()

        // BPE合并(链表): 候选=相邻两符号拼接在词表中的, 分数大者先合
        val prev = IntArray(n) { it - 1 }
        val next = IntArray(n) { if (it + 1 < n) it + 1 else -1 }
        val alive = BooleanArray(n) { true }
        data class Cand(val negScore: Float, val i: Int, val pid: Int, val j: Int)
        val heap = java.util.PriorityQueue<Cand>(compareBy({ it.negScore }, { it.i }))

        fun push(i: Int) {
            if (i < 0 || i >= n || !alive[i]) return
            val j = next[i]
            if (j < 0 || !alive[j]) return
            val pid = pieceToId[syms[i] + syms[j]] ?: return
            heap.add(Cand(-pieces[pid].score, i, pid, j))
        }
        for (i in 0 until n) push(i)

        while (heap.isNotEmpty()) {
            val c = heap.poll()
            if (!alive[c.i] || !alive[c.j] || next[c.i] != c.j) continue
            if (pieceToId[syms[c.i] + syms[c.j]] != c.pid) continue
            syms[c.i] = syms[c.i] + syms[c.j]
            alive[c.j] = false
            next[c.i] = next[c.j]
            if (next[c.j] >= 0) prev[next[c.j]] = c.i
            push(prev[c.i])
            push(c.i)
        }

        val ids = mutableListOf<Int>()
        var i = 0
        while (i >= 0) {
            val pid = pieceToId[syms[i]] ?: return emptyList()
            ids.add(pid)
            i = next[i]
        }
        return ids
    }

    /** SentencePiece nmt_nfkc 归一化等价: NFKC → 裁首尾空白 → 空白串→'▁' → 前置'▁' */
    private fun normalize(text: String): String {
        val nfkc = java.text.Normalizer.normalize(text, java.text.Normalizer.Form.NFKC)
        val trimmed = nfkc.trim { it.isWhitespace() }
        val sb = StringBuilder(trimmed.length + 1)
        sb.append('▁')
        var inWs = false
        for (ch in trimmed) {
            if (ch.isWhitespace()) {
                inWs = true
            } else {
                if (inWs) { sb.append('▁'); inWs = false }
                sb.append(ch)
            }
        }
        return sb.toString()
    }

    fun decode(ids: List<Int>): String {
        val sb = StringBuilder()
        for (id in ids) {
            if (id in pieces.indices) {
                sb.append(pieces[id].text.replace("▁", " "))
            }
        }
        return sb.toString().trim()
    }

    /**
     * 解析protobuf格式的SentencePiece model
     * 简化的protobuf解析器，只处理我们需要的字段
     */
    private fun loadModel(path: String): List<Piece> {
        val file = File(path)
        if (!file.exists()) {
            Log.e(TAG, "Model file not found: $path")
            return emptyList()
        }

        val bytes = file.readBytes()
        val result = mutableListOf<Piece>()

        try {
            var pos = 0
            while (pos < bytes.size) {
                val (tag, newPos) = readVarint(bytes, pos)
                pos = newPos
                val fieldNumber = (tag shr 3).toInt()
                val wireType = (tag and 0x7).toInt()

                if (fieldNumber == 1 && wireType == 2) {
                    // pieces字段 (length-delimited)
                    val (len, afterLen) = readVarint(bytes, pos)
                    pos = afterLen
                    val endPos = pos + len.toInt()

                    var pieceText = ""
                    var pieceScore = 0.0f
                    var pieceType = 1  // NORMAL

                    // 解析SentencePiece子消息
                    while (pos < endPos) {
                        val (subTag, subPos) = readVarint(bytes, pos)
                        pos = subPos
                        val subField = (subTag shr 3).toInt()
                        val subWire = (subTag and 0x7).toInt()

                        when (subField) {
                            1 -> { // piece (string)
                                val (sLen, sPos) = readVarint(bytes, pos)
                                pos = sPos
                                pieceText = String(bytes, pos, sLen.toInt(), Charsets.UTF_8)
                                pos += sLen.toInt()
                            }
                            2 -> { // score (float)
                                pieceScore = ByteBuffer.wrap(bytes, pos, 4)
                                    .order(ByteOrder.LITTLE_ENDIAN).float
                                pos += 4
                            }
                            3 -> { // type (enum)
                                val (tVal, tPos) = readVarint(bytes, pos)
                                pos = tPos
                                pieceType = tVal.toInt()
                            }
                            else -> pos = skipField(bytes, subWire, pos)
                        }
                    }

                    if (pieceText.isNotEmpty()) {
                        result.add(Piece(pieceText, pieceScore, pieceType))
                    }
                } else {
                    pos = skipField(bytes, wireType.toInt(), pos)
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Parsed ${result.size} pieces before error: ${e.message}")
        }

        Log.i(TAG, "Parsed ${result.size} pieces")
        return result
    }

    private fun readVarint(bytes: ByteArray, startPos: Int): Pair<Long, Int> {
        var result = 0L
        var shift = 0
        var pos = startPos
        while (pos < bytes.size) {
            val b = bytes[pos].toInt() and 0xFF
            result = result or ((b and 0x7F).toLong() shl shift)
            pos++
            if (b and 0x80 == 0) break
            shift += 7
        }
        return result to pos
    }

    private fun skipField(bytes: ByteArray, wireType: Int, startPos: Int): Int {
        var pos = startPos
        when (wireType) {
            0 -> { // varint
                while (pos < bytes.size && (bytes[pos].toInt() and 0x80) != 0) pos++
                pos++
            }
            1 -> pos += 8  // 64-bit
            2 -> { // length-delimited
                val (len, newPos) = readVarint(bytes, pos)
                pos = newPos + len.toInt()
            }
            5 -> pos += 4  // 32-bit
        }
        return pos
    }
}
