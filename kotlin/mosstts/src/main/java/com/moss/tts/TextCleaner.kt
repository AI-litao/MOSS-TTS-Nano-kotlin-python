package com.moss.tts

/**
 * TTS文本清理器 — MOSS专用精简版
 * 保留标点: 。 ， ？ ； 、  (5个中文标点, !/！转。)
 * 其他所有符号→逗号或删除，避免MOSS静音/怪音
 */
object TextCleaner {

    /** 判断字符是否为CJK汉字或日文假名（这些语言不需要空格分词） */
    private fun isCjkOrKana(ch: Char): Boolean {
        val code = ch.code
        return (code in 0x4E00..0x9FFF)    // CJK统一汉字
            || (code in 0x3400..0x4DBF)    // CJK扩展A
            || (code in 0x3040..0x309F)    // 平假名
            || (code in 0x30A0..0x30FF)    // 片假名
            || (code in 0x3000..0x303F)    // CJK标点符号
            || (code in 0xFF00..0xFFEF)    // 全角字符
    }

    fun clean(text: String): String {
        var s = text
        android.util.Log.i("TextCleaner", "clean() in: '${text.take(50)}'")

        // === 1. 基础清理 ===
        s = s.replace(Regex("<think>.*?</think>"), "")
        s = s.replace(Regex("[\\u200B-\\u200D\\uFEFF]"), "")
        s = s.replace(Regex("```[\\s\\S]*?```"), "")
        s = s.replace(Regex("`([^`]+)`"), "$1")
        s = s.replace(Regex("\\*\\*([^*]+)\\*\\*"), "$1")
        s = s.replace(Regex("\\*([^*]+)\\*"), "$1")
        s = s.replace(Regex("\\[([^\\]]+)\\]\\([^)]+\\)"), "$1")
        s = s.replace(Regex("https?://\\S+"), "")
        s = s.replace(Regex("<[^>]+>"), "")
        s = s.replace(Regex("[\\u2600-\\u27BF\\u2B50\\u2B55\\u23CF\\u23E9-\\u23F3\\u231A\\u231B\\u25AA-\\u25C0\\u25FB-\\u25FE\\u2648-\\u2653\\u267F\\u2934\\u2935\\u2B05-\\u2B07]+"), "")

        // === 2. 引号全部删除 ===
        s = s.replace(Regex("[\"'\u201C\u201D\u2018\u2019]"), "")

        // === 2.5. v+数字 → 去掉v前缀 (兜底: MOSS读v+数字会跳过) ===
        s = s.replace(Regex("([vV])(\\d)"), "$2")

        // === 3. 数字+点号 (必须在半角.→。之前!) ===
        val beforeDecimal = s
        s = s.replace(Regex("(\\d+)\\.(\\d)"), "$1点$2")
        s = s.replace(Regex("(\\d+)\\."), "$1，")
        if (beforeDecimal != s) {
            android.util.Log.i("TextCleaner", "decimal: '${beforeDecimal.take(30)}' → '${s.take(30)}'")
        }

        // === 4. 括号全部删除 ===
        s = s.replace(Regex("[()\\[\\]{}（）【】「」『』《》]"), "")

        // === 5. 长符号 → 句号 ===
        s = s.replace(Regex("[—–―]{2,}"), "。")
        s = s.replace(Regex("-{3,}"), "。")
        s = s.replace(Regex("…{1,}|\\.{3,}"), "。")

        // === 6. 箭头 → 逗号 ===
        s = s.replace(Regex("[→←⇒⇐⟶⟼⟷⟹⟺↦↪↩]|<-+|-+>|<=+|=+>"), "，")

        // === 7. 半角→中文标点 (!→。 因为！会导致MOSS乱音) ===
        s = s.replace(",", "，")
        s = s.replace("!", "。")   // ! → 。 (不是！)
        s = s.replace("?", "？")
        s = s.replace(";", "；")
        s = s.replace(":", "：")
        s = s.replace(".", "。")

        // === 7.4. 数学符号（上下文感知：数字间的才转） ===
        // 减号：数字-数字 → 减，其他情况后面step8转逗号
        s = s.replace(Regex("(\\d)\\s*-\\s*(\\d)"), "$1减$2")
        // 乘号：数字*数字 → 乘（step1-2已清理markdown **bold**，残留的*才走这里）
        s = s.replace(Regex("(\\d)\\s*\\*\\s*(\\d)"), "$1乘$2")
        // 除号
        s = s.replace("÷", "除")
        s = s.replace("×", "乘")

        // === 7.5. 常见符号→文字 (MOSS读不出来就乱音) ===
        s = s.replace("=", "等于")
        s = s.replace("+", "加")
        s = s.replace("/", "或")
        s = s.replace("%", "百分之")
        s = s.replace("$", "")
        s = s.replace("€", "")
        s = s.replace("£", "")
        s = s.replace("¥", "")
        s = s.replace("℃", "度")
        s = s.replace("℉", "度")
        s = s.replace("°", "度")
        s = s.replace("#", "")
        s = s.replace("@", "")
        s = s.replace("&", "和")
        s = s.replace("~", "")
        s = s.replace("|", "，")
        s = s.replace("\\", "")
        s = s.replace("^", "")
        s = s.replace("_", "")
        s = s.replace("`", "")

        // === 8. 其他符号→逗号 (只保留SAFE标点+字母数字+空白) ===
        val sb = StringBuilder()
        for (ch in s) {
            val code = ch.code
            when {
                code == 0x3002 -> sb.append(ch)  // 。
                code == 0xFF0C -> sb.append(ch)  // ，
                code == 0xFF01 -> sb.append('。') // ！ → 。 (MOSS乱音)
                code == 0xFF1F -> sb.append(ch)  // ？
                code == 0xFF1B -> sb.append(ch)  // ；
                code == 0x3001 -> sb.append(ch)  // 、
                ch.isLetterOrDigit() -> sb.append(ch)
                ch.isWhitespace() -> sb.append(ch)
                else -> sb.append('，')
            }
        }
        s = sb.toString()

        // === 9. 连续标点收敛 ===
        s = s.replace(Regex("。{2,}"), "。")
        s = s.replace(Regex("，{2,}"), "，")
        s = s.replace(Regex("？{2,}"), "？")

        // === 10. 换行→句号 ===
        s = s.replace("\n", "。")

        // === 11. 智能空格处理 (CJK/假名间删空格, 英文/韩文/数字间保留) ===
        val sb2 = StringBuilder()
        for (i in s.indices) {
            val ch = s[i]
            if (ch.isWhitespace()) {
                val prev = if (i > 0) s[i - 1] else null
                val next = if (i + 1 < s.length) s[i + 1] else null
                val prevIsCjk = prev != null && isCjkOrKana(prev)
                val nextIsCjk = next != null && isCjkOrKana(next)
                if (!prevIsCjk && !nextIsCjk) {
                    // 两边都不是CJK: 英文/韩文/数字之间的空格, 保留
                    sb2.append(ch)
                }
                // 其他情况(CJK-CJK, CJK-非CJK, 非CJK-CJK): 删除空格
            } else {
                sb2.append(ch)
            }
        }
        s = sb2.toString()

        // === 11.5. Fix D(2026-10-10): 首尾孤悬"，"修剪 ===
        // emoji/符号逐char转"，"后残留的纯标点尾巴, 如"好😊。"→"好，。"→"好。"
        s = s.replace("，。", "。").replace("，？", "？").replace("，；", "；")
        s = s.trim().trimStart('，', '、', '；').trimEnd('，', '、', '；').trim()

        // === 12. 句末补标点 ===
        s = s.trim()
        if (s.isNotEmpty() && s.last() != '。' && s.last() != '？'
            && s.last() != '；' && s.last() != '，' && s.last() != '、') {
            s += "。"
        }

        // === 13. Fix D(2026-10-10): 实质内容检查 ===
        // 纯标点/emoji句(如"😊"→"，")返回空串, 上游isBlank()自然跳过, 不进MOSS防runaway怪音
        if (s.none { it.isLetterOrDigit() }) {
            android.util.Log.i("TextCleaner", "clean() 无实质内容, 丢弃: '${text.take(30)}'")
            return ""
        }

        android.util.Log.i("TextCleaner", "clean() out: '${s.take(60)}'")
        return s
    }

    /** 拆分TTS短句 */
    fun splitSentences(text: String, maxLen: Int = 20): List<String> {
        val parts = text.split(Regex("([。？；\\n])"))
        val raw = mutableListOf<String>()
        var i = 0
        while (i < parts.size) {
            val seg = parts[i]
            val punct = if (i + 1 < parts.size && parts[i + 1].matches(Regex("[。？；\\n]"))) {
                i++; parts[i]
            } else ""
            i++
            val combined = (seg + punct).trim()
            if (combined.isNotBlank()) raw.add(combined)
        }
        val result = mutableListOf<String>()
        for (seg in raw) {
            if (seg.length <= maxLen) {
                if (seg.length > 1) result.add(seg)
            } else {
                val parts = seg.split(Regex("[，,\\s]"))
                val buf = StringBuilder()
                for (p in parts) {
                    if (buf.length + p.length > maxLen && buf.isNotEmpty()) {
                        result.add(buf.toString().trim())
                        buf.clear()
                    }
                    buf.append(p)
                }
                if (buf.isNotEmpty() && buf.length > 1) result.add(buf.toString().trim())
            }
        }
        return result
    }

    /** 根据停顿时长加标点 */
    fun addPunctuation(text: String, silenceMs: Int): String {
        val t = text.trim()
        if (t.isEmpty()) return t
        if (t.last() in "。？，；：、") return t
        return when {
            silenceMs >= 1500 -> "$t。"
            silenceMs >= 500 -> "$t，"
            else -> "$t。"
        }
    }

    /** 线性插值重采样 */
    fun resample(pcm: ShortArray, origSr: Int, targetSr: Int): ShortArray {
        if (origSr == targetSr) return pcm
        val ratio = origSr.toDouble() / targetSr
        val targetLen = (pcm.size / ratio).toInt()
        return ShortArray(targetLen) { i ->
            val srcPos = i * ratio
            val idx = srcPos.toInt().coerceIn(0, pcm.size - 2)
            val frac = (srcPos - idx).toFloat()
            val v = pcm[idx] * (1 - frac) + pcm[idx + 1] * frac
            v.coerceIn(-32768f, 32767f).toInt().toShort()
        }
    }
}
