# MOSS-TTS-Nano-kotlin-python

MOSS-TTS-Nano 100M 离线语音合成 — 双端精简版
**Kotlin (Android)** + **Python (笔记本/台式 CPU)**，共享一套模型。

- 纯离线、18 内置音色（推荐 12 个多语种）、中/英/日/韩
- 基于 OpenMOSS 开源 ONNX CPU 版精简：**已取消语音克隆**（见下文）
- 融入 2026-10 全部工程优化（文本管线/防跑飞/线程安全/确定性/停顿压缩，见"优化清单"）
- 模型包（Apache-2.0，16 文件 204MB）：**[魔搭 MOSS-TTS-Nano-int8](https://modelscope.cn/models/ArchitectLi/MOSS-TTS-Nano-int8)**

---

## 目录结构

```
MOSS-TTS-Nano-kotlin-python/
├── README.md                 本文件
├── docs/                     许可合规 (LICENSE/NOTICE, Apache-2.0 血统)
├── kotlin/                   Android 端 (Gradle 工程)
│   ├── mosstts/              TTS 库模块 (AAR) — 可组装到其他项目
│   │   └── src/main/jniLibs/arm64-v8a/libonnxruntime.so   微软官方 ORT (MIT, API 27 与头文件同代)
│   └── demo/                 最小 demo app (12音色UI + 首启下载模型 + 存 Download/MOSS-TTS)
└── python/                   CPU 端
    ├── run.sh                一键: 装依赖→下模型→起 Web UI (8080)
    ├── infer.py              推理入口 (命令行 + 可 import 的模块 API)
    ├── webui.py              极简 Web UI (纯标准库, 零额外依赖)
    ├── download_model.py     模型下载/检测
    ├── requirements.txt      onnxruntime + numpy + sentencepiece (无 torch!)
    ├── moss_runtime/         推理运行时(官方精简版 + TextCleaner 移植)
    └── static/               Web UI 页面
```

模型不随源码分发：`model/`（Python 端）与手机端模型目录由
`download_model.py` / demo 首启自动下载，见下文「模型」。
生成的 wav 统一落 `output/`（工程根，运行时自动创建）。

---

## 快速开始 — CPU 端 (Python)

### 一键启动（推荐）

```bash
bash python/run.sh
```

run.sh 自动完成：①首次创建 venv 并安装依赖（已装则跳过）
→ ②检测模型，缺失自动从魔搭逐文件下载（16 个文件 204MB，逐个 md5 校验）
→ ③启动 Web UI：浏览器打开 **http://localhost:8080**
（端口可用 `PORT=9000 bash python/run.sh` 改）

Web UI = 文本框 + 12 音色下拉 + 语速(停顿压缩) + 压伸(WSOLA, 音高不变) + 生成（保存 `output/` 并在线试听）。

### 手动命令行

```bash
cd python
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # 依赖
.venv/bin/python download_model.py                                   # 模型(幂等)
.venv/bin/python infer.py --text "今天天气真好" --voice 4 --out ../output/a.wav
# 注意: --out 相对路径按"当前目录"解析; 不传 --out 时默认落工程根 output/
# (从 python/ 目录跑, 写工程根 output/ 用 ../output/xxx.wav)
.venv/bin/python webui.py --port 8080                                # 或手动起 Web UI
```

### infer.py 参数

| 参数 | 说明 |
|------|------|
| `--text` / `--file` | 文本或文本文件 |
| `--voice` | 音色 id 0~17 或名字（默认 4 悦文） |
| `--speed` | 停顿压缩 1.0~3.0（默认 1.0；只剪静音，不升调） |
| `--stretch` | WSOLA 整句压伸 1.0~1.5（默认 1.0 关闭；语速加快但**音高/音色不变**） |
| `--sample-mode` | `full`(默认) / `fixed` / `greedy` |
| `--seed` | 采样种子，默认 1234（与 Android 端一致的确定性输出），`-1`=随机 |
| `--threads` | 线程数（默认 4） |
| `--out` | 输出 wav 路径（默认 `output/output.wav`（工程根）） |
| `--list-voices` | 列出音色 |

---

## 快速开始 — Android 端

### 构建 demo APK

```bash
cd kotlin
gradle :demo:assembleDebug        # 产物: demo/build/outputs/apk/debug/
# 或只构建库: gradle :mosstts:assembleRelease → mosstts-release.aar
```

1. 安装构建出的 demo APK
2. 首次启动**自动下载模型**（16 个文件约 204MB，Wi-Fi 下载，逐文件进度显示在状态栏与下载通知；断点续传，中断下次自动续上，下载期屏幕常亮 + 前台服务，熄屏/锁屏/切后台不中断）
3. 输入文本 → 选音色（12 个）→ 生成 → 自动保存到公共 **`Download/MOSS-TTS/`**
   （文件管理/系统播放器直接可见）并播放；下方「重播 / 分享 / 删除」三个按钮管理最近一次产物

手动放置模型（可选，免下载）：把模型 16 个文件放到应用 `files/moss_tts/` 下
（平铺；引擎也兼容 `MOSS-TTS-Nano-100M-ONNX/` 子目录布局）。

---

## 模型

- **下载**：[魔搭 MOSS-TTS-Nano-int8](https://modelscope.cn/models/ArchitectLi/MOSS-TTS-Nano-int8)
  （16 个文件**平铺**，共 204MB，逐文件直链 + md5 校验）
  直链前缀常量：`python/download_model.py` 与 `kotlin/demo/.../ModelManager.kt` 顶部

### 文件清单（16 个）

| 文件 | 大小 | 用途 |
|------|------|------|
| global_shared.data | 110.5MB | TTS 权重(int8 MatMul + u8 embedding) |
| local_shared.data | 85.8MB | TTS 权重(local decoder) |
| moss_audio_tokenizer_decode_full_int8.onnx | 12.2MB | codec 解码(整帧, 自量化 int8) |
| moss_tts_prefill / decode_step / local_decoder / local_*_frame ×3 | 0.1~0.8MB | TTS 推理图 |
| tokenizer.model | 0.5MB | SentencePiece 分词 |
| browser_poc_manifest.json | 0.5MB | 配置清单(18 音色 prompt codes) |
| tts_browser_onnx_meta.json / codec_browser_onnx_meta.json / config.json / quantization_config.json | <0.1MB | 元数据 |

### 血统与许可（详见 docs/NOTICE-models.md）

**全部 16 个文件均为 Apache-2.0 授权作品或其衍生作品，依原许可分发。**

- 上游底模：[OpenMOSS-Team/MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano)（Apache-2.0）
- TTS 权重 = linlinkj / REALBITS 公开 int8 量化版（与 REALBITS 发布版逐字节同源，Apache-2.0）
- codec = OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano 的 FP32 解码器（Apache-2.0）
  → 本项目用 ONNX Runtime `quantize_dynamic` 自量化 int8（结构不变，仅权重压缩）

---

## 音色列表（18 个内置，推荐 12 个）

> 2026-09 音色质量筛选（18 音色 × 中英日韩长文本听感测试）：
> 已剔除 6 个（仅 UI 不暴露，prompt codes 仍在模型中）：
> v3 小雨（英文长静音+放音乐）、v10 Nathan（日韩弱+最慢）、v11 Soyo（全语言弱+音量仅1/3）、
> v13 Mortis（英日差）、v16 Anon（中韩弱+慢）、v17 Arisa（中日弱+英文过慢）。
> 限制标注：v5 灵玉（中文弱+日文乱音）、v15 Mei（中文略弱）。
> 最快最稳：v14 Umiri、v4 悦文（默认）。

**推荐 12 个：中文 0/1/2/4/5，English 6/7/8/9，日本語 12/14/15**

| id | 名称 | 语种/性别 | | id | 名称 | 语种/性别 |
|----|------|-----------|--|----|------|-----------|
| 0 | Junhao | 中/男 | | 9 | Adam | 英/男 |
| 1 | Zhiming | 中/男 | | 12 | Saki | 日/女 |
| 2 | Weiguo | 中/男 | | 14 | Umiri | 日/女 |
| 4 | Yuewen | 中/女(默认) | | 15 | Mei | 日/女 |
| 5 | Lingyu | 中/女 | | | | |

其余 6 个（3 Xiaoyu、6→10 Nathan、11 Soyo、13 Mortis、16 Anon、17 Arisa）质量弱，未在 UI 暴露。

---

## 优化清单（相比官方 ONNX CPU 版）

全部为 2026-09/10 真机实测验证过的工程优化：

1. **TextCleaner 文本管线**（`moss_runtime/text_cleaner.py` / `TextCleaner.kt`，双端同规则）
   - emoji/符号：逐字符转逗号或删除；markdown/URL/HTML 清理
   - 标点过滤：只保留 `。 ， ？ ； 、`，`!/！`→`。`（！会导致 MOSS 乱音，实测 400 帧垃圾音频）
   - 语种分开：CJK/假名之间删空格，英文/韩文/数字之间保留（不破坏英文 tokenization）
   - 小数点保护：`3.14`→`3点14`；数字间数学符号转文字（`=`→等于、`%`→百分之）
   - 首尾孤悬"，"修剪；**无实质内容（纯标点/emoji）直接丢弃** — 防 runaway 怪音
     （实测：单独"😊"/"，" 曾生成 9~21 秒爆响垃圾音频，现已归零）
2. **退化输入防跑飞**：帧数封顶 `estimateMaxFrames`（≤5 token→40 帧 / ≤15→80 / ≤40→200 / else 400）
   — 模型不发停止信号时最多 3.5 秒，不再是 30 秒
3. **线程安全**（Fix A）：`generate()` 上下文全程局部化 + 互斥锁
   — 消灭"预热与对话并发互删上下文 → 句子无声蒸发"的并发丢句 bug
4. **有界预热**（Fix B）：`prewarm()` 引擎级 maxFrames=2，杜绝预热文本生成超长垃圾音频
5. **确定性输出**（Fix E）：RNG 固定种子 1234（原时钟种子 → 每次生成都不一样）
   — 同文本同种子逐字节可复现，A/B 对比不再靠运气
6. **停顿压缩语速**（Fix F，`compress_gaps`）：只剪静音段，发音本体零改动，**不用重采样变速（不升调）**
   - 公式：每段静音保留 `clamp(min(段长,800ms)/speed^5, 15ms, 段长)`（speed 上限 3.0）
   - speed=1.0 即生效：>800ms 的怪静音截到 800ms（治日韩 1~2 秒怪静音）
   - 实测：中文 -10%、日韩 -16%；speed 1.0 vs 1.3 发音段逐样本 100% 一致
   - 注意：speed≥2.0 后停顿已压到 15ms 下限，再往上调收益趋零——整句加速用下面的 WSOLA
7. **WSOLA 整句压伸**（`--stretch` / `setStretch()` / WebUI"压伸"）：语速加快但
   **音高/音色不变**（区别于重采样丢样本=升调）。1.0=关闭，1.3=快 30%，上限 1.5。
   - 算法：Hann 窗 OLA + 自然延续互相关对齐 + 窗和归一化 + RMS 增益补偿（实测时长比精确、
     f0 偏差 <6%（测量噪声内）、RMS 与原音一致、零削顶）
   - 与停顿压缩组合：`--speed 3.0 --stretch 1.3` 实测总加速 1.57×
8. **采样率修正**（2026-10-10）：输出 48000Hz（codec_meta 官方 sample_rate，
   3840 样本/帧=80ms）。旧版 44100 硬编码=错误标注，所有音频慢放 8.8%、低 8.8% 音高，
   已修正
9. **切句**：软切句 ≥5 字、硬切句按 `。！？；` + 英文 `. `（小数点保护）
10. **codec 喂食 dtype 自适应**：JNI 层自动 int64/int32 转置+cast（兼容新旧 decoder 图）
11. **runtime 精简**：删 torch/torchaudio/wetext/自动下载（requirements 3 个包，无 torch）
12. **模型单文件夹平铺**：16 文件一个目录，双端同一布局

---

## 已取消：语音克隆

官方 ONNX CPU 版的克隆能力（参考音频→提取音色）在本精简版**已删除**：

- **原因**：克隆编码器为 int8 量化，精度不足，克隆输出远劣于内置音色
  （2026-09 实测结论）
- **影响**：仅支持上表 18 个内置音色（prompt codes 已固化在 manifest 中）；
  相关代码/依赖（torchaudio、克隆编码器、流式解码图）已一并删除
- 如需克隆：请用官方 OpenMOSS 仓库的完整版（PyTorch 路径）

---

## 模块组装 — CPU 端

`python/infer.py` 的 `MossTts` 类可直接嵌入其他 Python 项目：

```python
import sys
sys.path.insert(0, "path/to/MOSS-TTS-Nano-kotlin-python/python")
from infer import MossTts

tts = MossTts(
    model_dir="path/to/model/MOSS-TTS-Nano-100M-ONNX",  # 16文件平铺目录
    voice=4,          # 或 "Yuewen"
    speed=1.0,        # 停顿压缩 0.5~3.0
    stretch=1.0,      # WSOLA压伸 1.0~1.5(音高不变), 1.0=关闭
    sample_mode="full",
    seed=1234,        # None=随机
)
wav_path = tts.synthesize("今天天气真好，我们出去玩吧。", out_path="out.wav")
```

底层接口（分句/清理/帧级控制）在 `moss_runtime/`，需要自定义流水线时可直接用
`OnnxTtsRuntime`（参考 `infer.py` 实现，~40 行）。

---

## 模块组装 — Android 端

### 方式 A：源码模块（推荐）

1. 把 `kotlin/mosstts/` 拷入宿主工程，`settings.gradle.kts` 加 `include(":mosstts")`
2. app 模块加依赖：`implementation(project(":mosstts"))`
3. 宿主 `minSdk ≥ 27`，`abiFilters` 含 `arm64-v8a`（so 只打 arm64）
4. 模型放宿主 `filesDir/moss_tts/`（16 文件平铺）

```kotlin
val engine = MossTtsEngine()
engine.load("${filesDir}/moss_tts", voiceId = 4, speed = 1.0f)   // 挂起函数
engine.setStretch(1.3f)                                          // WSOLA压伸(可选, 音高不变)
val result: TtsResult? = engine.generate("今天天气真好。")         // 16bit PCM + sampleRate(48000)
engine.unload()
```

### 方式 B：AAR

```bash
cd kotlin && gradle :mosstts:assembleRelease
# 产物: mosstts/build/outputs/aar/mosstts-release.aar (内含 libonnxruntime.so + libort_jni.so)
```

宿主 `libs/` 放 AAR + `implementation(files("libs/mosstts-release.aar"))`。

**注意**：若宿主工程已带 `libonnxruntime.so`（如 sherpa-onnx），打包会重复：

```kotlin
android { packaging { jniLibs { pickFirsts += "**/libonnxruntime.so" } } }
```

（本库用 C API + dlopen，与 sherpa-onnx 的 ORT 同 soname，可共存取其一。）

### adb 批量测试（TestTtsActivity）

```bash
adb push test.txt /data/local/tmp/tt.txt        # WSL 用 D:\ 路径 push
adb shell am force-stop com.moss.tts
adb shell "am start -n com.moss.tts/.TestTtsActivity --es file /data/local/tmp/tt.txt \
  --es tag t0 --ei voice 4 --ez play false --ez raw false --ef speed 1.0 --ef stretch 1.3"
# 完成标志 logcat: DONE; 产物: files/tts_tests/<tag>.wav
```

---

## 输出

- CPU：`output/`（工程根，wav）
- Android：`Download/MOSS-TTS/`（公共下载目录，wav）

---

## 性能参考（真机实测）

- 真我 GT5 Pro（8Gen3）：RTF 0.26~1.06（CPU 2 线程，帧生成 ~23ms/帧）
- 逐帧自回归：线程数只影响速度不影响音质，无需调大
- codec 解码 ~4.2ms/帧；100M 模型加载 2~4 秒

## 已知限制

- 100M 小模型：日韩语种能力弱（韩文无专属音色），长句偶发 1~2 秒怪静音（speed=1.0 已自动截到 800ms）
- 生成带随机采样（text_temperature=0.8）：同文本不同 seed 音色表现有波动，固定 seed 可复现
- 原始发音语速固定（~0.5s/字），`--speed` 只压缩停顿，不能改变发音本体速度
