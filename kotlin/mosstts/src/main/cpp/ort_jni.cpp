#include <jni.h>
#include <android/log.h>
#include <dlfcn.h>
#include <string>
#include <vector>
#include <memory>
#include <set>
#include <map>
#include <set>
#include <mutex>
#include <cstring>

#define ORT_API_MANUAL_INIT
// MOSS TTS 上下文 — C++管理past cache避免JNI边界传大量数据
#include "ort/onnxruntime_cxx_api.h"


#define TAG "OrtNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#include <condition_variable>
#include <chrono>

static Ort::Env* g_env = nullptr;
static bool g_ort_initialized = false;

struct SessionHandle {
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
};

static bool initOrt() {
    if (g_ort_initialized) return true;
    void* handle = dlopen("libonnxruntime.so", RTLD_NOW);
    if (!handle) { LOGE("dlopen failed: %s", dlerror()); return false; }

    auto getBase = (const OrtApiBase*(*)())dlsym(handle, "OrtGetApiBase");
    if (!getBase) { LOGE("dlsym OrtGetApiBase failed"); return false; }

    const OrtApiBase* base = getBase();
    const OrtApi* api = base->GetApi(ORT_API_VERSION);
    if (!api) { LOGE("GetApi failed"); return false; }

    Ort::InitApi(api);
    g_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "copenai");
    g_ort_initialized = true;
    LOGI("ORT init OK, ver: %s", base->GetVersionString());
    return true;
}

static std::vector<std::string> jarrayToVec(JNIEnv* env, jobjectArray arr) {
    int n = env->GetArrayLength(arr);
    std::vector<std::string> out;
    out.reserve(n);
    for (int i = 0; i < n; i++) {
        auto js = (jstring)env->GetObjectArrayElement(arr, i);
        out.push_back(env->GetStringUTFChars(js, nullptr));
        env->DeleteLocalRef(js);
    }
    return out;
}

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_moss_tts_OrtNative_nativeCreateSession(
    JNIEnv* env, jobject, jstring modelPath, jint numThreads, jboolean forceCpu) {
    if (!initOrt()) return 0;
    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    try {
        auto* h = new SessionHandle();
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // CPU only — NNAPI已删除(2026-10-09): 8Gen3实测NNAPI全部fallback到nnapi-reference
        // (CPU参考实现), 比ORT CPU慢5-10倍(负优化)。真NPU走QNN HTP,不走NNAPI。
        // forceCpu参数保留仅作接口兼容,不再使用。
        (void)forceCpu;
        opts.SetIntraOpNumThreads(numThreads);

        h->session = std::make_unique<Ort::Session>(*g_env, path, opts);
        env->ReleaseStringUTFChars(modelPath, path);
        LOGI("Session created: %s -> %p (provider=CPU)", path, h);
        return reinterpret_cast<jlong>(h);
    } catch (const Ort::Exception& e) {
        LOGE("CreateSession failed: %s", e.what());
        env->ReleaseStringUTFChars(modelPath, path);
        return 0;
    }
}

JNIEXPORT jobjectArray JNICALL
Java_com_moss_tts_OrtNative_nativeRun(
    JNIEnv* env, jobject, jlong handle,
    jobjectArray jInputNames, jobjectArray jInputShapes, jobjectArray jInputDatas,
    jobjectArray jOutputNames) {
    if (!handle) return nullptr;
    auto* h = reinterpret_cast<SessionHandle*>(handle);

    try {
        auto nameStrs = jarrayToVec(env, jInputNames);
        int nIn = (int)nameStrs.size();
        std::vector<const char*> namePtrs;
        for (auto& s : nameStrs) namePtrs.push_back(s.c_str());

        // 构建输入tensor — 保持JNI数组pin住直到Run完成
        std::vector<Ort::Value> tensors;
        std::vector<jfloatArray> jDataRefs;
        std::vector<float*> dataPtrs;
        for (int i = 0; i < nIn; i++) {
            jintArray jShape = (jintArray)env->GetObjectArrayElement(jInputShapes, i);
            jint* sh = env->GetIntArrayElements(jShape, nullptr);
            int shLen = env->GetArrayLength(jShape);
            std::vector<int64_t> shape(sh, sh + shLen);
            env->ReleaseIntArrayElements(jShape, sh, JNI_ABORT);
            env->DeleteLocalRef(jShape);

            jfloatArray jData = (jfloatArray)env->GetObjectArrayElement(jInputDatas, i);
            jDataRefs.push_back(jData);
            float* data = env->GetFloatArrayElements(jData, nullptr);
            dataPtrs.push_back(data);
            size_t len = env->GetArrayLength(jData);

            auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            tensors.push_back(Ort::Value::CreateTensor<float>(mem, data, len, shape.data(), shape.size()));
        }

        // 输出名
        auto outNameStrs = jarrayToVec(env, jOutputNames);
        int nOut = (int)outNameStrs.size();
        std::vector<const char*> outPtrs;
        for (auto& s : outNameStrs) outPtrs.push_back(s.c_str());

        // Run
        auto outputs = h->session->Run(Ort::RunOptions{nullptr},
            namePtrs.data(), tensors.data(), nIn,
            outPtrs.data(), nOut);

        // 释放输入JNI数组
        for (int i = 0; i < nIn; i++) {
            env->ReleaseFloatArrayElements(jDataRefs[i], dataPtrs[i], JNI_ABORT);
            env->DeleteLocalRef(jDataRefs[i]);
        }

        // 转换输出
        jclass faCls = env->FindClass("[F");
        jobjectArray result = env->NewObjectArray(nOut, faCls, nullptr);
        for (int i = 0; i < nOut; i++) {
            auto info = outputs[i].GetTensorTypeAndShapeInfo();
            size_t cnt = info.GetElementCount();
            const float* ptr = outputs[i].GetTensorData<float>();
            jfloatArray ja = env->NewFloatArray(cnt);
            env->SetFloatArrayRegion(ja, 0, cnt, ptr);
            env->SetObjectArrayElement(result, i, ja);
            env->DeleteLocalRef(ja);
        }

        return result;
    } catch (const Ort::Exception& e) {
        LOGE("Run failed: %s", e.what());
        return nullptr;
    }
}

JNIEXPORT void JNICALL
Java_com_moss_tts_OrtNative_nativeRelease(JNIEnv*, jobject, jlong handle) {
    delete reinterpret_cast<SessionHandle*>(handle);
}

// 获取session的输入名列表
JNIEXPORT jobjectArray JNICALL
Java_com_moss_tts_OrtNative_nativeGetInputNames(
    JNIEnv* env, jobject, jlong handle) {
    if (!handle) return nullptr;
    auto* h = reinterpret_cast<SessionHandle*>(handle);
    try {
        auto names = h->session->GetInputNames();
        jclass strCls = env->FindClass("java/lang/String");
        jobjectArray result = env->NewObjectArray(names.size(), strCls, nullptr);
        for (size_t i = 0; i < names.size(); i++) {
            jstring s = env->NewStringUTF(names[i].c_str());
            env->SetObjectArrayElement(result, i, s);
            env->DeleteLocalRef(s);
        }
        return result;
    } catch (const Ort::Exception& e) {
        LOGE("GetInputNames failed: %s", e.what());
        return nullptr;
    }
}

// 支持混合tensor类型的Run (float32/int32/int64)
// inputTypes: 0=float32, 1=int32, 2=int64
JNIEXPORT jobjectArray JNICALL
Java_com_moss_tts_OrtNative_nativeRunTyped(
    JNIEnv* env, jobject, jlong handle,
    jobjectArray jInputNames, jobjectArray jInputShapes, jobjectArray jInputDatas, jintArray jInputTypes,
    jobjectArray jOutputNames) {
    if (!handle) return nullptr;
    auto* h = reinterpret_cast<SessionHandle*>(handle);

    try {
    // DEBUG LOG
    {
        int dn = env->GetArrayLength(jInputTypes);
        jint* dt = env->GetIntArrayElements(jInputTypes, nullptr);
        LOGI("nativeRunTyped: %d inputs", dn);
        for (int i = 0; i < dn; i++) LOGI("  type[%d]=%d", i, dt[i]);
        env->ReleaseIntArrayElements(jInputTypes, dt, JNI_ABORT);
    }
        auto nameStrs = jarrayToVec(env, jInputNames);
        int nIn = (int)nameStrs.size();
        std::vector<const char*> namePtrs;
        for (auto& s : nameStrs) namePtrs.push_back(s.c_str());

        jint* inputTypes = env->GetIntArrayElements(jInputTypes, nullptr);

        std::vector<Ort::Value> tensors;
        std::vector<jarray> jDataRefs;  // 通用引用
        std::vector<void*> dataPtrs;
        std::vector<int> typeTracker;

        for (int i = 0; i < nIn; i++) {
            jintArray jShape = (jintArray)env->GetObjectArrayElement(jInputShapes, i);
            jint* sh = env->GetIntArrayElements(jShape, nullptr);
            int shLen = env->GetArrayLength(jShape);
            std::vector<int64_t> shape(sh, sh + shLen);
            env->ReleaseIntArrayElements(jShape, sh, JNI_ABORT);
            env->DeleteLocalRef(jShape);

            int type = inputTypes[i];
            auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            // inputDatas统一是ByteArray，按type解释
            jbyteArray jData = (jbyteArray)env->GetObjectArrayElement(jInputDatas, i);
            jDataRefs.push_back(jData);
            jbyte* bytes = env->GetByteArrayElements(jData, nullptr);
            dataPtrs.push_back(bytes);
            typeTracker.push_back(type);

            if (type == 0) {  // float32: ByteArray直接当float[]用
                size_t byteLen = env->GetArrayLength(jData);
                size_t len = byteLen / 4;
                if (shape.empty()) shape = {(int64_t)len};
                tensors.push_back(Ort::Value::CreateTensor<float>(mem, (float*)bytes, len, shape.data(), shape.size()));
            } else if (type == 1) {  // int32: ByteArray直接当int32[]用
                size_t byteLen = env->GetArrayLength(jData);
                size_t len = byteLen / 4;
                if (shape.empty()) shape = {(int64_t)len};
                tensors.push_back(Ort::Value::CreateTensor<int32_t>(mem, (int32_t*)bytes, len, shape.data(), shape.size()));
            } else if (type == 2) {  // int64: ByteArray直接当int64[]用
                size_t byteLen = env->GetArrayLength(jData);
                size_t len = byteLen / 8;
                if (shape.empty()) shape = {(int64_t)len};
                tensors.push_back(Ort::Value::CreateTensor<int64_t>(mem, (int64_t*)bytes, len, shape.data(), shape.size()));
            }
        }

        auto outNameStrs = jarrayToVec(env, jOutputNames);
        int nOut = (int)outNameStrs.size();
        std::vector<const char*> outPtrs;
        for (auto& s : outNameStrs) outPtrs.push_back(s.c_str());

        auto outputs = h->session->Run(Ort::RunOptions{nullptr},
            namePtrs.data(), tensors.data(), nIn,
            outPtrs.data(), nOut);

        // 释放输入 (统一是ByteArray)
        for (int i = 0; i < nIn; i++) {
            env->ReleaseByteArrayElements((jbyteArray)jDataRefs[i], (jbyte*)dataPtrs[i], JNI_ABORT);
            env->DeleteLocalRef(jDataRefs[i]);
        }
        env->ReleaseIntArrayElements(jInputTypes, inputTypes, JNI_ABORT);

        // 输出统一转为ByteArray (每个元素8字节，兼容所有类型)
        jclass baCls = env->FindClass("[B");
        jobjectArray result = env->NewObjectArray(nOut, baCls, nullptr);
        for (int i = 0; i < nOut; i++) {
            auto info = outputs[i].GetTensorTypeAndShapeInfo();
            size_t cnt = info.GetElementCount();
            auto elemType = info.GetElementType();

            size_t elemSize = 4;
            if (elemType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) elemSize = 8;
            if (elemType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) elemSize = 2;

            size_t totalBytes = cnt * elemSize;
            jbyteArray ja = env->NewByteArray(totalBytes);

            if (elemType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || elemType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
                const float* ptr = outputs[i].GetTensorData<float>();
                env->SetByteArrayRegion(ja, 0, totalBytes, (const jbyte*)ptr);
            } else if (elemType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
                const int32_t* ptr = outputs[i].GetTensorData<int32_t>();
                env->SetByteArrayRegion(ja, 0, totalBytes, (const jbyte*)ptr);
            } else if (elemType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                const int64_t* ptr = outputs[i].GetTensorData<int64_t>();
                env->SetByteArrayRegion(ja, 0, totalBytes, (const jbyte*)ptr);
            }
            env->SetObjectArrayElement(result, i, ja);
            env->DeleteLocalRef(ja);
        }

        return result;
    } catch (const Ort::Exception& e) {
        LOGE("RunTyped failed: %s", e.what());
        return nullptr;
    }
}

// Benchmark: 加载模型并运行N次，返回每次耗时(ms)
JNIEXPORT jintArray JNICALL
Java_com_moss_tts_OrtNative_nativeBenchmark(
    JNIEnv* env, jobject, jstring modelPath, jint numThreads, jint numRuns) {
    if (!initOrt()) return nullptr;
    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    LOGI("Benchmark: %s threads=%d runs=%d", path, numThreads, numRuns);

    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(numThreads);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
        auto session = std::make_unique<Ort::Session>(*g_env, path, opts);
        env->ReleaseStringUTFChars(modelPath, path);

        Ort::AllocatorWithDefaultOptions alloc;
        size_t nIn = session->GetInputCount();

        // 构建输入tensor (随机数据)
        std::vector<Ort::Value> tensors;
        std::vector<std::vector<int64_t>> shapes;
        std::vector<std::vector<float>> buffers;

        for (size_t i = 0; i < nIn; i++) {
            auto typeInfo = session->GetInputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            auto dims = tensorInfo.GetShape();
            auto elemType = tensorInfo.GetElementType();

            size_t cnt = 1;
            for (auto d : dims) cnt *= (d > 0 ? d : 1);

            if (elemType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                std::vector<float> buf(cnt, 0.5f);
                auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                tensors.push_back(Ort::Value::CreateTensor<float>(mem, buf.data(), cnt, dims.data(), dims.size()));
                buffers.push_back(std::move(buf));
            } else if (elemType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                // timestep等int64输入
                std::vector<int64_t> ibuf(cnt, 0);
                auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                tensors.push_back(Ort::Value::CreateTensor<int64_t>(mem, ibuf.data(), cnt, dims.data(), dims.size()));
            } else {
                LOGI("  Input %zu: unsupported type %d, using float", i, elemType);
                std::vector<float> buf(cnt, 0.5f);
                auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                tensors.push_back(Ort::Value::CreateTensor<float>(mem, buf.data(), cnt, dims.data(), dims.size()));
                buffers.push_back(std::move(buf));
            }
            shapes.push_back(dims);
            LOGI("  Input %zu: type=%d elems=%zu", i, elemType, cnt);
        }

        // 输入输出名
        auto inNameVec = session->GetInputNames();
        auto outNameVec = session->GetOutputNames();
        size_t nOut = outNameVec.size();
        std::vector<const char*> inPtrs, outPtrs;
        for (auto& s : inNameVec) inPtrs.push_back(s.c_str());
        for (auto& s : outNameVec) outPtrs.push_back(s.c_str());

        // Warmup
        for (int i = 0; i < 2; i++) {
            session->Run(Ort::RunOptions{nullptr}, inPtrs.data(), tensors.data(), nIn, outPtrs.data(), nOut);
        }
        LOGI("Warmup done, benchmarking...");

        // Benchmark
        jintArray result = env->NewIntArray(numRuns);
        jint* times = env->GetIntArrayElements(result, nullptr);
        for (int i = 0; i < numRuns; i++) {
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            session->Run(Ort::RunOptions{nullptr}, inPtrs.data(), tensors.data(), nIn, outPtrs.data(), nOut);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            times[i] = (jint)((t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_nsec - t0.tv_nsec) / 1000000);
            LOGI("  Run %d: %dms", i, times[i]);
        }
        env->ReleaseIntArrayElements(result, times, 0);
        return result;

    } catch (const Ort::Exception& e) {
        LOGE("Benchmark failed: %s", e.what());
        env->ReleaseStringUTFChars(modelPath, path);
        return nullptr;
    }
}

} // extern "C"

// ============================================================
// MOSS TTS: C++层管理past cache，避免ByteArray GC压力
// ============================================================


struct MossTtsContext {
    SessionHandle* prefillSession;
    SessionHandle* decodeSession;
    SessionHandle* localFixedSession;
    SessionHandle* codecDecodeSession;

    // past cache: output_name -> raw bytes + shape (在C++持有，不传回Java)
    struct TensorBuf {
        std::vector<uint8_t> data;
        std::vector<int64_t> shape;
    };
    std::map<std::string, TensorBuf> pastKv;  // key: "past_key_0" ... "past_value_11"

    std::vector<uint8_t> globalHidden;  // float32, shape [1, hidden_size]
    int pastValidLength = 0;
    int nVq = 16;
    int rowWidth = 17;
    int audioPadTokenId = 0;
    int audioAssistantSlotTokenId = 0;
    int heads = 12;
    int headDim = 64;
};

extern "C" {

// 初始化MOSS TTS上下文：运行prefill，存储past cache
JNIEXPORT jlong JNICALL
Java_com_moss_tts_OrtNative_nativeMossInit(
    JNIEnv* env, jobject,
    jlong prefillHandle, jlong decodeHandle, jlong localFixedHandle, jlong codecDecodeHandle,
    jbyteArray jInputIds, jintArray jInputIdsShape,
    jbyteArray jAttnMask, jintArray jAttnMaskShape,
    jint nVq, jint rowWidth, jint padTokenId, jint assistantSlotTokenId) {

    if (!prefillHandle || !decodeHandle || !localFixedHandle) return 0;

    struct timespec tInitStart;
    clock_gettime(CLOCK_MONOTONIC, &tInitStart);

    auto* ctx = new MossTtsContext();
    ctx->prefillSession = reinterpret_cast<SessionHandle*>(prefillHandle);
    ctx->decodeSession = reinterpret_cast<SessionHandle*>(decodeHandle);
    ctx->localFixedSession = reinterpret_cast<SessionHandle*>(localFixedHandle);
    ctx->codecDecodeSession = reinterpret_cast<SessionHandle*>(codecDecodeHandle);
    ctx->nVq = nVq;
    ctx->rowWidth = rowWidth;
    ctx->audioPadTokenId = padTokenId;
    ctx->audioAssistantSlotTokenId = assistantSlotTokenId;

    try {
        // --- Run prefill ---
        struct timespec tPrefillStart;
        clock_gettime(CLOCK_MONOTONIC, &tPrefillStart);

        auto prefillOutputs = ctx->prefillSession->session->GetOutputNames();
        auto prefillInputs = ctx->prefillSession->session->GetInputNames();
        std::vector<const char*> inNames, outNames;
        for (auto& s : prefillInputs) inNames.push_back(s.c_str());
        for (auto& s : prefillOutputs) outNames.push_back(s.c_str());

        // 构建input_ids tensor
        jbyte* idsBytes = env->GetByteArrayElements(jInputIds, nullptr);
        jint* idsShape = env->GetIntArrayElements(jInputIdsShape, nullptr);
        int idsShapeLen = env->GetArrayLength(jInputIdsShape);
        std::vector<int64_t> idsShapeVec(idsShape, idsShape + idsShapeLen);
        size_t idsLen = env->GetArrayLength(jInputIds) / 4;
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto idsTensor = Ort::Value::CreateTensor<int32_t>(mem, (int32_t*)idsBytes, idsLen, idsShapeVec.data(), idsShapeVec.size());

        // 构建attention_mask tensor
        jbyte* maskBytes = env->GetByteArrayElements(jAttnMask, nullptr);
        jint* maskShape = env->GetIntArrayElements(jAttnMaskShape, nullptr);
        int maskShapeLen = env->GetArrayLength(jAttnMaskShape);
        std::vector<int64_t> maskShapeVec(maskShape, maskShape + maskShapeLen);
        size_t maskLen = env->GetArrayLength(jAttnMask) / 4;
        auto maskTensor = Ort::Value::CreateTensor<int32_t>(mem, (int32_t*)maskBytes, maskLen, maskShapeVec.data(), maskShapeVec.size());

        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(idsTensor));
        inputs.push_back(std::move(maskTensor));

        auto outputs = ctx->prefillSession->session->Run(Ort::RunOptions{nullptr},
            inNames.data(), inputs.data(), 2,
            outNames.data(), outNames.size());

        // 释放JNI数组
        env->ReleaseByteArrayElements(jInputIds, idsBytes, JNI_ABORT);
        env->ReleaseIntArrayElements(jInputIdsShape, idsShape, JNI_ABORT);
        env->ReleaseByteArrayElements(jAttnMask, maskBytes, JNI_ABORT);
        env->ReleaseIntArrayElements(jAttnMaskShape, maskShape, JNI_ABORT);

        // 提取global_hidden: output[0] shape [1, seq, hidden_size]，取最后hidden_size个float
        {
            auto info = outputs[0].GetTensorTypeAndShapeInfo();
            size_t cnt = info.GetElementCount();
            auto dims = info.GetShape();
            size_t hiddenFloats = dims.size() >= 3 ? dims.back() : 768;
            const float* ptr = outputs[0].GetTensorData<float>();
            size_t start = cnt - hiddenFloats;
            ctx->globalHidden.resize(hiddenFloats * 4);
            memcpy(ctx->globalHidden.data(), ptr + start, hiddenFloats * 4);
        }

        // 提取past_kv: 从present_key_0..present_value_11 存入C++ map
        for (size_t i = 1; i < prefillOutputs.size(); i++) {
            auto info = outputs[i].GetTensorTypeAndShapeInfo();
            size_t cnt = info.GetElementCount();
            auto dims = info.GetShape();
            size_t totalBytes = cnt * 4;  // float32

            std::string name = prefillOutputs[i];
            std::string pastName = name;
            // present_key_0 -> past_key_0
            auto pos = pastName.find("present_");
            if (pos != std::string::npos) pastName.replace(pos, 8, "past_");

            MossTtsContext::TensorBuf buf;
            buf.data.resize(totalBytes);
            memcpy(buf.data.data(), outputs[i].GetTensorData<float>(), totalBytes);
            buf.shape = dims;
            ctx->pastKv[pastName] = std::move(buf);
        }

        // 计算pastValidLength = sum(attention_mask)
        {
            // jAttnMask是ByteArray，内部是int32数据
            jbyte* maskArr = env->GetByteArrayElements(jAttnMask, nullptr);
            int maskLenI = env->GetArrayLength(jAttnMask) / 4;  // int32 count
            int32_t* maskInt = (int32_t*)maskArr;
            int sum = 0;
            for (int i = 0; i < maskLenI; i++) if (maskInt[i]) sum++;
            ctx->pastValidLength = sum;
            env->ReleaseByteArrayElements(jAttnMask, maskArr, JNI_ABORT);
        }

        // 检测heads和headDim from第一个past_kv
        if (!ctx->pastKv.empty()) {
            auto& firstShape = ctx->pastKv.begin()->second.shape;
            if (firstShape.size() == 4) {
                // shape: [1, seq_len, heads, head_dim]
                ctx->heads = (int)firstShape[2];
                ctx->headDim = (int)firstShape[3];
            }
        }

        LOGI("MossInit OK: pastKv=%zu, globalHidden=%zu, pastValidLen=%d, heads=%d, headDim=%d",
             ctx->pastKv.size(), ctx->globalHidden.size(), ctx->pastValidLength, ctx->heads, ctx->headDim);

        struct timespec tInitEnd;
        clock_gettime(CLOCK_MONOTONIC, &tInitEnd);
        int initMs = (tInitEnd.tv_sec - tInitStart.tv_sec) * 1000 + (tInitEnd.tv_nsec - tInitStart.tv_nsec) / 1000000;
        int prefillMs = (tInitEnd.tv_sec - tPrefillStart.tv_sec) * 1000 + (tInitEnd.tv_nsec - tPrefillStart.tv_nsec) / 1000000;
        LOGI("[TTS-BENCH] Prefill: %dms (init total: %dms, input tokens: %d)",
             prefillMs, initMs, ctx->pastValidLength);

        return reinterpret_cast<jlong>(ctx);
    } catch (const Ort::Exception& e) {
        LOGE("MossInit failed: %s", e.what());
        delete ctx;
        return 0;
    }
}

// 生成一帧 + decode step（全部在C++内完成，past cache不传回Java）
// 返回: int[] = {shouldContinue, frameTokenId_0, ..., frameTokenId_15}
JNIEXPORT jintArray JNICALL
Java_com_moss_tts_OrtNative_nativeMossNextFrame(
    JNIEnv* env, jobject, jlong contextHandle,
    jbyteArray jRepMask, jintArray jRepMaskShape,
    jbyteArray jAssistantRandomU,
    jbyteArray jAudioRandomU, jintArray jAudioRandomUShape) {

    if (!contextHandle) return nullptr;
    auto* ctx = reinterpret_cast<MossTtsContext*>(contextHandle);

    try {
        // === Step 1: local_fixed_sampled_frame ===
        auto lfsNames = ctx->localFixedSession->session->GetInputNames();
        auto lfsOutNames = ctx->localFixedSession->session->GetOutputNames();

        // 构建inputs: global_hidden, [repetition_seen_mask], [assistant_random_u], [audio_random_u]
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // 过滤: 只传session实际接受的输入
        std::set<std::string> acceptedSet(lfsNames.begin(), lfsNames.end());

        std::vector<Ort::Value> lfsInputs;
        std::vector<const char*> lfsInNames;

        // global_hidden — 始终需要
        if (acceptedSet.count("global_hidden")) {
            size_t ghFloats = ctx->globalHidden.size() / 4;
            std::vector<int64_t> ghShape = {1, (int64_t)ghFloats};
            lfsInputs.push_back(Ort::Value::CreateTensor<float>(mem,
                (float*)ctx->globalHidden.data(), ghFloats, ghShape.data(), ghShape.size()));
            lfsInNames.push_back("global_hidden");
        }

        // repetition_seen_mask
        jbyte* repBytes = nullptr;
        if (acceptedSet.count("repetition_seen_mask") && jRepMask) {
            repBytes = env->GetByteArrayElements(jRepMask, nullptr);
            jint* repShape = env->GetIntArrayElements(jRepMaskShape, nullptr);
            int repShapeLen = env->GetArrayLength(jRepMaskShape);
            std::vector<int64_t> rs(repShape, repShape + repShapeLen);
            size_t repLen = env->GetArrayLength(jRepMask) / 4;
            lfsInputs.push_back(Ort::Value::CreateTensor<int32_t>(mem, (int32_t*)repBytes, repLen, rs.data(), rs.size()));
            lfsInNames.push_back("repetition_seen_mask");
            env->ReleaseIntArrayElements(jRepMaskShape, repShape, JNI_ABORT);
        }

        // assistant_random_u
        jbyte* arBytes = nullptr;
        if (acceptedSet.count("assistant_random_u") && jAssistantRandomU) {
            arBytes = env->GetByteArrayElements(jAssistantRandomU, nullptr);
            std::vector<int64_t> arShape = {1};
            lfsInputs.push_back(Ort::Value::CreateTensor<float>(mem, (float*)arBytes, 1, arShape.data(), arShape.size()));
            lfsInNames.push_back("assistant_random_u");
        }

        // audio_random_u
        jbyte* auBytes = nullptr;
        if (acceptedSet.count("audio_random_u") && jAudioRandomU) {
            auBytes = env->GetByteArrayElements(jAudioRandomU, nullptr);
            jint* auShape = env->GetIntArrayElements(jAudioRandomUShape, nullptr);
            int auShapeLen = env->GetArrayLength(jAudioRandomUShape);
            std::vector<int64_t> as(auShape, auShape + auShapeLen);
            size_t auLen = env->GetArrayLength(jAudioRandomU) / 4;
            lfsInputs.push_back(Ort::Value::CreateTensor<float>(mem, (float*)auBytes, auLen, as.data(), as.size()));
            lfsInNames.push_back("audio_random_u");
            env->ReleaseIntArrayElements(jAudioRandomUShape, auShape, JNI_ABORT);
        }

        // 输出名
        std::vector<const char*> lfsOutPtrs;
        for (auto& s : lfsOutNames) lfsOutPtrs.push_back(s.c_str());

        auto lfsOutputs = ctx->localFixedSession->session->Run(Ort::RunOptions{nullptr},
            lfsInNames.data(), lfsInputs.data(), lfsInputs.size(),
            lfsOutPtrs.data(), lfsOutPtrs.size());

        // 释放JNI输入
        if (repBytes) env->ReleaseByteArrayElements(jRepMask, repBytes, JNI_ABORT);
        if (arBytes) env->ReleaseByteArrayElements(jAssistantRandomU, arBytes, JNI_ABORT);
        if (auBytes) env->ReleaseByteArrayElements(jAudioRandomU, auBytes, JNI_ABORT);

        // 解析should_continue + frame_token_ids
        int shouldContinue = 0;
        std::vector<int> frameTokenIds(ctx->nVq, 0);

        for (size_t i = 0; i < lfsOutNames.size(); i++) {
            if (lfsOutNames[i] == "should_continue") {
                // should_continue是int32: 1=继续, 0=停止
                shouldContinue = *(int32_t*)lfsOutputs[i].GetTensorData<int32_t>();
            } else if (lfsOutNames[i] == "frame_token_ids") {
                const int32_t* ptr = lfsOutputs[i].GetTensorData<int32_t>();
                for (int ch = 0; ch < ctx->nVq; ch++) frameTokenIds[ch] = ptr[ch];
            }
        }

        // 准备返回值: {shouldContinue, tokenId0, tokenId1, ...}
        jintArray result = env->NewIntArray(1 + ctx->nVq);
        jint* arr = env->GetIntArrayElements(result, nullptr);
        arr[0] = shouldContinue;
        for (int ch = 0; ch < ctx->nVq; ch++) arr[1 + ch] = frameTokenIds[ch];
        env->ReleaseIntArrayElements(result, arr, 0);

        if (shouldContinue == 0) return result;

        // === Step 2: decode_step (更新past cache + global_hidden) ===
        auto decodeInNames = ctx->decodeSession->session->GetInputNames();
        auto decodeOutNames = ctx->decodeSession->session->GetOutputNames();

        // 构建input_ids: [audio_assistant_slot_token_id, frameTokenIds..., pad...]
        std::vector<int32_t> nextRow(ctx->rowWidth, ctx->audioPadTokenId);
        nextRow[0] = ctx->audioAssistantSlotTokenId;
        for (int ch = 0; ch < ctx->nVq; ch++) nextRow[ch + 1] = frameTokenIds[ch];

        std::vector<Ort::Value> decodeInputs;
        std::vector<const char*> decodeInPtrs;
        std::vector<std::vector<uint8_t>> tempBuffers;  // 临时buffer保持tensor数据存活

        for (auto& name : decodeInNames) {
            if (name == "input_ids") {
                std::vector<int64_t> shape = {1, 1, (int64_t)ctx->rowWidth};
                decodeInputs.push_back(Ort::Value::CreateTensor<int32_t>(mem,
                    nextRow.data(), nextRow.size(), shape.data(), shape.size()));
                decodeInPtrs.push_back(name.c_str());
            } else if (name == "past_valid_lengths") {
                int32_t pvl = ctx->pastValidLength;
                std::vector<int64_t> shape = {1};
                decodeInputs.push_back(Ort::Value::CreateTensor<int32_t>(mem,
                    &pvl, 1, shape.data(), shape.size()));
                decodeInPtrs.push_back(name.c_str());
            } else if (ctx->pastKv.count(name)) {
                auto& buf = ctx->pastKv[name];
                size_t floatCnt = buf.data.size() / 4;
                decodeInputs.push_back(Ort::Value::CreateTensor<float>(mem,
                    (float*)buf.data.data(), floatCnt, buf.shape.data(), buf.shape.size()));
                decodeInPtrs.push_back(name.c_str());
            }
        }

        std::vector<const char*> decodeOutPtrs;
        for (auto& s : decodeOutNames) decodeOutPtrs.push_back(s.c_str());

        auto decodeOutputs = ctx->decodeSession->session->Run(Ort::RunOptions{nullptr},
            decodeInPtrs.data(), decodeInputs.data(), decodeInputs.size(),
            decodeOutPtrs.data(), decodeOutPtrs.size());

        // 更新global_hidden
        {
            auto info = decodeOutputs[0].GetTensorTypeAndShapeInfo();
            size_t cnt = info.GetElementCount();
            auto dims = info.GetShape();
            size_t hiddenFloats = dims.size() >= 3 ? dims.back() : 768;
            const float* ptr = decodeOutputs[0].GetTensorData<float>();
            size_t start = cnt - hiddenFloats;
            ctx->globalHidden.resize(hiddenFloats * 4);
            memcpy(ctx->globalHidden.data(), ptr + start, hiddenFloats * 4);
        }

        // 更新past_kv: 先收集新数据到临时map，再替换（避免旧tensor引用被释放的内存）
        std::map<std::string, MossTtsContext::TensorBuf> newPastKv;
        for (size_t i = 1; i < decodeOutNames.size(); i++) {
            auto info = decodeOutputs[i].GetTensorTypeAndShapeInfo();
            size_t cnt = info.GetElementCount();
            auto dims = info.GetShape();
            size_t totalBytes = cnt * 4;

            std::string name = decodeOutNames[i];
            // present_key_0 -> past_key_0
            auto pos = name.find("present_");
            if (pos != std::string::npos) name.replace(pos, 8, "past_");

            MossTtsContext::TensorBuf buf;
            buf.data.resize(totalBytes);
            memcpy(buf.data.data(), decodeOutputs[i].GetTensorData<float>(), totalBytes);
            buf.shape = dims;
            newPastKv[name] = std::move(buf);
        }
        // 先释放decodeOutputs中的tensor（它们引用旧pastKv数据）
        decodeOutputs.clear();
        // 现在安全替换pastKv
        ctx->pastKv = std::move(newPastKv);

        ctx->pastValidLength++;

        return result;
    } catch (const Ort::Exception& e) {
        LOGE("MossNextFrame failed: %s", e.what());
        return nullptr;
    }
}

// Codec decode: 16通道tokens -> 44.1kHz音频 (不涉及past cache，直接用session)
JNIEXPORT jbyteArray JNICALL
Java_com_moss_tts_OrtNative_nativeMossCodecDecode(
    JNIEnv* env, jobject, jlong codecHandle,
    jbyteArray jCodes, jintArray jCodesShape,
    jbyteArray jCodesLengths, jintArray jCodesLengthsShape) {

    if (!codecHandle) return nullptr;
    auto* h = reinterpret_cast<SessionHandle*>(codecHandle);

    try {
        auto inNames = h->session->GetInputNames();
        auto outNames = h->session->GetOutputNames();

        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> inputs;
        std::vector<const char*> inPtrs;

        // 输入格式自适应(2026-10-10 codec换血):
        //   旧 decoder_int8: codes int64[16,b,T] + codes_lengths int64
        //   新 decode_full:  audio_codes int32[b,T,16] + audio_code_lengths int32
        // Kotlin侧固定打包int64[16,b,T], 这里按session实际输入dtype自动转置+cast
        bool newFormat = (h->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType()
                          == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
        std::vector<int32_t> codes32Buf;
        std::vector<int32_t> cl32Buf;

        // codes tensor
        jbyte* codesBytes = env->GetByteArrayElements(jCodes, nullptr);
        jint* codesShape = env->GetIntArrayElements(jCodesShape, nullptr);
        int codesShapeLen = env->GetArrayLength(jCodesShape);
        std::vector<int64_t> cs(codesShape, codesShape + codesShapeLen);
        size_t codesLen = env->GetArrayLength(jCodes) / 8;
        if (newFormat && codesShapeLen == 3) {
            // int64[16,b,T] -> int32[b,T,16]
            int64_t d16 = cs[0], b = cs[1], T = cs[2];
            codes32Buf.resize((size_t)(b * T * d16));
            const int64_t* src = (const int64_t*)codesBytes;
            for (int64_t bi = 0; bi < b; ++bi)
                for (int64_t t = 0; t < T; ++t)
                    for (int64_t c = 0; c < d16; ++c)
                        codes32Buf[(size_t)((bi * T + t) * d16 + c)] = (int32_t)src[(size_t)((c * b + bi) * T + t)];
            cs = {b, T, d16};
            inputs.push_back(Ort::Value::CreateTensor<int32_t>(mem, codes32Buf.data(), codes32Buf.size(), cs.data(), cs.size()));
        } else {
            inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem, (int64_t*)codesBytes, codesLen, cs.data(), cs.size()));
        }
        inPtrs.push_back(inNames[0].c_str());

        // codes_lengths tensor
        jbyte* clBytes = env->GetByteArrayElements(jCodesLengths, nullptr);
        jint* clShape = env->GetIntArrayElements(jCodesLengthsShape, nullptr);
        int clShapeLen = env->GetArrayLength(jCodesLengthsShape);
        std::vector<int64_t> cls(clShape, clShape + clShapeLen);
        size_t clLen = env->GetArrayLength(jCodesLengths) / 8;
        if (newFormat) {
            cl32Buf.assign((const int64_t*)clBytes, (const int64_t*)clBytes + clLen);
            inputs.push_back(Ort::Value::CreateTensor<int32_t>(mem, cl32Buf.data(), cl32Buf.size(), cls.data(), cls.size()));
        } else {
            inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem, (int64_t*)clBytes, clLen, cls.data(), cls.size()));
        }
        inPtrs.push_back(inNames[1].c_str());

        std::vector<const char*> outPtrs;
        for (auto& s : outNames) outPtrs.push_back(s.c_str());

        auto outputs = h->session->Run(Ort::RunOptions{nullptr},
            inPtrs.data(), inputs.data(), inputs.size(),
            outPtrs.data(), outPtrs.size());

        env->ReleaseByteArrayElements(jCodes, codesBytes, JNI_ABORT);
        env->ReleaseIntArrayElements(jCodesShape, codesShape, JNI_ABORT);
        env->ReleaseByteArrayElements(jCodesLengths, clBytes, JNI_ABORT);
        env->ReleaseIntArrayElements(jCodesLengthsShape, clShape, JNI_ABORT);

        // 返回audio (float32) + audio_lengths 合并为一个ByteArray
        // 前8字节 = audio_length(int64), 后面 = audio float32 data
        size_t audioLen = 0;
        if (outNames.size() > 1) {
            // audio_lengths dtype自适应: 旧decoder_int8=int64, 新decode_full=int32
            if (outputs[1].GetTensorTypeAndShapeInfo().GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
                audioLen = (size_t)(*outputs[1].GetTensorData<int32_t>());
            else
                audioLen = (size_t)(*outputs[1].GetTensorData<int64_t>());
        }
        auto audioInfo = outputs[0].GetTensorTypeAndShapeInfo();
        size_t audioBytes = audioInfo.GetElementCount() * 4;

        size_t totalBytes = 8 + audioBytes;
        jbyteArray result = env->NewByteArray(totalBytes);
        // 前8字节放audio_length
        env->SetByteArrayRegion(result, 0, 8, (const jbyte*)&audioLen);
        // 后面放audio data
        env->SetByteArrayRegion(result, 8, audioBytes, (const jbyte*)outputs[0].GetTensorData<float>());
        return result;
    } catch (const Ort::Exception& e) {
        LOGE("MossCodecDecode failed: %s", e.what());
        return nullptr;
    }
}

// 释放MOSS TTS上下文
JNIEXPORT void JNICALL
Java_com_moss_tts_OrtNative_nativeMossCleanup(
    JNIEnv*, jobject, jlong contextHandle) {
    delete reinterpret_cast<MossTtsContext*>(contextHandle);
}

// ============================================================
// MOSS TTS: 一次JNI调用生成全部音频
// 优化: pastKv零拷贝(Ort::Value) + 每帧计时 + streaming codec
// 返回: ByteArray = 前4字节sampleCount(int32), 后面int16 PCM
// ============================================================
JNIEXPORT jbyteArray JNICALL
Java_com_moss_tts_OrtNative_nativeMossGenerateAudio(
    JNIEnv* env, jobject, jlong contextHandle,
    jint maxFrames, jint audioCodebookSize) {

    if (!contextHandle) return nullptr;
    auto* ctx = reinterpret_cast<MossTtsContext*>(contextHandle);

    // Fix E(2026-10-10): 固定RNG种子 — 之前用clock_gettime纳秒做种子, 同文本每次生成
    // 波形50%不同("不定时怪音"无法复现对比)。固定种子后输出可复现, 便于定位量化/模型问题。
    uint32_t rng_state = 1234u;  // 固定种子(如需随机采样可换回时钟种子)
    auto rng_float = [&]() -> float {
        rng_state = rng_state * 1103515245 + 12345;
        return (float)(rng_state >> 8) / (float)(1 << 24);
    };

    try {
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        auto lfsInNames = ctx->localFixedSession->session->GetInputNames();
        auto lfsOutNames = ctx->localFixedSession->session->GetOutputNames();
        auto decodeInNames = ctx->decodeSession->session->GetInputNames();
        auto decodeOutNames = ctx->decodeSession->session->GetOutputNames();

        std::set<std::string> lfsAcceptedSet(lfsInNames.begin(), lfsInNames.end());
        std::vector<const char*> lfsOutPtrs;
        for (auto& s : lfsOutNames) lfsOutPtrs.push_back(s.c_str());

        // pastKv零拷贝: 从context TensorBuf创建Ort::Value wrapper
        std::map<std::string, Ort::Value> pastKvValues;
        for (auto& [name, buf] : ctx->pastKv) {
            pastKvValues[name] = Ort::Value::CreateTensor<float>(
                mem, (float*)buf.data.data(), buf.data.size() / 4,
                buf.shape.data(), buf.shape.size());
        }

        std::vector<std::set<int32_t>> prevTokenSets(ctx->nVq);
        std::vector<std::vector<int32_t>> allFrames;
        int framesGenerated = 0;
        struct timespec genStart;
        clock_gettime(CLOCK_MONOTONIC, &genStart);

        for (int step = 0; step < maxFrames; step++) {
            struct timespec frameStart;
            clock_gettime(CLOCK_MONOTONIC, &frameStart);

            // === local_fixed_sampled_frame ===
            std::vector<Ort::Value> lfsInputs;
            std::vector<const char*> lfsInPtrs;

            if (lfsAcceptedSet.count("global_hidden")) {
                size_t ghFloats = ctx->globalHidden.size() / 4;
                std::vector<int64_t> ghShape = {1, (int64_t)ghFloats};
                lfsInputs.push_back(Ort::Value::CreateTensor<float>(mem,
                    (float*)ctx->globalHidden.data(), ghFloats, ghShape.data(), ghShape.size()));
                lfsInPtrs.push_back("global_hidden");
            }

            std::vector<int32_t> repMaskData(ctx->nVq * audioCodebookSize, 0);
            if (lfsAcceptedSet.count("repetition_seen_mask")) {
                for (int ch = 0; ch < ctx->nVq; ch++)
                    for (int tid : prevTokenSets[ch])
                        if (tid >= 0 && tid < audioCodebookSize)
                            repMaskData[ch * audioCodebookSize + tid] = 1;
                std::vector<int64_t> rs = {1, ctx->nVq, (int64_t)audioCodebookSize};
                lfsInputs.push_back(Ort::Value::CreateTensor<int32_t>(mem,
                    repMaskData.data(), repMaskData.size(), rs.data(), rs.size()));
                lfsInPtrs.push_back("repetition_seen_mask");
            }

            float asstRand = std::min(0.99999994f, std::max(0.0f, rng_float()));
            if (lfsAcceptedSet.count("assistant_random_u")) {
                std::vector<int64_t> arShape = {1};
                lfsInputs.push_back(Ort::Value::CreateTensor<float>(mem, &asstRand, 1, arShape.data(), arShape.size()));
                lfsInPtrs.push_back("assistant_random_u");
            }

            std::vector<float> audioRandU(ctx->nVq);
            for (int ch = 0; ch < ctx->nVq; ch++) audioRandU[ch] = std::min(0.99999994f, std::max(0.0f, rng_float()));
            if (lfsAcceptedSet.count("audio_random_u")) {
                std::vector<int64_t> auShape = {1, ctx->nVq};
                lfsInputs.push_back(Ort::Value::CreateTensor<float>(mem, audioRandU.data(), audioRandU.size(), auShape.data(), auShape.size()));
                lfsInPtrs.push_back("audio_random_u");
            }

            auto lfsOutputs = ctx->localFixedSession->session->Run(Ort::RunOptions{nullptr},
                lfsInPtrs.data(), lfsInputs.data(), lfsInputs.size(), lfsOutPtrs.data(), lfsOutPtrs.size());

            int shouldContinue = 0;
            std::vector<int32_t> frameTokenIds(ctx->nVq, 0);
            for (size_t i = 0; i < lfsOutNames.size(); i++) {
                if (lfsOutNames[i] == "should_continue")
                    shouldContinue = *(int32_t*)lfsOutputs[i].GetTensorData<int32_t>();
                else if (lfsOutNames[i] == "frame_token_ids") {
                    const int32_t* ptr = lfsOutputs[i].GetTensorData<int32_t>();
                    for (int ch = 0; ch < ctx->nVq; ch++) frameTokenIds[ch] = ptr[ch];
                }
            }

            struct timespec lfsEnd;
            clock_gettime(CLOCK_MONOTONIC, &lfsEnd);
            int lfsMs = (lfsEnd.tv_sec - frameStart.tv_sec) * 1000 + (lfsEnd.tv_nsec - frameStart.tv_nsec) / 1000000;

            if (shouldContinue == 0) {
                LOGI("Frame %d: shouldContinue=0, stopping (local_fixed=%dms)", step, lfsMs);
                break;
            }

            for (int ch = 0; ch < ctx->nVq; ch++) prevTokenSets[ch].insert(frameTokenIds[ch]);
            allFrames.push_back(frameTokenIds);
            framesGenerated++;

            // === decode_step (零拷贝pastKv) ===
            std::vector<int32_t> nextRow(ctx->rowWidth, ctx->audioPadTokenId);
            nextRow[0] = ctx->audioAssistantSlotTokenId;
            for (int ch = 0; ch < ctx->nVq; ch++) nextRow[ch + 1] = frameTokenIds[ch];

            std::vector<Ort::Value> decInputs;
            std::vector<const char*> decInPtrs;
            for (auto& name : decodeInNames) {
                if (name == "input_ids") {
                    std::vector<int64_t> shape = {1, 1, (int64_t)ctx->rowWidth};
                    decInputs.push_back(Ort::Value::CreateTensor<int32_t>(mem, nextRow.data(), nextRow.size(), shape.data(), shape.size()));
                    decInPtrs.push_back(name.c_str());
                } else if (name == "past_valid_lengths") {
                    int32_t pvl = ctx->pastValidLength;
                    std::vector<int64_t> shape = {1};
                    decInputs.push_back(Ort::Value::CreateTensor<int32_t>(mem, &pvl, 1, shape.data(), shape.size()));
                    decInPtrs.push_back(name.c_str());
                } else if (pastKvValues.count(name)) {
                    // 零拷贝: 从pastKvValues wrapper创建输入
                    auto& val = pastKvValues[name];
                    auto info = val.GetTensorTypeAndShapeInfo();
                    decInputs.push_back(Ort::Value::CreateTensor<float>(
                        mem, val.GetTensorMutableData<float>(),
                        info.GetElementCount(), info.GetShape().data(), info.GetShape().size()));
                    decInPtrs.push_back(name.c_str());
                }
            }
            std::vector<const char*> decOutPtrs;
            for (auto& s : decodeOutNames) decOutPtrs.push_back(s.c_str());

            auto decOutputs = ctx->decodeSession->session->Run(Ort::RunOptions{nullptr},
                decInPtrs.data(), decInputs.data(), decInputs.size(), decOutPtrs.data(), decOutPtrs.size());

            // 提取globalHidden
            {
                auto info = decOutputs[0].GetTensorTypeAndShapeInfo();
                size_t cnt = info.GetElementCount();
                auto dims = info.GetShape();
                size_t hiddenFloats = dims.size() >= 3 ? dims.back() : 768;
                const float* ptr = decOutputs[0].GetTensorData<float>();
                ctx->globalHidden.resize(hiddenFloats * 4);
                memcpy(ctx->globalHidden.data(), ptr + (cnt - hiddenFloats), hiddenFloats * 4);
            }

            // 零拷贝更新pastKv: move Ort::Value, 然后立即清空decOutputs释放ONNX输出buffer
            std::map<std::string, Ort::Value> newPastKv;
            for (size_t i = 1; i < decodeOutNames.size(); i++) {
                std::string dname = decodeOutNames[i];
                auto pos = dname.find("present_");
                if (pos != std::string::npos) dname.replace(pos, 8, "past_");
                newPastKv[dname] = std::move(decOutputs[i]);
            }
            decOutputs.clear();  // 释放旧ONNX输出buffer，避免内存叠加
            pastKvValues = std::move(newPastKv);
            ctx->pastValidLength++;

            struct timespec frameEnd;
            clock_gettime(CLOCK_MONOTONIC, &frameEnd);
            int frameMs = (frameEnd.tv_sec - frameStart.tv_sec) * 1000 + (frameEnd.tv_nsec - frameStart.tv_nsec) / 1000000;
            int decodeMs = frameMs - lfsMs;
            LOGI("Frame %d: total=%dms (lfs=%dms decode=%dms) pvl=%d",
                 step, frameMs, lfsMs, decodeMs, ctx->pastValidLength);
        }

        struct timespec genEnd;
        clock_gettime(CLOCK_MONOTONIC, &genEnd);
        int genMs = (genEnd.tv_sec - genStart.tv_sec) * 1000 + (genEnd.tv_nsec - genStart.tv_nsec) / 1000000;
        LOGI("Frame generation done: %d frames in %dms (%.1fms/frame)",
             framesGenerated, genMs, framesGenerated > 0 ? (float)genMs / framesGenerated : 0.0f);

        // === Codec decode (streaming: 每10帧解码一次) ===
        struct timespec tCodecStart;
        clock_gettime(CLOCK_MONOTONIC, &tCodecStart);
        if (allFrames.empty()) {
            jbyteArray result = env->NewByteArray(4);
            int32_t zero = 0;
            env->SetByteArrayRegion(result, 0, 4, (const jbyte*)&zero);
            return result;
        }

        int nVq = ctx->nVq;
        int frameCount = (int)allFrames.size();
        constexpr int CODEC_BATCH = 10;

        // 收集所有解码的音频float
        std::vector<float> allAudioFloats;
        allAudioFloats.reserve(frameCount * 3840); // 44100Hz / ~11.5fps ≈ 3840 samples/frame

        auto codecInNames = ctx->codecDecodeSession->session->GetInputNames();
        auto codecOutNames = ctx->codecDecodeSession->session->GetOutputNames();
        // codec输入格式自适应(2026-10-10 codec换血):
        //   旧decoder_int8: codes int64[16,1,T] + codes_lengths int64
        //   新decode_full:  audio_codes int32[1,T,16] + audio_code_lengths int32
        bool codecNewFormat = (ctx->codecDecodeSession->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType()
                               == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);

        for (int batchStart = 0; batchStart < frameCount; batchStart += CODEC_BATCH) {
            int batchEnd = std::min(batchStart + CODEC_BATCH, frameCount);
            int batchFrames = batchEnd - batchStart;

            std::vector<int64_t> codesData(nVq * batchFrames);
            for (int ch = 0; ch < nVq; ch++)
                for (int f = 0; f < batchFrames; f++)
                    codesData[ch * batchFrames + f] = (int64_t)allFrames[batchStart + f][ch];
            int64_t codesLen = batchFrames;

            std::vector<int32_t> codes32;  // 新格式转换缓冲
            std::vector<int64_t> codesShape;
            std::vector<int64_t> clShape = {1};
            std::vector<Ort::Value> codecInputs;
            if (codecNewFormat) {
                // int64[16,1,T] -> int32[1,T,16]
                codes32.resize((size_t)nVq * batchFrames);
                for (int ch = 0; ch < nVq; ch++)
                    for (int f = 0; f < batchFrames; f++)
                        codes32[(size_t)(f * nVq + ch)] = (int32_t)allFrames[batchStart + f][ch];
                codesShape = {1, (int64_t)batchFrames, (int64_t)nVq};
                int32_t codesLen32 = batchFrames;
                codecInputs.push_back(Ort::Value::CreateTensor<int32_t>(mem, codes32.data(), codes32.size(), codesShape.data(), codesShape.size()));
                codecInputs.push_back(Ort::Value::CreateTensor<int32_t>(mem, &codesLen32, 1, clShape.data(), clShape.size()));
            } else {
                codesShape = {(int64_t)nVq, 1, (int64_t)batchFrames};
                codecInputs.push_back(Ort::Value::CreateTensor<int64_t>(mem, codesData.data(), codesData.size(), codesShape.data(), codesShape.size()));
                codecInputs.push_back(Ort::Value::CreateTensor<int64_t>(mem, &codesLen, 1, clShape.data(), clShape.size()));
            }
            std::vector<const char*> codecInPtrs, codecOutPtrs;
            for (auto& s : codecInNames) codecInPtrs.push_back(s.c_str());
            for (auto& s : codecOutNames) codecOutPtrs.push_back(s.c_str());

            auto codecOutputs = ctx->codecDecodeSession->session->Run(Ort::RunOptions{nullptr},
                codecInPtrs.data(), codecInputs.data(), codecInputs.size(), codecOutPtrs.data(), codecOutPtrs.size());

            size_t audioLen = 0;
            if (codecOutNames.size() > 1) {
                // audio_lengths dtype自适应: 旧decoder_int8=int64, 新decode_full=int32
                if (codecOutputs[1].GetTensorTypeAndShapeInfo().GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
                    audioLen = (size_t)(*codecOutputs[1].GetTensorData<int32_t>());
                else
                    audioLen = (size_t)(*codecOutputs[1].GetTensorData<int64_t>());
            }
            auto audioInfo = codecOutputs[0].GetTensorTypeAndShapeInfo();
            size_t audioFloatCount = audioInfo.GetElementCount();
            const float* audioPtr = codecOutputs[0].GetTensorData<float>();
            if (audioLen > 0 && audioLen < audioFloatCount) audioFloatCount = audioLen;

            for (size_t i = 0; i < audioFloatCount; i++)
                allAudioFloats.push_back(audioPtr[i]);

            LOGI("Codec batch %d-%d: %zu samples", batchStart, batchEnd - 1, audioFloatCount);
        }

        struct timespec tCodecEnd;
        clock_gettime(CLOCK_MONOTONIC, &tCodecEnd);
        int codecMs = (tCodecEnd.tv_sec - tCodecStart.tv_sec) * 1000 + (tCodecEnd.tv_nsec - tCodecStart.tv_nsec) / 1000000;

        int32_t sampleCount = (int32_t)allAudioFloats.size();
        size_t resultBytes = 4 + sampleCount * 2;
        jbyteArray result = env->NewByteArray(resultBytes);
        env->SetByteArrayRegion(result, 0, 4, (const jbyte*)&sampleCount);

        std::vector<int16_t> pcm16(sampleCount);
        for (int32_t i = 0; i < sampleCount; i++) {
            float s = allAudioFloats[i] * 32767.0f;
            if (s > 32767.0f) s = 32767.0f;
            if (s < -32768.0f) s = -32768.0f;
            pcm16[i] = (int16_t)s;
        }
        env->SetByteArrayRegion(result, 4, sampleCount * 2, (const jbyte*)pcm16.data());

        LOGI("MossGenerateAudio: %d frames, %d samples, gen=%dms", framesGenerated, sampleCount, genMs);

        float audioDurationMs = (float)sampleCount / 44100.0f * 1000.0f;
        float rtf = genMs > 0 ? (float)genMs / audioDurationMs : 0.0f;
        LOGI("[TTS-BENCH] 总结: frames=%d, gen=%dms, codec=%dms, audio=%.0fms, RTF=%.2f (1.0=实时)",
             framesGenerated, genMs, codecMs, audioDurationMs, rtf);
        LOGI("[TTS-BENCH] 帧率: %.1fms/帧 (音频帧率~11.5fps, 生成需<87ms/帧才能实时)",
             framesGenerated > 0 ? (float)genMs / framesGenerated : 0.0f);

        return result;

    } catch (const Ort::Exception& e) {
        LOGE("MossGenerateAudio failed: %s", e.what());
        return nullptr;
    }
}

} // extern "C"
