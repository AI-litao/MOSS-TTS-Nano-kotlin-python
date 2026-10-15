// MOSS-TTS-Nano Kotlin 库 (AAR)
// 纯 TTS: MossTtsEngine + TextCleaner + OrtNative(JNI/dlopen ONNX Runtime)
// ORT 运行时 = 微软官方 libonnxruntime.so (MIT, jniLibs/arm64-v8a, 25.8MB)
plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.moss.tts"
    compileSdk = 34
    ndkVersion = "28.0.13004108"

    defaultConfig {
        minSdk = 27
        consumerProguardFiles("consumer-rules.pro")
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
        externalNativeBuild {
            cmake {
                targets += "ort_jni"
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    // 零第三方依赖: ONNX Runtime 走 jniLibs/libonnxruntime.so + dlopen
}
