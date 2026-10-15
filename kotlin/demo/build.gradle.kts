// 最小 demo app: 12音色UI + 首启下载模型 + 生成/播放/保存 output
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.moss.tts.demo"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.moss.tts"
        minSdk = 27
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
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
    implementation(project(":mosstts"))
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
}
