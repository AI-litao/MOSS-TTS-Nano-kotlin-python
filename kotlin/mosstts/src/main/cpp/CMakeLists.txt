cmake_minimum_required(VERSION 3.18)
project(moss_tts_jni C CXX)

set(CMAKE_CXX_STANDARD 17)

# ORT JNI — 通过dlopen调sherpa-onnx已有的libonnxruntime.so
add_library(ort_jni SHARED ort_jni.cpp)
target_include_directories(ort_jni PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

find_library(log-lib log)
target_link_libraries(ort_jni ${log-lib} dl)
