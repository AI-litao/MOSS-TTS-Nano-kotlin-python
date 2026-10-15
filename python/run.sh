#!/usr/bin/env bash
# MOSS-TTS-Nano CPU 一键启动: 检测/装依赖 -> 检测/下载模型 -> 起 Web UI (8080)
# 手动方式见 README (pip install + python infer.py / python webui.py)
set -e
cd "$(dirname "$0")"

PY=python3
if [ ! -d .venv ]; then
  echo "[run] 首次运行: 创建虚拟环境 .venv ..."
  python3 -m venv .venv
fi
PY=.venv/bin/python

# 依赖检测: 已装则跳过
if "$PY" -c "import onnxruntime, numpy, sentencepiece" 2>/dev/null; then
  echo "[run] 依赖已安装, 跳过"
else
  echo "[run] 安装依赖 (requirements.txt) ..."
  "$PY" -m pip install -q -r requirements.txt
fi

# 模型检测: 无则下载
if "$PY" download_model.py --check-only >/dev/null 2>&1; then
  echo "[run] 模型已就绪"
else
  echo "[run] 模型缺失, 开始下载 ..."
  "$PY" download_model.py
fi

PORT="${PORT:-8080}"
echo "[run] Web UI: http://localhost:${PORT}"
exec "$PY" webui.py --port "${PORT}"
