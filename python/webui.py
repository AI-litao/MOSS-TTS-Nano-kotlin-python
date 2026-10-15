#!/usr/bin/env python3
"""
webui.py — 极简 Web UI (纯标准库 http.server, 零额外依赖)
  python webui.py [--port 8080]   或经 run.sh 一键启动
页面: 文本框 + 12音色下拉 + 语速 + 生成(保存 output/ 并在线试听)
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE.parent / "output"  # 统一产物目录=工程根 output/ (与infer.py/README一致)
sys.path.insert(0, str(HERE))

lock = threading.Lock()
tts = None  # 懒加载 MossTts
tts_error = ""


def get_tts():
    global tts, tts_error
    with lock:
        if tts is None:
            from infer import MossTts
            tts = MossTts()
        return tts


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # 静音访问日志
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = (HERE / "static" / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
        elif self.path.startswith("/output/"):
            name = Path(self.path).name
            f = OUTPUT_DIR / name
            if f.is_file():
                data = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self._json({"ok": False, "error": "not found"}, 404)
        elif self.path == "/api/voices":
            try:
                voices = [{"id": v["id"], "name": v["name"]}
                          for v in get_tts().list_voices()]
                self._json({"ok": True, "voices": voices})
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, 500)
        else:
            self._json({"ok": False, "error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/api/generate":
            self._json({"ok": False, "error": "not found"}, 404)
            return
        n = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(n) or b"{}")
        text = str(req.get("text", ""))
        voice = req.get("voice", 4)
        speed = float(req.get("speed", 1.0))
        stretch = float(req.get("stretch", 1.0))
        try:
            engine = get_tts()
            engine.voice = engine._resolve_voice(voice)
            engine.speed = speed
            engine.stretch = stretch
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out = OUTPUT_DIR / f"tts_{stamp}.wav"
            t0 = time.perf_counter()
            path = engine.synthesize(text, out_path=out)
            self._json({"ok": True, "wav": f"/output/{path.name}",
                        "ms": int((time.perf_counter() - t0) * 1000)})
        except Exception as e:
            self._json({"ok": False, "error": str(e)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[webui] http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
