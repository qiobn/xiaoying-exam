from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    if not (ROOT / "frontend" / "node_modules").exists():
        npm = "npm.cmd" if os.name == "nt" else "npm"
        subprocess.run([npm, "install"], cwd=ROOT / "frontend", env=env, check=True, shell=(os.name == "nt"))
    if not (ROOT / "frontend" / "dist" / "index.html").exists():
        npm = "npm.cmd" if os.name == "nt" else "npm"
        frontend = subprocess.Popen([npm, "run", "dev", "--", "--host", "127.0.0.1"], cwd=ROOT / "frontend", env=env)
        url = "http://127.0.0.1:5173"
    else:
        frontend = None
        url = "http://127.0.0.1:8000"
    backend = subprocess.Popen([sys.executable, "-m", "knowledge_agent.server"], cwd=ROOT, env=env)
    print(f"知识问答工作台启动中：{url}", flush=True)
    time.sleep(1.5)
    webbrowser.open(url)
    try:
        return backend.wait()
    except KeyboardInterrupt:
        return 0
    finally:
        backend.terminate()
        if frontend:
            frontend.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
