import json
import os
import subprocess
import sys
import time
from urllib.request import urlopen

from filelock import FileLock, Timeout

import engine as e


def spawn(script, logname):
    with (
        (e.STATE.parent / f"{logname}.out").open("ab") as out,
        (e.STATE.parent / f"{logname}.err").open("ab") as err,
    ):
        return subprocess.Popen(
            [sys.executable, "-u", str(e.ROOT / script)],
            cwd=e.ROOT,
            stdout=out,
            stderr=err,
            stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )


def main():
    e.initialize()
    host = os.getenv("MMD_HOST", "127.0.0.1")
    port = int(os.getenv("MMD_PORT", "7860"))
    url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}"

    def live():
        try:
            with urlopen(url + "/config", timeout=2) as r:
                return json.load(r).get("title") == "MMD 立体转制 · Quest 2"
        except (OSError, ValueError):
            return False

    with FileLock(e.STATE / "launch.lock", timeout=30):
        try:
            with FileLock(e.STATE / "worker.lock", timeout=0):
                pass
        except Timeout:
            pass
        else:
            spawn("engine.py", "worker")
        if not live():
            proc = spawn("app.py", "web")
            for _ in range(60):
                if live():
                    break
                if proc.poll() is not None:
                    raise RuntimeError("WebUI 启动失败，请查看 web.err")
                time.sleep(0.5)
            else:
                raise RuntimeError("WebUI 未在预期时间内启动")
    print(url)


if __name__ == "__main__":
    main()
