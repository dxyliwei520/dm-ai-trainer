"""诊断用：跑同一条配方，但每 90 秒把 Python 栈打出来（定位卡在哪一行）。"""

import faulthandler
import runpy
import sys

faulthandler.dump_traceback_later(90, repeat=True, exit=False)
sys.argv = ["run.py", r"E:\ai-train\recipes\qwen_smoke.yaml"]
runpy.run_path(r"E:\ai-train\src\ai-toolkit\run.py", run_name="__main__")
