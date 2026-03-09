# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""绘图代码执行器。"""

import base64
import io
import contextlib
import re
import logging
from typing import Any
from io import StringIO

import matplotlib.pyplot as plt

logger = logging.getLogger("PlotExecutor")


def execute_plot_code_with_details(code_text: str, dpi: int = 300) -> dict[str, Any]:
    """
    在独立进程中执行绘图代码并返回结构化结果。

    返回字段：
    - success: 是否成功渲染
    - base64_jpg: 成功时的 JPEG base64
    - stdout / stderr: 代码执行输出
    - exception: 异常文本
    - figure_detected: 是否创建了 matplotlib figure
    """
    match = re.search(r"```python(.*?)```", code_text, re.DOTALL)
    code_clean = match.group(1).strip() if match else code_text.strip()

    plt.switch_backend("Agg")
    plt.close("all")
    plt.rcdefaults()
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()

    try:
        exec_globals = {}
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            exec(code_clean, exec_globals)

        if plt.get_fignums():
            buf = io.BytesIO()
            plt.savefig(buf, format="jpeg", bbox_inches="tight", dpi=dpi)
            plt.close("all")

            buf.seek(0)
            img_bytes = buf.read()
            return {
                "success": True,
                "base64_jpg": base64.b64encode(img_bytes).decode("utf-8"),
                "stdout": stdout_buffer.getvalue(),
                "stderr": stderr_buffer.getvalue(),
                "exception": None,
                "figure_detected": True,
            }

        return {
            "success": False,
            "base64_jpg": None,
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "exception": None,
            "figure_detected": False,
        }

    except Exception as e:
        logger.error(f"❌ 执行绘图代码出错: {e}")
        return {
            "success": False,
            "base64_jpg": None,
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "exception": f"{type(e).__name__}: {e}",
            "figure_detected": False,
        }
    finally:
        plt.close("all")


def execute_plot_code(code_text: str, dpi: int = 300) -> str | None:
    """兼容旧调用：仅返回 base64 图片。"""
    result = execute_plot_code_with_details(code_text, dpi=dpi)
    return result.get("base64_jpg")
