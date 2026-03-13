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

"""
PaperBanana-Pro CLI — 全局入口。

支持：
    paperbanana                  启动 GUI
    paperbanana gui              启动 GUI
    paperbanana run [...]        运行批处理 CLI
    paperbanana viewer ...       启动 viewer

兼容别名：
    paperbanana-pro
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def _safe_print(msg: str) -> None:
    """Print that never crashes on non-UTF-8 Windows consoles."""
    try:
        print(msg)
    except UnicodeEncodeError:
        fallback = msg.encode(
            getattr(sys.stdout, "encoding", None) or "utf-8",
            errors="backslashreplace",
        ).decode("ascii", errors="replace")
        print(fallback)


VIEWER_MODULES = {
    "evolution": "visualize.show_pipeline_evolution",
    "pipeline": "visualize.show_pipeline_evolution",
    "eval": "visualize.show_referenced_eval",
    "review": "visualize.show_referenced_eval",
}


def resolve_module_script_path(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin:
        raise FileNotFoundError(f"无法解析模块脚本：{module_name}")
    return Path(spec.origin).resolve()


def launch_streamlit_module(
    module_name: str,
    extra_args: list[str],
    *,
    default_port: int | None = None,
) -> int:
    script_path = resolve_module_script_path(module_name)
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script_path),
    ]
    if default_port is not None:
        cmd.extend(["--server.port", str(default_port)])
    cmd.extend(extra_args)
    _safe_print(f"[PaperBanana-Pro] 启动 Streamlit 应用：{module_name}")
    return subprocess.call(cmd)


def launch_python_module(module_name: str, extra_args: list[str]) -> int:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise FileNotFoundError(f"无法解析 Python 模块：{module_name}")
    cmd = [sys.executable, "-m", module_name, *extra_args]
    _safe_print(f"[PaperBanana-Pro] 运行 CLI 模块：{module_name}")
    return subprocess.call(cmd)


def _launch_gui(extra_args: list[str]) -> int:
    return launch_streamlit_module("demo", extra_args, default_port=8501)


def _launch_cli(extra_args: list[str]) -> int:
    return launch_python_module("main", extra_args)


def _launch_viewer(viewer_name: str, extra_args: list[str]) -> int:
    module_name = VIEWER_MODULES.get(viewer_name, "")
    if not module_name:
        _safe_print(f"[PaperBanana-Pro] 未知 viewer：{viewer_name}\n")
        _print_viewer_help()
        return 1
    return launch_streamlit_module(module_name, extra_args)


def _print_viewer_help() -> None:
    print(
        """
Viewer 子命令：
    paperbanana viewer evolution [args]   启动流程回放 viewer
    paperbanana viewer eval [args]        启动参考评测 viewer

兼容别名：
    paperbanana-pro viewer evolution [args]
    paperbanana-pro viewer eval [args]

别名映射：
    pipeline -> evolution
    review   -> eval
"""
    )


def _print_help() -> None:
    print(
        """
PaperBanana-Pro 🍌  —  Academic Illustration Workbench

主命令：
    paperbanana
    paperbanana gui [args]
    paperbanana run [args]
    paperbanana viewer evolution [args]
    paperbanana viewer eval [args]
    paperbanana --help

兼容别名：
    paperbanana-pro

示例：
    paperbanana
    paperbanana gui --server.port 9000
    paperbanana run --exp_mode dev_full --task_name diagram
    paperbanana run --resume
    paperbanana viewer evolution
    paperbanana viewer eval
    paperbanana-pro --help

安装方式（当前正式支持）：
    uv sync --locked
    uv tool install --editable . --force

未来路线（暂未支持）：
    非 editable uv tool install
    PyPI / 索引发布
"""
    )


def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] == "gui":
        extra = args[1:] if args else []
        raise SystemExit(_launch_gui(extra))

    if args[0] == "run":
        raise SystemExit(_launch_cli(args[1:]))

    if args[0] == "viewer":
        if len(args) == 1 or args[1] in ("--help", "-h"):
            _print_viewer_help()
            return
        raise SystemExit(_launch_viewer(args[1], args[2:]))

    if args[0] in ("--help", "-h"):
        _print_help()
        return

    _safe_print(f"[PaperBanana-Pro] Unknown command: {args[0]}\n")
    _print_help()
    raise SystemExit(1)


if __name__ == "__main__":
    main()
