# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PaperBanana CLI — global entry point.

Usage:
    paperbanana              Launch the Streamlit GUI (default)
    paperbanana gui          Same as above
    paperbanana run [args]   Run CLI batch processing (main.py)
    paperbanana --help       Show help
"""

import os
import sys
import subprocess
from pathlib import Path

# Project root = directory where this file lives
PROJECT_ROOT = Path(__file__).resolve().parent


def _launch_gui(extra_args: list[str]) -> int:
    """Start the Streamlit interactive demo."""
    demo_path = PROJECT_ROOT / "demo.py"
    if not demo_path.exists():
        print(f"[PaperBanana] Error: demo.py not found at {demo_path}")
        return 1

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(demo_path),
        "--server.port", "8501",
        *extra_args,
    ]
    print(f"[PaperBanana] Starting Streamlit GUI ...")
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def _launch_cli(extra_args: list[str]) -> int:
    """Run the CLI batch-processing pipeline (main.py)."""
    main_path = PROJECT_ROOT / "main.py"
    if not main_path.exists():
        print(f"[PaperBanana] Error: main.py not found at {main_path}")
        return 1

    cmd = [sys.executable, str(main_path), *extra_args]
    print(f"[PaperBanana] Running CLI mode ...")
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def _print_help() -> None:
    print("""
PaperBanana 🍌  —  Automated Academic Illustration Generator

Usage:
    paperbanana              Launch the Streamlit GUI (default)
    paperbanana gui [args]   Same as above (extra args forwarded to Streamlit)
    paperbanana run [args]   Run CLI batch processing (main.py args)
    paperbanana --help       Show this help message

Examples:
    paperbanana
    paperbanana gui --server.port 9000
    paperbanana run --exp_mode dev_full --task_name diagram
    paperbanana run --exp_mode demo_full --retrieval_setting auto
""")


def main() -> None:
    args = sys.argv[1:]

    # No args or 'gui' → launch Streamlit
    if not args or args[0] == "gui":
        extra = args[1:] if args else []
        raise SystemExit(_launch_gui(extra))

    # 'run' → CLI batch mode
    if args[0] == "run":
        raise SystemExit(_launch_cli(args[1:]))

    # Help
    if args[0] in ("--help", "-h"):
        _print_help()
        return

    # Unknown command — show help
    print(f"[PaperBanana] Unknown command: {args[0]}\n")
    _print_help()
    raise SystemExit(1)


if __name__ == "__main__":
    main()
