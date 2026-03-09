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

"""Configuration for experiments."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from utils.runtime_settings import RuntimeSettings, resolve_runtime_settings

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # Python < 3.9 fallback


@dataclass
class ExpConfig:
    """Experiment configuration"""

    dataset_name: str
    task_name: Literal["diagram", "plot"] = "diagram"
    split_name: str = "test"
    temperature: float = 1.0
    exp_mode: str = ""
    retrieval_setting: Literal["auto", "auto-full", "manual", "random", "none"] = "auto"
    max_critic_rounds: int = 3
    concurrency_mode: Literal["auto", "manual"] = "auto"
    max_concurrent: int = 20
    model_name: str = ""
    image_model_name: str = ""
    provider: str = "evolink"
    work_dir: Path = Path(__file__).parent.parent
    timezone: str = "America/Los_Angeles"

    timestamp: str | None = None
    runtime_settings: RuntimeSettings = field(init=False)

    def __post_init__(self):
        self.runtime_settings = resolve_runtime_settings(
            self.provider,
            model_name=self.model_name,
            image_model_name=self.image_model_name,
            concurrency_mode=self.concurrency_mode,
            max_concurrent=self.max_concurrent,
            max_critic_rounds=self.max_critic_rounds,
            base_dir=self.work_dir,
        )
        self.provider = self.runtime_settings.provider
        self.model_name = self.runtime_settings.model_name
        self.image_model_name = self.runtime_settings.image_model_name
        self.concurrency_mode = self.runtime_settings.concurrency_mode
        self.max_concurrent = self.runtime_settings.max_concurrent
        self.max_critic_rounds = self.runtime_settings.max_critic_rounds

        if self.timestamp is None:
            tz = ZoneInfo(self.timezone)
            self.timestamp = datetime.now(tz).strftime("%m%d_%H%M")

        self.exp_name = f"{self.timestamp}_{self.retrieval_setting}ret_{self.exp_mode}_{self.split_name}"

        # mkdir result_dir if not exists
        self.result_dir = self.work_dir / "results" / f"{self.dataset_name}_{self.task_name}"
        self.result_dir.mkdir(exist_ok=True, parents=True)
