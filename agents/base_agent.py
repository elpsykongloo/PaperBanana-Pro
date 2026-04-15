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
Base class for agents
"""

from typing import List, Dict, Any
from abc import ABC, abstractmethod

from utils.config import ExpConfig
from utils import image_utils


class BaseAgent(ABC):
    """Base class for agents"""

    def __init__(
        self,
        model_name: str = "",
        system_prompt: str = "",
        exp_config: "ExpConfig" = None,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.exp_config = exp_config

    @abstractmethod
    async def process(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        处理输入数据并返回结果。

        Args:
            data: 输入数据字典
            **kwargs: 子类附加参数

        Returns:
            处理后的数据字典
        """

    # ==================== 统一 Provider 路由 ====================

    async def call_text_api(
        self,
        contents: List[Dict[str, Any]],
        *,
        model_name: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int = 50000,
        max_attempts: int = 5,
        retry_delay: float = 5,
        error_context: str = "",
    ) -> List[str]:
        """
        统一的文本生成 API 调用，根据 provider 自动路由到 Evolink / Gemini。

        Args:
            contents: 通用内容列表（文本和图片混合）
            model_name: 模型名称，默认使用 self.model_name
            system_prompt: 系统提示词，默认使用 self.system_prompt
            temperature: 温度参数，默认使用 self.exp_config.temperature
            max_output_tokens: 最大输出 token 数
            max_attempts: 最大重试次数
            retry_delay: 重试间隔（秒）
            error_context: 错误上下文信息

        Returns:
            响应文本列表
        """
        from utils import generation_utils

        _model = model_name or self.model_name
        _sys = system_prompt or self.system_prompt
        _temp = temperature if temperature is not None else self.exp_config.temperature

        provider = str(getattr(self.exp_config, "provider", "") or "").strip().lower()

        if provider == "evolink":
            return await generation_utils.call_evolink_text_with_retry_async(
                model_name=_model,
                contents=contents,
                config={
                    "system_prompt": _sys,
                    "temperature": _temp,
                    "max_output_tokens": max_output_tokens,
                },
                max_attempts=max_attempts,
                retry_delay=retry_delay,
                error_context=error_context,
            )
        if provider == "gemini":
            from google.genai import types

            return await generation_utils.call_gemini_with_retry_async(
                model_name=_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=_sys,
                    temperature=_temp,
                    candidate_count=1,
                    max_output_tokens=max_output_tokens,
                ),
                max_attempts=max_attempts,
                retry_delay=retry_delay,
                error_context=error_context,
            )

        if provider in {"openrouter", "openai_compatible"}:
            return await generation_utils.call_openai_with_retry_async(
                model_name=_model,
                contents=contents,
                config={
                    "system_prompt": _sys,
                    "temperature": _temp,
                    "candidate_num": 1,
                    "max_completion_tokens": max_output_tokens,
                },
                max_attempts=max_attempts,
                retry_delay=retry_delay,
                error_context=error_context,
            )

        raise ValueError(
            f"Unsupported provider for text generation: {self.exp_config.provider!r}"
        )

    async def call_image_api(
        self,
        prompt: str,
        *,
        model_name: str | None = None,
        contents: List[Dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        aspect_ratio: str = "1:1",
        image_resolution: str = "2K",
        image_urls: List[str] | None = None,
        max_output_tokens: int = 50000,
        max_attempts: int = 5,
        retry_delay: float = 30,
        error_context: str = "",
    ) -> List[str]:
        """
        统一的图像生成 API 调用，根据 provider 自动路由到 Evolink / Gemini / OpenAI。

        Args:
            prompt: 图像描述提示词
            model_name: 图像模型名称，默认使用 self.model_name
            contents: 通用内容列表（Gemini 模式使用）
            system_prompt: 系统提示词
            temperature: 温度参数
            aspect_ratio: 宽高比
            image_resolution: 图像分辨率
            image_urls: 参考图片 URL 列表（Evolink 用于 image-to-image）
            max_output_tokens: 最大输出 token 数
            max_attempts: 最大重试次数
            retry_delay: 重试间隔（秒）
            error_context: 错误上下文信息

        Returns:
            base64 编码的图像字符串列表
        """
        from utils import generation_utils, image_utils

        _model = model_name or self.model_name
        _sys = system_prompt or self.system_prompt
        _temp = temperature if temperature is not None else self.exp_config.temperature
        _contents = contents or [{"type": "text", "text": prompt}]

        provider = str(getattr(self.exp_config, "provider", "") or "").strip().lower()

        if provider == "evolink":
            config = {
                "aspect_ratio": aspect_ratio,
                "quality": image_resolution,
            }
            if image_urls:
                config["image_urls"] = image_urls
            return await generation_utils.call_evolink_image_with_retry_async(
                model_name=_model,
                prompt=prompt,
                config=config,
                max_attempts=max_attempts,
                retry_delay=retry_delay,
                error_context=error_context,
            )
        if provider == "gemini":
            from google.genai import types

            prompt_with_hints = image_utils.build_gemini_image_prompt(
                prompt,
                aspect_ratio=aspect_ratio,
                image_size=image_resolution,
            )
            gemini_contents = self._inject_prompt_into_contents(_contents, prompt_with_hints)
            return await generation_utils.call_gemini_with_retry_async(
                model_name=_model,
                contents=gemini_contents,
                config=types.GenerateContentConfig(
                    system_instruction=_sys,
                    temperature=_temp,
                    candidate_count=1,
                    max_output_tokens=max_output_tokens,
                    response_modalities=["IMAGE"],
                ),
                max_attempts=max_attempts,
                retry_delay=retry_delay,
                error_context=error_context,
            )

        if provider == "openrouter":
            return await generation_utils.call_openrouter_image_generation_with_retry_async(
                model_name=_model,
                prompt=prompt,
                contents=_contents,
                system_prompt=_sys,
                config={
                    "aspect_ratio": aspect_ratio,
                    "image_size": image_resolution,
                    "output_format": "png",
                },
                max_attempts=max_attempts,
                retry_delay=retry_delay,
                error_context=error_context,
            )

        if provider == "openai_compatible":
            return await generation_utils.call_openai_image_generation_with_retry_async(
                model_name=_model,
                prompt=prompt,
                config={
                    "size": "1536x1024",
                    "quality": "high",
                    "background": "opaque",
                    "output_format": "png",
                },
                max_attempts=max_attempts,
                retry_delay=retry_delay,
                error_context=error_context,
            )

        raise ValueError(
            f"Unsupported provider for image generation: {self.exp_config.provider!r}"
        )

    @staticmethod
    def _inject_prompt_into_contents(
        contents: List[Dict[str, Any]],
        prompt_text: str,
    ) -> List[Dict[str, Any]]:
        """Replace the first text part with the prompt, preserving reference images."""
        updated_contents: List[Dict[str, Any]] = []
        prompt_inserted = False
        for item in contents:
            if item.get("type") == "text" and not prompt_inserted:
                updated_contents.append({"type": "text", "text": prompt_text})
                prompt_inserted = True
            else:
                updated_contents.append(item)
        if not prompt_inserted:
            updated_contents.insert(0, {"type": "text", "text": prompt_text})
        return updated_contents
