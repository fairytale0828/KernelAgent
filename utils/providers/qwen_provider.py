# Copyright (c) Meta Platforms, Inc. and affiliates.
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

"""Qwen provider implementation (OpenAI-compatible endpoint)."""

from .openai_base import OpenAICompatibleProvider


class QwenProvider(OpenAICompatibleProvider):
    """Qwen API provider (OpenAI-compatible)."""

    def __init__(self):
        super().__init__(
            api_key_env="DASHSCOPE_API_KEY",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    @property
    def name(self) -> str:
        return "qwen"

    def get_max_tokens_limit(self, model_name: str) -> int:
        """Get max tokens limit for Qwen models."""
        return 8192
