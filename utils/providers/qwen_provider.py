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
