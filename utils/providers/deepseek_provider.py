"""DeepSeek provider implementation."""

from .openai_base import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek API provider (OpenAI-compatible)."""

    def __init__(self):
        super().__init__(
            api_key_env="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com"
        )

    @property
    def name(self) -> str:
        return "deepseek"

    def get_max_tokens_limit(self, model_name: str) -> int:
        """Get max tokens limit for DeepSeek models."""
        # DeepSeek API 的 max_tokens 限制是 8192
        # 注意：这是输出token限制，不是上下文长度限制
        return 8192

    def supports_multiple_completions(self) -> bool:
        """DeepSeek API does not support n > 1."""
        return False