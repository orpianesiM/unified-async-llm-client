"""Selector de proveedor y configuración desde el entorno."""

import os
from collections.abc import AsyncIterator

from unified_async_llm_client.base import BaseLLMClient
from unified_async_llm_client.clients import AnthropicClient, OpenAIClient
from unified_async_llm_client.schemas import (
    ChatRequest,
    ErrorCategory,
    LLMError,
    ModelConfig,
    ModelResponse,
    Provider,
    StreamChunk,
)

_DEFAULT_MODELS = {
    Provider.OPENAI: "gpt-4o-mini",
    Provider.ANTHROPIC: "claude-3-5-haiku-latest",
}


class AsyncLLMManager:
    """Expone una interfaz común para OpenAI y Anthropic."""

    def __init__(self, config: ModelConfig, api_key: str | None = None) -> None:
        self.config = config
        self._api_key = api_key or self._api_key_from_environment(config.provider)
        self._client: BaseLLMClient | None = None
        if self._api_key:
            if config.provider is Provider.OPENAI:
                self._client = OpenAIClient(config, self._api_key)
            else:
                self._client = AnthropicClient(config, self._api_key)

    @classmethod
    def from_env(cls) -> "AsyncLLMManager":
        """Crea el manager usando variables de entorno previamente cargadas."""
        provider = Provider(os.getenv("LLM_PROVIDER", Provider.OPENAI.value).lower())
        config = ModelConfig(
            provider=provider,
            model=os.getenv("LLM_MODEL") or _DEFAULT_MODELS[provider],
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "512")),
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
        )
        return cls(config)

    @staticmethod
    def _api_key_from_environment(provider: Provider) -> str | None:
        key_name = "OPENAI_API_KEY" if provider is Provider.OPENAI else "ANTHROPIC_API_KEY"
        return os.getenv(key_name) or None

    def _missing_key_error(self) -> LLMError:
        return LLMError(
            category=ErrorCategory.CONFIGURATION,
            message=f"Falta configurar la clave de API para {self.config.provider.value}.",
            provider=self.config.provider,
        )

    async def generate(self, request: ChatRequest) -> ModelResponse:
        """Genera una respuesta completa o devuelve un error controlado."""
        if self._client is None:
            return ModelResponse(
                provider=self.config.provider,
                model=self.config.model,
                error=self._missing_key_error(),
            )
        return await self._client.generate(request)

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Emite fragmentos asíncronos o un evento con error controlado."""
        if self._client is None:
            yield StreamChunk(error=self._missing_key_error())
            return
        async for chunk in self._client.stream(request):
            yield chunk

    async def aclose(self) -> None:
        """Libera los recursos HTTP del proveedor, si fueron inicializados."""
        if self._client is not None:
            await self._client.aclose()
