"""Implementaciones asíncronas para OpenAI y Anthropic."""

from collections.abc import AsyncIterator
from typing import Any

import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from unified_async_llm_client.base import BaseLLMClient
from unified_async_llm_client.schemas import (
    ChatRequest,
    ErrorCategory,
    LLMError,
    ModelConfig,
    ModelResponse,
    Provider,
    StreamChunk,
    TokenUsage,
)


def _error_from_exception(error: Exception, provider: Provider) -> LLMError:
    """Traduce errores de los SDK a una estructura estable."""
    rate_limit_errors = (openai.RateLimitError, anthropic.RateLimitError)
    connection_errors = (
        openai.APIConnectionError,
        openai.APITimeoutError,
        anthropic.APIConnectionError,
        anthropic.APITimeoutError,
    )
    authentication_errors = (openai.AuthenticationError, anthropic.AuthenticationError)
    status_errors = (openai.APIStatusError, anthropic.APIStatusError)

    if isinstance(error, rate_limit_errors):
        return LLMError(
            category=ErrorCategory.RATE_LIMIT,
            message="El proveedor alcanzó el límite de solicitudes. Intenta nuevamente más tarde.",
            provider=provider,
            retryable=True,
            status_code=429,
        )
    if isinstance(error, connection_errors):
        return LLMError(
            category=ErrorCategory.NETWORK,
            message="No fue posible conectar con el proveedor de LLM.",
            provider=provider,
            retryable=True,
        )
    if isinstance(error, authentication_errors):
        return LLMError(
            category=ErrorCategory.AUTHENTICATION,
            message="El proveedor rechazó la clave de API configurada.",
            provider=provider,
            status_code=getattr(error, "status_code", 401),
        )
    if isinstance(error, status_errors):
        status_code = error.status_code
        return LLMError(
            category=ErrorCategory.PROVIDER,
            message="El proveedor no pudo completar la solicitud.",
            provider=provider,
            retryable=status_code >= 500,
            status_code=status_code,
        )
    return LLMError(
        category=ErrorCategory.UNKNOWN,
        message="Ocurrió un error inesperado al comunicarse con el proveedor.",
        provider=provider,
    )


def _configuration_error(provider: Provider) -> LLMError:
    """Construye un error sin exponer valores sensibles."""
    return LLMError(
        category=ErrorCategory.CONFIGURATION,
        message=f"Falta configurar la clave de API para {provider.value}.",
        provider=provider,
    )


def _openai_messages(request: ChatRequest) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in request.messages]


def _anthropic_messages(request: ChatRequest) -> tuple[str | None, list[dict[str, str]]]:
    system_messages = [message.content for message in request.messages if message.role == "system"]
    conversation = [
        {"role": message.role, "content": message.content}
        for message in request.messages
        if message.role != "system"
    ]
    system = "\n\n".join(system_messages) or None
    return system, conversation


def _usage(input_tokens: int, output_tokens: int) -> TokenUsage:
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)


class OpenAIClient(BaseLLMClient):
    """Cliente basado en el SDK asíncrono oficial de OpenAI."""

    def __init__(self, config: ModelConfig, api_key: str) -> None:
        super().__init__(config)
        self._client = AsyncOpenAI(
            api_key=api_key,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )

    async def generate(self, request: ChatRequest) -> ModelResponse:
        try:
            completion = await self._client.chat.completions.create(
                model=self.config.model,
                messages=_openai_messages(request),
                temperature=self.config.temperature,
                max_completion_tokens=self.config.max_tokens,
            )
            content = completion.choices[0].message.content or ""
            usage = None
            if completion.usage is not None:
                usage = _usage(completion.usage.prompt_tokens, completion.usage.completion_tokens)
            return ModelResponse(
                provider=Provider.OPENAI,
                model=self.config.model,
                content=content,
                usage=usage,
            )
        except Exception as error:
            return ModelResponse(
                provider=Provider.OPENAI,
                model=self.config.model,
                error=_error_from_exception(error, Provider.OPENAI),
            )

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        try:
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=_openai_messages(request),
                temperature=self.config.temperature,
                max_completion_tokens=self.config.max_tokens,
                stream=True,
            )
            async for chunk in response:
                text = chunk.choices[0].delta.content
                if text:
                    yield StreamChunk(text=text)
        except Exception as error:
            yield StreamChunk(error=_error_from_exception(error, Provider.OPENAI))

    async def aclose(self) -> None:
        await self._client.close()


class AnthropicClient(BaseLLMClient):
    """Cliente basado en el SDK asíncrono oficial de Anthropic."""

    def __init__(self, config: ModelConfig, api_key: str) -> None:
        super().__init__(config)
        self._client = AsyncAnthropic(
            api_key=api_key,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )

    async def generate(self, request: ChatRequest) -> ModelResponse:
        system, messages = _anthropic_messages(request)
        try:
            response = await self._client.messages.create(
                model=self.config.model,
                system=system or anthropic.NOT_GIVEN,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            content = "".join(
                block.text for block in response.content if isinstance(getattr(block, "text", None), str)
            )
            return ModelResponse(
                provider=Provider.ANTHROPIC,
                model=self.config.model,
                content=content,
                usage=_usage(response.usage.input_tokens, response.usage.output_tokens),
            )
        except Exception as error:
            return ModelResponse(
                provider=Provider.ANTHROPIC,
                model=self.config.model,
                error=_error_from_exception(error, Provider.ANTHROPIC),
            )

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        system, messages = _anthropic_messages(request)
        try:
            async with self._client.messages.stream(
                model=self.config.model,
                system=system or anthropic.NOT_GIVEN,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            ) as response:
                async for text in response.text_stream:
                    if text:
                        yield StreamChunk(text=text)
        except Exception as error:
            yield StreamChunk(error=_error_from_exception(error, Provider.ANTHROPIC))

    async def aclose(self) -> None:
        await self._client.close()
