"""Esquemas validados de entrada y salida."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Provider(StrEnum):
    """Proveedores de modelos compatibles."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ErrorCategory(StrEnum):
    """Categorías de error controladas por el cliente."""

    CONFIGURATION = "configuration"
    RATE_LIMIT = "rate_limit"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    PROVIDER = "provider"
    UNKNOWN = "unknown"


class ChatMessage(BaseModel):
    """Mensaje individual de una conversación."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    """Solicitud de generación con mensajes validados."""

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(min_length=1)

    @model_validator(mode="after")
    def require_conversational_message(self) -> "ChatRequest":
        if not any(message.role != "system" for message in self.messages):
            raise ValueError("La solicitud debe incluir al menos un mensaje user o assistant.")
        return self


class ModelConfig(BaseModel):
    """Parámetros validados para una llamada al modelo."""

    model_config = ConfigDict(extra="forbid")

    provider: Provider
    model: str = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=512, gt=0)
    timeout_seconds: float = Field(default=60, gt=0)
    max_retries: int = Field(default=2, ge=0)


class LLMError(BaseModel):
    """Error serializable y seguro para el consumidor."""

    category: ErrorCategory
    message: str
    provider: Provider
    retryable: bool = False
    status_code: int | None = None


class TokenUsage(BaseModel):
    """Conteo de tokens informado por el proveedor."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class ModelResponse(BaseModel):
    """Respuesta completa o error controlado de un proveedor."""

    provider: Provider
    model: str
    content: str = ""
    usage: TokenUsage | None = None
    error: LLMError | None = None


class StreamChunk(BaseModel):
    """Fragmento de texto o error ocurrido durante un stream."""

    text: str | None = None
    error: LLMError | None = None

    @model_validator(mode="after")
    def require_text_or_error(self) -> "StreamChunk":
        """Impide eventos vacíos o ambiguos."""
        if (self.text is None) == (self.error is None):
            raise ValueError("Un fragmento debe contener texto o un error, pero no ambos.")
        return self
