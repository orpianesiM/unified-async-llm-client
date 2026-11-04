"""Interfaz abstracta compartida por los proveedores."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from unified_async_llm_client.schemas import ChatRequest, ModelConfig, ModelResponse, StreamChunk


class BaseLLMClient(ABC):
    """Contrato común para generación normal y streaming."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    async def generate(self, request: ChatRequest) -> ModelResponse:
        """Genera una respuesta completa sin bloquear el event loop."""
        raise NotImplementedError

    @abstractmethod
    def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Emite fragmentos de texto o un error controlado."""
        raise NotImplementedError

    @abstractmethod
    async def aclose(self) -> None:
        """Cierra el cliente HTTP asíncrono del proveedor."""
        raise NotImplementedError
