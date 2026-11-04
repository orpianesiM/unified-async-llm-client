import pytest
from pydantic import ValidationError

from unified_async_llm_client.schemas import (
    ChatMessage,
    ChatRequest,
    ErrorCategory,
    ModelConfig,
    Provider,
    StreamChunk,
)


def test_model_config_accepts_temperature_bounds() -> None:
    config = ModelConfig(provider=Provider.OPENAI, model="test-model", temperature=2)

    assert config.temperature == 2


@pytest.mark.parametrize("temperature", [-0.01, 2.01])
def test_model_config_rejects_temperature_out_of_range(temperature: float) -> None:
    with pytest.raises(ValidationError):
        ModelConfig(provider=Provider.OPENAI, model="test-model", temperature=temperature)


def test_chat_request_rejects_system_only_messages() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(messages=[ChatMessage(role="system", content="Instrucción")])


def test_stream_chunk_requires_exactly_one_payload() -> None:
    with pytest.raises(ValidationError):
        StreamChunk()


def test_stream_chunk_accepts_controlled_error() -> None:
    chunk = StreamChunk(
        error={
            "category": ErrorCategory.RATE_LIMIT,
            "message": "Reintenta más tarde.",
            "provider": Provider.OPENAI,
            "retryable": True,
        }
    )

    assert chunk.error is not None
    assert chunk.error.retryable is True
