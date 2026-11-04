# Unified Async LLM Client

Cliente Python 3.12 con interfaz asíncrona común para OpenAI y Anthropic. Valida mensajes y parámetros con Pydantic, soporta generación normal y streaming, y convierte fallos del proveedor en respuestas/eventos de error estructurados.

## Requisitos

- Python 3.12
- Una clave de API para OpenAI o Anthropic

## Instalación

Desde PowerShell, en la carpeta del proyecto:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
Copy-Item .env.example .env
```

Edita `.env` y completa la clave del proveedor que vas a usar. No compartas ese archivo: está excluido por `.gitignore`.

## Configuración

| Variable | Descripción | Valor predeterminado |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` o `anthropic` | `openai` |
| `LLM_MODEL` | Identificador del modelo | `gpt-4o-mini` o `claude-3-5-haiku-latest` según el proveedor |
| `LLM_TEMPERATURE` | Temperatura entre 0 y 2 | `0.7` |
| `LLM_MAX_TOKENS` | Límite de tokens de salida, mayor que 0 | `512` |
| `LLM_TIMEOUT_SECONDS` | Timeout HTTP en segundos | `60` |
| `LLM_MAX_RETRIES` | Reintentos del SDK | `2` |
| `OPENAI_API_KEY` | Clave de OpenAI | Sin valor |
| `ANTHROPIC_API_KEY` | Clave de Anthropic | Sin valor |

## Ejecutar

```powershell
python main.py
```

El script pregunta “¿Qué es la entropía?” en modo normal y luego en streaming (si es que no se ingresó una pregunta). Selecciona el proveedor con `LLM_PROVIDER`.

## Usar desde Python

```python
import asyncio

from unified_async_llm_client import AsyncLLMManager, ChatMessage, ChatRequest


async def main() -> None:
    manager = AsyncLLMManager.from_env()
    solicitud = ChatRequest(messages=[ChatMessage(role="user", content="Hola")])
    try:
        respuesta = await manager.generate(solicitud)
        if respuesta.error:
            print(respuesta.error.category, respuesta.error.message)
        else:
            print(respuesta.content)

        async for fragmento in manager.stream(solicitud):
            if fragmento.error:
                print(fragmento.error.message)
                break
            print(fragmento.text, end="", flush=True)
    finally:
        await manager.aclose()


asyncio.run(main())
```

`generate()` devuelve un `ModelResponse`; `stream()` emite `StreamChunk` con texto o un `LLMError` tipado. Las categorías incluyen autenticación, red, límite de tasa, configuración y error del proveedor. El SDK aplica los reintentos configurados y los errores finales no rompen el loop principal.

## Pruebas

```powershell
pytest
```

Las pruebas de esquemas no requieren conexión ni claves de API.
