"""Ejecuta una pregunta de ejemplo en modo normal y streaming."""

import asyncio
import sys

from dotenv import load_dotenv

from unified_async_llm_client import AsyncLLMManager, ChatMessage, ChatRequest


async def main() -> None:
    load_dotenv()
    manager = AsyncLLMManager.from_env()
    mensaje = input("Escribe tu pregunta (Enter para usar la pregunta de ejemplo): ").strip()
    request = ChatRequest(
        messages=[
            ChatMessage(
                role="user",
                content=mensaje or "¿Qué es la entropía? Responde brevemente.",
            )
        ]
    )

    try:
        print(f"Proveedor: {manager.config.provider.value} | Modelo: {manager.config.model}")
        print("\nRespuesta normal:")
        response = await manager.generate(request)
        if response.error is not None:
            print(f"Error [{response.error.category.value}]: {response.error.message}")
        else:
            print(response.content)

        print("\nRespuesta en streaming:")
        async for chunk in manager.stream(request):
            if chunk.error is not None:
                print(f"\nError [{chunk.error.category.value}]: {chunk.error.message}", file=sys.stderr)
                break
            print(chunk.text, end="", flush=True)
        print()
    finally:
        await manager.aclose()


if __name__ == "__main__":
    asyncio.run(main())
