import pytest
import asyncio
from typing import Optional, AsyncGenerator

from hacky_llama.gemma import create_app, chat
from hacky_llama import gemma

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient
import uvicorn

from util import MockLlamaInterface, fake_process_chat


@pytest.mark.asyncio
async def test_stream_with_injected_mock():
    app = await create_app(mock_llama_interface=MockLlamaInterface())

    with TestClient(app) as client:
        response = client.post("/stream", json={
            "text": "Hello",
            "images": []
        })

        content = "".join(chunk for chunk in response.iter_text())
        assert "this" in content
        assert "is" in content


def run_test_server():
    config = {"lib_path": None, "model_path": None, "mmproj_path": None, "overrides": None}

    async def run_app(config):
        app = await create_app(config, mock_llama_interface=MockLlamaInterface())
        uvicorn_config = uvicorn.Config(app=app, host="0.0.0.0", port=8000, log_level="debug")
        server = uvicorn.Server(uvicorn_config)
        await server.serve()
    asyncio.run(run_app(config))


def test_run_fake_server():
    config = {"lib_path": None, "model_path": None, "mmproj_path": None, "overrides": None}
    gemma.process_chat = fake_process_chat

    async def create_app(config, mock_llama_interface=None) -> Starlette:
        """
        Create the Starlette application.
        """
        app = Starlette(routes=[
            Route("/completions", chat, methods=["POST"]),
            Route("/chat/completions", chat, methods=["POST"]),
            Route("/reset_context", gemma.reset_context, methods=["GET"]),
        ])

        async def startup():
            if mock_llama_interface is not None:
                app.state.llama_interface = mock_llama_interface
            else:
                loop = asyncio.get_running_loop()
                app.state.llama_interface = MockLlamaInterface(
                    loop=loop,
                    **config
                )

        app.add_event_handler("startup", startup)
        return app

    async def run_app(config):
        app = await create_app(config, mock_llama_interface=MockLlamaInterface())
        uvicorn_config = uvicorn.Config(app=app, host="0.0.0.0", port=8000, log_level="debug")
        server = uvicorn.Server(uvicorn_config)
        await server.serve()
    asyncio.run(run_app(config))
