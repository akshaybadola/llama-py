import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from starlette.testclient import TestClient
import uvicorn

from hacky_llama.simple import create_app

from util import MockLlamaInterface


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
