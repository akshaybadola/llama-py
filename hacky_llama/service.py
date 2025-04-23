import asyncio
import json
import subprocess
import sys
import os
import logging
import httpx

from typing import AsyncGenerator, Optional, Any
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse
from starlette.routing import Route

from . import simple


logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, config):
        self.config = config
        self.llama = None
        self.process = None
        self.service_url = "http://localhost:8000"  # Default URL, can be configurable
        self.config = config

    async def start_process(self):
        """Starts the llama.cpp process."""
        command = [
            "python",
            "main.py",  # Assuming service.py contains the Llama routes
            "--model_path", self.config["model_path"],
            "--lib_path", self.config["lib_path"],
            "--mmproj_path", self.config["mmproj_path"],
            "--n_predict", str(self.config["n_predict"]),
            "--port", str(8001),
            "--overrides", json.dumps(self.config["self.overrides"])
        ]
        logger.info(f"Starting llama.cpp process with command: {' '.join(command)}")
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return self.process

    async def stop_process(self):
        """Stops the llama.cpp process."""
        if self.process:
            logger.info("Stopping llama.cpp process...")
            self.process.terminate()  # Or .kill() if needed
            self.process.wait()
            self.process = None
            logger.info("llama.cpp process stopped.")

    async def load_model(self, new_config):
        """Loads a new model."""
        await self.stop_process()
        self.config.update(new_config)
        self.process = await self.start_process()
        await asyncio.sleep(5)  # Give it some time to load

    async def proxy_request(self, endpoint, request):
        """Proxies a request to the service.py process."""
        url = f"{self.service_url}/{endpoint}"
        headers = request.headers.copy()
        headers["Content-Type"] = request.content_type

        try:
            if request.method == "GET":
                async with httpx.AsyncClient() as client:
                    return await client.get(url, headers=headers, params=request.query_params)
            elif request.method == "POST":
                async with httpx.AsyncClient() as client:
                    data = await request.json()
                    async with client.stream("POST", url, json=data, timeout=None) as response:
                        if response.status_code == 200 and\
                           'Content-Type' in response.headers and\
                           response.headers['Content-Type'] == 'text/event-stream':
                            async def stream_generator():
                                async for chunk in response.aiter_text():
                                    if chunk:
                                        yield f"data: {chunk}\n\n"
                            return StreamingResponse(stream_generator(), media_type="text/event-stream")
                        elif response.status_code == 200:
                            data = await response.json()
                            return JSONResponse(data, staus_code=response.status_code)
            else:
                return JSONResponse(await response.json(), status_code=response.status_code)
        except Exception as e:
            logger.error(f"Error proxying request to service.py: {e}")
            return JSONResponse({"error": f"Failed to proxy request: {e}"}, status_code=500)


async def model_switch_endpoint(request, model_manager):
    """Endpoint to switch models."""
    params = await request.json()
    logger.info(f"Switching model with params: {params}")
    await model_manager.load_model(params)
    return JSONResponse({"message": "Model switched"})


def model_manager_app(config):
    """Starlette application for model management."""
    model_manager = ModelManager(config)

    app = Starlette()

    @app.get("/switch_model")
    async def switch_model(request):
        return await model_switch_endpoint(request, model_manager)

    # Proxy endpoints
    @app.get("/stream")
    async def stream(request):
        return await model_manager.proxy_request("stream", request)

    @app.post("/completions")
    async def completions(request):
        return await model_manager.proxy_request("completions", request)

    @app.post("/chat/completions")
    async def chat_completions(request):
        return await model_manager.proxy_request("chat/completions", request)

    @app.post("/reset_context")
    async def reset_context(request):
        return await model_manager.proxy_request("reset_context", request)

    @app.post("/interrupt")
    async def interrupt(request):
        return await model_manager.proxy_request("interrupt", request)

    @app.get("/is_generating")
    async def is_generating(request):
        return await model_manager.proxy_request("is_generating", request)

    app.state.model_manager = model_manager
    return app
