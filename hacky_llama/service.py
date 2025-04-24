import asyncio
import json
import subprocess
import sys
import os
import logging
import httpx
from threading import Thread

from typing import AsyncGenerator, Optional, Any
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse
from starlette.routing import Route
from starlette.background import BackgroundTask

from . import simple


logger = logging.getLogger(__name__)


async def stream_response(upstream_url: str, data):
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", upstream_url, json=data, timeout=None) as response:
            async for chunk in response.aiter_bytes():
                yield chunk


class ModelManager:
    def __init__(self, config):
        self.config = config
        self.llama = None
        self.process = None
        self.service_port = 8001
        self.service_url = f"http://localhost:{self.service_port}"
        self.config = config
        self.python = config["python"]
        self.start_process()

    def _start_process(self):
        """Starts the llama.cpp process."""
        print("Starting process")
        command = [
            self.python,
            "main.py",  # Assuming service.py contains the Llama routes
            "--model_path", self.config["model_path"],
            "--lib_path", self.config["lib_path"],
            "--mmproj_path", self.config["mmproj_path"],
            "--n_predict", str(self.config["n_predict"]),
            "--port", str(self.service_port),
            "--overrides", json.dumps(self.config["overrides"])
        ]
        logger.info(f"Starting llama.cpp process with command: {' '.join(command)}")
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = self.process.communicate()

    def start_process(self):
        self.process_thread = Thread(target=self._start_process)
        self.process_thread.daemon = True
        self.process_thread.start()

    def stop_process(self):
        """Stops the llama.cpp process."""
        if self.process:
            logger.info("Stopping llama.cpp process...")
            self.process.terminate()  # Or .kill() if needed
            self.process.wait()
            self.process = None
            logger.info("llama.cpp process stopped.")

    def load_model(self, new_config):
        """Loads a new model."""
        self.stop_process()
        self.config.update(new_config)
        self.process = self.start_process()

    async def proxy_request(self, endpoint: str, request: Request):
        """Proxies a request to the service.py process."""
        url = f"{self.service_url}/{endpoint}"
        headers = request.headers.mutablecopy()
        try:
            if request.method == "GET":
                async with httpx.AsyncClient() as client:
                    return await client.get(url, headers=headers, params=request.query_params)
            elif request.method == "POST":
                if endpoint in {"stream", "completions", "chat/completions"}:
                    data = await request.json()
                    return StreamingResponse(stream_response(url, data),
                                             background=BackgroundTask(lambda: None),
                                             media_type="text/event-stream")
                else:
                    async with httpx.AsyncClient() as client:
                        data = await request.json()
                        return client.post(url, json=data, timeout=2)
            else:
                return JSONResponse({"Error": "Method not allowed"}, status_code=405)
        except Exception as e:
            logger.error(f"Error proxying request to service.py: {e}")
            return JSONResponse({"error": f"Failed to proxy request: {e}"}, status_code=500)


async def model_switch_endpoint(request, model_manager):
    """Endpoint to switch models."""
    params = await request.json()
    logger.info(f"Switching model with params: {params}")
    model_manager.load_model(params)
    return JSONResponse({"message": "Model switched"})


def model_manager_app(config):
    """Starlette application for model management."""
    model_manager = ModelManager(config)

    async def switch_model(request):
        return await model_switch_endpoint(request, model_manager)

    async def stream(request):
        return await model_manager.proxy_request("stream", request)

    async def model_info(request):
        return JSONResponse(model_manager.config, status_code=200)

    async def completions(request):
        return await model_manager.proxy_request("completions", request)

    async def chat_completions(request):
        return await model_manager.proxy_request("chat/completions", request)

    async def reset_context(request):
        return await model_manager.proxy_request("reset_context", request)

    async def interrupt(request):
        return await model_manager.proxy_request("interrupt", request)

    async def is_generating(request):
        return await model_manager.proxy_request("is_generating", request)

    routes = [
        Route("/switch_model", endpoint=switch_model, methods=["POST"]),
        Route("/stream", endpoint=stream, methods=["POST"]),
        Route("/completions", endpoint=completions, methods=["POST"]),
        Route("/chat/completions", endpoint=chat_completions, methods=["POST"]),
        Route("/reset_context", endpoint=reset_context, methods=["GET"]),
        Route("/model_info", endpoint=model_info, methods=["GET"]),
        Route("/interrupt", endpoint=interrupt, methods=["GET"]),
        Route("/is_generating", endpoint=is_generating, methods=["GET"]),
    ]

    app = Starlette(routes=routes, debug=True)
    app.state.model_manager = model_manager
    return app
