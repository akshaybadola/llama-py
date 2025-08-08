from typing import Optional, Any
import json
import subprocess
import logging
import copy
from pathlib import Path
from threading import Thread
import re
import glob

import httpx

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse, Response
from starlette.routing import Route
from starlette.background import BackgroundTask


logger = logging.getLogger(__name__)


async def stream_response(upstream_url: str, data):
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", upstream_url, json=data, timeout=None) as response:
            async for chunk in response.aiter_bytes():
                yield chunk


class ModelManager:
    def __init__(self, config):
        self._initial_config = config
        self.config = copy.deepcopy(self._initial_config)
        self.processes: dict[int, dict[str, Any]] = {}
        self.service_url_base = "http://localhost"
        self.python = config["python"]
        self.gpus = list(filter(lambda x: isinstance(x, int), self.config.keys()))
        self.use_multiple_models = config["use_multiple_models"]

        self.port_base = 8001
        self.ports = {}
        if self.use_multiple_models:
            for i in self.gpus:
                self.ports[i] = self.port_base + i
                self.start_process(i)
        else:
            self.start_process(0)

    def _print_stream(self, stream):
        while True:
            output = stream.readline()
            if len(output):
                print(output.strip())

    def _start_llama_process(self, model_config, gpu_id: Optional[int] = None):
        """Starts a llama.cpp process on a specific GPU."""
        if gpu_id is not None:
            return "Error"
        port = self.ports[gpu_id]
        cmd_args = [
            "--model_root", model_config["model_root"],
            "--model_path", model_config["model_path"],
            "--lib_path", model_config["lib_path"],
            "--mmproj_path", model_config["mmproj_path"],
            "--n_predict", str(model_config["n_predict"]),
            "--port", str(port),
            "--overrides", json.dumps(model_config["overrides"])
        ]
        print(f"Starting llama.cpp process on GPU {gpu_id} with args {cmd_args}")
        command = [self.python, "-u", "main.py", *cmd_args]
        logger.info(f"Starting llama.cpp process: {' '.join(command)}")
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)

        thread_stdout = Thread(target=self._print_stream, args=[process.stdout])
        thread_stderr = Thread(target=self._print_stream, args=[process.stderr])
        thread_stdout.start()
        thread_stderr.start()

        self.processes[gpu_id] = {
            "process": process,
            "port": port,
            "thread_stdout": thread_stdout,
            "thread_stderr": thread_stderr
        }

    def _start_llama_server_process(self, model_config, gpu_id: Optional[int] = None):
        """Starts a llama-server process on a specific GPU."""
        gpu_id = gpu_id or 0
        port = self.ports[gpu_id]
        llama_server_path = Path(self.config["lib_path"]).parent.joinpath("llama-server")
        model_path = str(Path(self.config["model_root"]).joinpath(model_config["model_path"]))

        # Build args
        args = [
            "--model", model_path,
            "--n-predict", str(model_config["n_predict"]),
            "--port", str(port),
            "--log-file", "~/logs/llama.log",
        ]

        for k, v in model_config["overrides"].items():
            if v is True:
                args.append(f"--{k.replace('_', '-')}")
            else:
                args.extend([f"--{k.replace('_', '-')}", str(v)])

        if "--device" in args:
            logger.info(f"Will launch on GPU {gpu_id}")
        else:
            logger.info(f"Will launch on GPU {gpu_id}")
            args.extend(["--device", f"CUDA{gpu_id}"])
        command = [str(llama_server_path), *args]
        logger.info(f"Starting llama-server: {' '.join(command)}")
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)

        thread_stdout = Thread(target=self._print_stream, args=[process.stdout])
        thread_stderr = Thread(target=self._print_stream, args=[process.stderr])
        thread_stdout.start()
        thread_stderr.start()

        self.processes[gpu_id] = {
            "process": process,
            "port": port,
            "thread_stdout": thread_stdout,
            "thread_stderr": thread_stderr
        }

    def start_process(self, gpu_id):
        print(f"Launching process for {gpu_id}")
        if self.use_multiple_models:
            model_config = self.config[gpu_id]
            if "gemma" in model_config["model_path"].lower():
                self._start_llama_process(model_config, gpu_id)
            else:
                self._start_llama_server_process(model_config, gpu_id)
        else:
            model_config = self.config["default"]
            if "gemma" in model_config["model_path"]:
                self._start_llama_process(model_config)
            else:
                self._start_llama_server_process(model_config)

    def stop_process(self, gpu_id):
        data = self.processes[gpu_id]
        if data["process"]:
            logger.info(f"Stopping process on GPU {gpu_id}")
            data["process"].terminate()
            data["process"].wait()
            data["process"] = None
            # Clean up threads
        if data["thread_stdout"]:
            data["thread_stdout"].join()
        if data["thread_stderr"]:
            data["thread_stderr"].join()

    def load_model(self, new_config) -> bool:
        """Load a new model (or multiple models)."""
        if self.use_multiple_models and "gpu" not in new_config:
            print("Bad new config")
            return False
        elif self.use_multiple_models and "gpu" in new_config:
            gpu = new_config.pop("gpu")
        else:
            gpu = 0
        if model_name := new_config.get("model_name"):
            model_list = self.list_models()
            matches = list(filter(lambda x: re.match(".+" + model_name + ".+", x, flags=re.IGNORECASE),
                                  model_list))
            if matches:
                new_config.pop("model_name")
                new_config["model_path"] = matches[0]
        elif "model_path" not in new_config:
            print("Bad new config")
            return False
        if not self.use_multiple_models:
            self.config["default"].update(new_config)
        else:
            self.config[gpu].update(new_config)
        print(f"New config for device {gpu}: {self.config[gpu]}")
        self.stop_process(gpu)
        self.start_process(gpu)
        return True

    def reset_config(self, gpu_id: Optional[int] = None):
        self.config = copy.deepcopy(self._initial_config)
        if self.use_multiple_models and gpu_id:
            self.stop_process(gpu_id)
            self.start_process(gpu_id)
        else:
            self.stop_process(0)
            self.start_process(0)

    def list_models(self):
        models = glob.glob(self.config["model_root"] + "/*.gguf")
        return [Path(x).name for x in models if not Path(x).name.startswith("mmproj")]

    def get_service_url(self, gpu_id=None):
        if gpu_id is None:
            port = self.port_base
        else:
            port = self.ports.get(gpu_id, self.port_base)
        return f"http://localhost:{port}"

    async def proxy_request(self, endpoint: str, request: Request, gpu_id: Optional[int] = None):
        """Proxies request to specific GPU model."""
        url = self.get_service_url(gpu_id) + f"/{endpoint}"

        headers = request.headers.mutablecopy()
        try:
            if request.method == "GET":
                resp = httpx.get(url, headers=headers, params=request.query_params)
                if not endpoint:
                    return Response(resp.content.decode(), status_code=200)
                else:
                    return JSONResponse(resp.json(), headers=resp.headers, status_code=200)
            elif request.method == "POST":
                data = await request.json()
                if endpoint == "stream" or\
                   endpoint in {"completions", "chat/completions", "v1/chat/completions"} and\
                   data.get("stream"):
                    return StreamingResponse(stream_response(url, data),
                                             background=BackgroundTask(lambda: None),
                                             media_type="text/event-stream")
                elif endpoint in {"completions", "chat/completions", "v1/chat/completions"}:
                    async with httpx.AsyncClient() as client:
                        resp = await client.post(url, json=data, timeout=None)
                        return JSONResponse(resp.json(), status_code=200)
                else:
                    async with httpx.AsyncClient() as client:
                        resp = await client.post(url, json=data, timeout=2)
                        return JSONResponse(resp.json(), status_code=200)
            else:
                return JSONResponse({"Error": "Method not allowed"}, status_code=405)
        except Exception as e:
            logger.error(f"Error proxying request to service.py: {e}")
            return JSONResponse({"error": f"Failed to proxy request: {e}"}, status_code=500)

    async def interrupt(self, request: Request, gpu_id=None):
        if gpu_id is not None:
            if self.processes.get(gpu_id) and self.processes[gpu_id]["process"]:
                self.processes[gpu_id]["process"].send_signal(subprocess.signal.SIGINT)
                return JSONResponse({"message": f"Interrupted GPU {gpu_id}"}, status_code=200)
            else:
                return JSONResponse({"error": "No process running on this GPU"}, status_code=404)
        else:
            # Interrupt all models
            for gpu_id in self.processes:
                if self.processes[gpu_id]["process"]:
                    self.processes[gpu_id]["process"].send_signal(subprocess.signal.SIGINT)
            return JSONResponse({"message": "Interrupted all models"}, status_code=200)


def model_manager_app(config):
    model_manager = ModelManager(config)

    async def list_models(request):
        return JSONResponse(model_manager.list_models(), status_code=200)

    async def switch_model(request):
        params = await request.json()
        logger.info(f"Switching model with params: {params}")
        status = model_manager.load_model(params)
        if status:
            return JSONResponse({"message": "Model switch initiated"}, status_code=200)
        else:
            return JSONResponse({"message": "Bad params"}, status_code=400)

    async def model_info(request):
        gpu_id = request.path_params.get("gpu_id")
        if not model_manager.use_multiple_models:
            return JSONResponse(model_manager.config["default"], status_code=200)
        elif model_manager.use_multiple_models and gpu_id:
            return JSONResponse(model_manager.config[gpu_id], status_code=200)
        else:
            return JSONResponse([model_manager.config[i]
                                 for i in model_manager.gpus], status_code=200)

    async def is_alive(request):
        if not model_manager.processes:
            msg = {"message": False}
        else:
            msg = {"message": [p["process"].poll() is None
                               for p in model_manager.processes.values()]}
        return JSONResponse(msg, status_code=200)

    async def reset_config(request):
        gpu_id = request.path_params.get("gpu_id")
        if not model_manager.use_multiple_models:
            model_manager.reset_config(0)
            return JSONResponse({"message": "Reset Config"}, status_code=200)
        else:
            if gpu_id is None:
                return JSONResponse({"message": "Need gpu_id with mulitple models"}, status_code=400)
            else:
                model_manager.reset_config(gpu_id)
                return JSONResponse({"message": f"Reset Config for {gpu_id}"}, status_code=200)

    async def proxy_endpoint(request: Request):
        endpoint_name = request.path_params["endpoint_name"]
        gpu_id = request.path_params.get("gpu_id")
        return await model_manager.proxy_request(endpoint_name, request, gpu_id=gpu_id)

    async def interrupt(request: Request, gpu_id: Optional[int] = None):
        gpu_id = request.path_params.get("gpu_id")
        return await model_manager.interrupt(request, gpu_id=gpu_id)

    async def is_generating(request: Request, gpu_id: Optional[int] = None):
        return await model_manager.proxy_request("is_generating", request, gpu_id=gpu_id)

    async def reset_context(request: Request, gpu_id: Optional[int] = None):
        return await model_manager.proxy_request("reset_context", request, gpu_id=gpu_id)

    routes = [
        Route("/list_models", endpoint=list_models, methods=["GET"]),
        Route("/switch_model", endpoint=switch_model, methods=["POST"]),
        Route("/model_info", endpoint=model_info, methods=["GET"]),
        Route("/is_alive", endpoint=is_alive, methods=["GET"]),
        Route("/reset_config", endpoint=reset_config, methods=["GET"]),
        Route("/interrupt", endpoint=interrupt, methods=["GET"]),
        Route("/is_generating", endpoint=is_generating, methods=["GET"]),
        Route("/reset_context", endpoint=reset_context, methods=["GET"]),
        Route("/{gpu_id:int}/{endpoint_name:path}", endpoint=proxy_endpoint, methods=["GET", "POST"]),
    ]

    app = Starlette(routes=routes, debug=True)
    app.state.model_manager = model_manager
    return app
