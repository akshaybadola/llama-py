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
        self.llama = None
        self.process = None
        self.service_port = 8001
        self.service_url = f"http://localhost:{self.service_port}"
        self.config = config
        self.python = config["python"]
        self.start_process()

    def _print_stream(self, stream):
        while True:
            output = stream.readline()
            if len(output):
                print(output.strip())

    def _start_llama_process(self):
        """Starts the llama.cpp process."""
        cmd_args = ["--model_root", self.config["model_root"],
                    "--model_path", self.config["model_path"],
                    "--lib_path", self.config["lib_path"],
                    "--mmproj_path", self.config["mmproj_path"],
                    "--n_predict", str(self.config["n_predict"]),
                    "--port", str(self.service_port),
                    "--overrides", json.dumps(self.config["overrides"])]
        print(f"Starting process with python: {self.python} and args {cmd_args}")
        command = [self.python, "-u", "main.py", *cmd_args]
        logger.info(f"Starting llama.cpp process with command: {' '.join(command)}")
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True)
        self._print_stdout_thread = Thread(target=self._print_stream,
                                           args=[self.process.stdout])
        self._print_stderr_thread = Thread(target=self._print_stream,
                                           args=[self.process.stderr])
        self._print_stdout_thread.start()
        self._print_stderr_thread.start()

    def _start_llama_server_process(self):
        """Starts the :code:`llama-server` process."""
        llama_server_path = Path(self.config["lib_path"]).parent.joinpath("llama-server")
        more_args = []
        for k, v in self.config["overrides"].items():
            if v == True:
                more_args.append("--" + k.replace("_", "-"))
            else:
                more_args.extend(["--" + k.replace("_", "-"), str(v)])
        cmd_args = ["--model", str(Path(self.config["model_root"]).joinpath(self.config["model_path"])),
                    "--n-predict", str(self.config["n_predict"]),
                    "--port", str(self.service_port),
                    "--log-file", "~/logs/llama.log",
                    *more_args]
        print(f"Starting llama-server process with args {cmd_args}")
        command = [str(llama_server_path), *cmd_args]
        logger.info(f"Starting llama-server process: {' '.join(command)}")
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True)
        self._print_stdout_thread = Thread(target=self._print_stream,
                                           args=[self.process.stdout])
        self._print_stderr_thread = Thread(target=self._print_stream,
                                           args=[self.process.stderr])
        self._print_stdout_thread.start()
        self._print_stderr_thread.start()

    def start_process(self):
        if "gemma-3" in self.config["model_path"]:
            self.process_thread = Thread(target=self._start_llama_process)
        else:
            self.process_thread = Thread(target=self._start_llama_server_process)
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

    def load_model(self, new_config) -> bool:
        """Loads a new model."""
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
        self.stop_process()
        self.config.update(new_config)
        print(f"New config {self.config}")
        self.process = self.start_process()
        return True

    def reset_config(self):
        self.config = copy.deepcopy(self._initial_config)

    async def proxy_request(self, endpoint: str, request: Request):
        """Proxies a request to the service.py process."""
        url = f"{self.service_url}/{endpoint}"
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
                    resp = httpx.post(url, json=data, timeout=None)
                    return JSONResponse(resp.json(), status_code=200)
                else:
                    resp = httpx.post(url, json=data, timeout=2)
                    return JSONResponse(resp.json(), status_code=200)
            else:
                return JSONResponse({"Error": "Method not allowed"}, status_code=405)
        except Exception as e:
            logger.error(f"Error proxying request to service.py: {e}")
            return JSONResponse({"error": f"Failed to proxy request: {e}"}, status_code=500)

    async def interrupt(self, request: Request):
        if "gemma-3" in self.config["model_path"]:
            return await self.proxy_request("interrupt", request)
        else:
            self.process.send_signal(subprocess.signal.SIGINT)  # type: ignore
            return JSONResponse({"message": "interrupted"}, status_code=200)

    def list_models(self):
        models = glob.glob(self.config["model_root"] + "/*.gguf")
        return [Path(x).name for x in models if not Path(x).name.startswith("mmproj")]


def model_manager_app(config):
    """Starlette application for model management."""
    model_manager = ModelManager(config)

    # Model manager only endpoints
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
        return JSONResponse(model_manager.config, status_code=200)

    async def is_alive(request):
        if model_manager.process is None:
            msg = {"message": False}
        else:
            msg = {"message": model_manager.process.poll() is None}
        return JSONResponse(msg, status_code=200)

    async def reset_config(request):
        model_manager.reset_config()
        return JSONResponse({"message": "Reset Config"}, status_code=200)

    # custom server specific endpoints
    async def interrupt(request):
        return await model_manager.interrupt(request)

    async def is_generating(request):
        return await model_manager.proxy_request("is_generating", request)

    async def reset_context(request):
        return await model_manager.proxy_request("reset_context", request)

    # endpoints common to both custom and llama-server are proxied
    async def proxy_endpoint(request: Request):
        endpoint_name = request.path_params["endpoint_name"]
        return await model_manager.proxy_request(endpoint_name, request)

    routes = [
        Route("/list_models", endpoint=list_models, methods=["GET"]),
        Route("/switch_model", endpoint=switch_model, methods=["POST"]),
        Route("/model_info", endpoint=model_info, methods=["GET"]),
        Route("/is_alive", endpoint=is_alive, methods=["GET"]),
        Route("/reset_config", endpoint=reset_config, methods=["GET"]),

        Route("/reset_context", endpoint=reset_context, methods=["GET"]),
        Route("/interrupt", endpoint=interrupt, methods=["GET"]),
        Route("/is_generating", endpoint=is_generating, methods=["GET"]),

        Route("/{endpoint_name:path}", endpoint=proxy_endpoint, methods=["GET", "POST"]),
    ]

    app = Starlette(routes=routes, debug=True)
    app.state.model_manager = model_manager
    return app
