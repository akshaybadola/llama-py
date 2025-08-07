import pytest
import os
import asyncio
import yaml
import json
import sys
from threading import Thread

from hacky_llama import gemma_service
from hacky_llama.gemma_service import create_app, chat


from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient
import uvicorn

from util import MockLlamaInterface, fake_process_chat


port = int(os.environ.get("LLAMA_TEST_PORT") or 8001)


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
        uvicorn_config = uvicorn.Config(app=app, host="0.0.0.0", port=port, log_level="debug")
        server = uvicorn.Server(uvicorn_config)
        await server.serve()
    asyncio.run(run_app(config))


def test_run_fake_server():
    config = {"lib_path": None, "model_path": None, "mmproj_path": None, "overrides": None}
    gemma.chat = fake_process_chat

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
        uvicorn_config = uvicorn.Config(app=app, host="0.0.0.0", port=port, log_level="debug")
        server = uvicorn.Server(uvicorn_config)
        await server.serve()
    asyncio.run(run_app(config))


def run_server():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    iface_args = {"lib_path": config["lib_path"],
                  "model_path": config["model_root"] + config["model_path"],
                  "mmproj_path": config["model_root"] + config["mmproj_path"],
                  "overrides": config["overrides"]}

    async def run_app(config):
        app = await create_app(iface_args)
        uvicorn_config = uvicorn.Config(app=app, host="0.0.0.0", port=port, log_level="debug")
        server = uvicorn.Server(uvicorn_config)
        await server.serve()
    asyncio.run(run_app(config))


def test_mtmd_image_stream():
    msg_txt = """I am having trouble with some of the math in this (file:///home/joe/nn_mat_test_1.png).
    This is a about NonNegative Matrices. This is the next page (file:///home/joe/nn_mat_test_2.png).

    Can you extract the text from the images and explain the math? YOU MUST FORMAT your answer in LaTeX
    for all the math symbols."""

    import re
    import base64
    import time
    import httpx
    import json
    from io import BytesIO
    from PIL import Image

    port = 8001
    url = f"http://192.168.1.101:{port}/v1/chat/completions"

    async def _test_image_stream(url, msg_txt):
        imgs = re.findall(r"\(file://(/.+)\)", msg_txt)
        msgs = {"model": "unset", "stream": True,
                "messages": [{
                    "role": "user",
                    "content": None
                }]}
        imgs_data = []
        if imgs:
            for i, img_path in enumerate(imgs):
                img = Image.open(img_path)
                img_bytes = BytesIO()
                img.save(img_bytes, format=img.format)
                img_str = base64.b64encode(img_bytes.getvalue())
                imgs_data.append({"data": img_str.decode(), "id": i})
                msg_txt = re.sub(r"\(file://(/.+)\)", f"[img-{i}]", msg_txt, count=1)
        messages = {
            "model": "unset", "stream": True,
            "reset": True,
            "messages": [{
                    "role": "user",
                    "content": msg_txt,
                    "image_data": imgs_data
                }]
        }

        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, json=messages, timeout=None) as response:
                if response.status_code == 200:
                    start_time = time.time()
                    count = 0
                    async for chunk in response.aiter_text():
                        try:
                            if isinstance(chunk, bytes):
                                chunk = chunk.decode()
                            temp = json.loads(chunk.replace('\n', '', 1)[6:])
                            token = temp["choices"][0]["delta"]["content"]
                            print(token, end="", flush=True)
                            if "usage" in temp:
                                print("USAGE\n\n", temp["usage"])
                        except Exception:
                            print(temp, file=sys.stderr, flush=True)
                        # print(chunk.replace('\n', '', 1), end="", flush=True)
                        count += 1
                    end_time = time.time()
                    duration = end_time - start_time
                    print(f"Received {count} chunks in {duration:.2f} seconds")
                else:
                    print(f"Error: {response.status_code} - {(await response.aread()).decode()}")
    asyncio.run(_test_image_stream(url, msg_txt))


def test_real_server():
    import httpx
    import time

    t = Thread(target=run_server, daemon=True)
    t.start()

    def test_image_stream():
        msg_txt = """I am having trouble with some of the math in this (file:///home/joe/nn_mat_test_1.png).
        This is a about NonNegative Matrices. This is the next page (file:///home/joe/nn_mat_test_2.png).

        Can you extract the text from the images and explain the math? YOU MUST FORMAT your answer in LaTeX
        for all the math symbols."""

        from PIL import Image
        import re
        import base64
        from io import BytesIO

        port = 8001
        url = f"http://192.168.1.101:{port}/v1/chat/completions"

        async def _test_image_stream(url, msg_txt):
            imgs = re.findall(r"\(file://(/.+)\)", msg_txt)
            msg = re.sub(r"\(file://(/.+)\)", "<__image__>", msg_txt)
            imgs_data = []
            if imgs:
                for img_path in imgs:
                    img = Image.open(img_path)
                    img_bytes = BytesIO()
                    img.save(img_bytes, format=img.format)
                    img_str = base64.b64encode(img_bytes.getvalue())
                    imgs_data.append(img_str.decode())
            messages = {
                "model": "unset", "stream": True,
                "reset": True,
                "messages": [{"role": "user",
                              "content": {"text": msg,
                                         "images": imgs_data}}]
            }

            async with httpx.AsyncClient() as client:
                async with client.stream("POST", url, json=messages, timeout=None) as response:
                    if response.status_code == 200:
                        start_time = time.time()
                        count = 0
                        async for chunk in response.aiter_text():
                            print(chunk.replace('\n', '', 1), end="", flush=True)
                            count += 1
                        end_time = time.time()
                        duration = end_time - start_time
                        print(f"Received {count} chunks in {duration:.2f} seconds")
                    else:
                        print(f"Error: {response.status_code} - {(await response.aread()).decode()}")
        asyncio.run(_test_image_stream(url, msg_txt))

    def test_simple_msg(host="192.168.1.101", port=8000, msg=None, gpu_id=None):
        if gpu_id is None:
            url = f"http://{host}:{port}/chat/completions"
        else:
            url = f"http://{host}:{port}/{gpu_id}/chat/completions"
        msg = msg or "This is a test"

        async def oai_compat(url, msg):
            messages = {"model": "unset", "stream": True,
                        "messages": [{
                            "role": "user",
                            "content": msg
                        }]}
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", url, json=messages, timeout=None) as response:
                    if response.status_code == 200:
                        start_time = time.time()
                        count = 0
                        async for chunk in response.aiter_bytes():
                            try:
                                temp = json.loads(chunk.decode().replace('\n', '', 1)[6:])
                                token = temp["choices"][0]["delta"]["content"]
                                print(token, end="", flush=True)
                                if "usage" in temp:
                                    print("USAGE\n\n", temp["usage"])
                            except Exception:
                                print(temp, file=sys.stderr, flush=True)
                            count += 1
                        end_time = time.time()
                        duration = end_time - start_time
                        print(f"Received {count} chunks in {duration:.2f} seconds")
                    else:
                        print(f"Error: {response.status_code} - {(await response.aread()).decode()}")
        asyncio.run(oai_compat(url, msg))

    time.sleep(3)
    test_simple_msg(port=8001)
    # test_image_stream()
