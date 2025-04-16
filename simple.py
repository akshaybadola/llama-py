import asyncio
import json
from typing import AsyncGenerator
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.routing import Route
import uvicorn

import llama


async def stream_response(request: Request) -> StreamingResponse:
    """
    Endpoint that streams tokens from the Llama model.
    """
    message = await request.json()
    #  get the llama interface from the app state.
    iface = request.app.state.llama_interface
    request_id = iface.eval_message(message, stream=True)
    # TODO: It's an int right now
    if request_id is None:
        raise Exception("eval_message failed to return a request ID for streaming")

    async def generate_tokens() -> AsyncGenerator[str, None]:
        try:
            async for token in iface.receive_tokens():
                yield token + "\n"  # Add a newline for easier client handling
        except KeyError as e:
            yield f"KeyError: {e}"
        except Exception as e:
            yield f"Exception: {e}"

    return StreamingResponse(generate_tokens(), media_type="text/plain")



async def create_app() -> Starlette:
    """
    Create the Starlette application.
    """
    app = Starlette(routes=[
        Route("/stream", stream_response, methods=["POST"]),
    ])

    async def startup():
        """
        Initialize the Llama interface on startup.
        """
        loop = asyncio.get_running_loop()
        #  Moved initialization to startup
        app.state.llama_interface = llama.LlamaInterface(
            "/home/joe/gemma-3-4b-it-q4_0.gguf",  # Replace with your model path
            "/home/joe/mmproj-model-f16-4B.gguf",  # Replace with your mmproj path
            overrides={"n_gpu_layers": 100},       # Adjust as needed
            loop=loop,
        )

    app.add_event_handler("startup", startup)
    return app


if __name__ == "__main__":
    async def run_app():
        app = await create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    asyncio.run(run_app())
