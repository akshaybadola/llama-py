from typing import Optional, AsyncGenerator
import asyncio
import json
from typing import AsyncGenerator
import sys

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import StreamingResponse, Response
from starlette.routing import Route

from . import llama


async def stream_response(request: Request) -> StreamingResponse:
    """
    Endpoint that streams tokens from the Llama model.
    """
    message = await request.json()
    #  get the llama interface from the app state.
    print(f"Got message {message}")
    iface = request.app.state.llama_interface
    request_id = iface.eval_message(message, stream=True)
    # TODO: It's an int right now
    if request_id is None:
        raise Exception("eval_message failed to return a request ID for streaming")

    async def generate_tokens() -> AsyncGenerator[str, None]:
        try:
            async for token in iface.receive_tokens():
                yield token + "\n\n"  # Add a newline for easier client handling
                print(token, end="")
                sys.stdout.flush()
        except KeyError as e:
            yield f"KeyError: {e}"
        except Exception as e:
            yield f"Exception: {e}"

    return StreamingResponse(generate_tokens(), media_type="text/plain",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def process_chat(iface, messages: list[dict[str, str]],
                       temperature: Optional[float] = None,
                       reset: bool = False) -> AsyncGenerator[str, None]:
    """
    Generates a mock streaming response.  Replace with your model logic.
    """
    # Need to format the prompt here
    sys.stdout.flush()
    if isinstance(messages[-1]["content"], dict):
        if messages[-1]["content"].keys() - {"text", "images"}:
            raise NotImplementedError("Only text and images implemented for now")
        prompt = messages[-1]["content"]
    elif isinstance(messages[-1]["content"], str):
        prompt = {"text": messages[-1]["content"], "images": []}
    else:
        raise NotImplementedError(f"Got bad message f{messages[-1]['content']}")
    add_bos = False
    if reset:
        iface.reset_context()
        add_bos = True
    print(f"prompt {prompt}, add_bos {add_bos}")
    sys.stdout.flush()
    iface.eval_message(prompt, stream=True, add_bos=add_bos)
    try:
        async for token in iface.receive_tokens():
            resp = {
                "choices": [
                    {
                        "delta": {"content": token},
                        "finish_reason": None,
                    }
                ],
            }
            yield json.dumps(resp)  # Add a newline for easier client handling
    except KeyError as e:
        yield f"KeyError: {e}"
        yield "[DONE]\n\n"
    except Exception as e:
        yield f"Exception: {e}"
        yield "[DONE]\n\n"


async def reset_context(request: Request) -> StreamingResponse:
    """
    Handles the /v1/chat/completions endpoint for streaming.
    """
    iface = request.app.state.llama_interface
    result = iface.reset_context()
    if not result:
        return Response("Successfully reset", status_code=200)
    else:
        return Response("Successfully reset", status_code=500)


async def chat(request: Request) -> StreamingResponse:
    """
    Handles the /v1/chat/completions endpoint for streaming.
    """
    iface = request.app.state.llama_interface
    try:
        body = await request.json()
        messages = body["messages"]
        stream = body.get("stream", True)
        temperature = body.get("temperature", 0.2)
        reset = body.get("reset", False)
    except Exception as e:
        async def error_generator(e):
            err = {'error': e}
            yield f"data: {json.dumps(err)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(error_generator(e), media_type="text/event-stream")

    async def generate() -> AsyncGenerator[str, None]:
        async for chunk in process_chat(iface, messages, temperature, reset):
            yield f"data: {chunk}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")


async def interrupt(request: Request) -> Response:
    request.app.state.llama_interface.interrupt()
    return Response("Interrupted")


async def is_generating(request: Request) -> Response:
    val = request.app.state.llama_interface.is_generating()
    return Response(str(val))


async def create_app(config, mock_llama_interface=None) -> Starlette:
    """
    Create the Starlette application.
    """
    app = Starlette(routes=[
        Route("/stream", stream_response, methods=["POST"]),
        Route("/completions", chat, methods=["POST"]),
        Route("/chat/completions", chat, methods=["POST"]),
        Route("/reset_context", reset_context, methods=["GET"]),
        Route("/interrupt", interrupt, methods=["GET"]),
        Route("/is_generating", is_generating, methods=["GET"]),
    ], debug=True)

    async def startup():
        if mock_llama_interface is not None:
            app.state.llama_interface = mock_llama_interface
        else:
            loop = asyncio.get_running_loop()
            app.state.llama_interface = llama.LlamaInterface(
                loop=loop,
                **config
            )

    app.add_event_handler("startup", startup)
    return app
