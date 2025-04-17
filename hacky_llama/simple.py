from typing import Optional, AsyncGenerator
import asyncio
import json
from typing import AsyncGenerator
import sys

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import StreamingResponse
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
                yield token + "\n"  # Add a newline for easier client handling
                print(token, end="")
                sys.stdout.flush()
        except KeyError as e:
            yield f"KeyError: {e}"
        except Exception as e:
            yield f"Exception: {e}"

    return StreamingResponse(generate_tokens(), media_type="text/plain",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def process_chat(iface, messages: list[dict[str, str]],
                       temperature: Optional[float] = None) -> AsyncGenerator[str, None]:
    """
    Generates a mock streaming response.  Replace with your model logic.
    """
    # Need to format the prompt here
    print("Got messages", messages)
    if isinstance(messages[-1]["content"], dict):
        if messages[-1]["content"].keys() - {"text", "images"}:
            raise NotImplementedError("Only text and images implemented for now")
        prompt = messages[-1]["content"]
    elif isinstance(messages[-1]["content"], str):
        prompt = {"text": messages[-1]["content"], "images": []}
    else:
        raise NotImplementedError(f"Got bad message f{messages[-1]['content']}")
    add_bos = len(messages) == 1
    request_id = iface.eval_message(prompt, stream=True, add_bos=add_bos)
    if request_id is None:
        raise Exception("eval_message failed to return a request ID for streaming")

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
            yield json.dumps(resp)  #  + "\n"  # Add a newline for easier client handling
        resp = {
            "choices": [
                {
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield json.dumps(resp)
        yield "[DONE]"
    except KeyError as e:
        yield f"KeyError: {e}"
    except Exception as e:
        yield f"Exception: {e}"

    # return StreamingResponse(generate_tokens(), media_type="application/json",
    #                          headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def mock_process_chat(iface, messages: list[dict[str, str]],
                            temperature: Optional[float] = None) -> AsyncGenerator[str, None]:
    """
    Generates a mock streaming response.  Replace with your model logic.
    """
    print("Got messages", messages)
    prompt = " ".join([m["content"] for m in messages])
    response_text = f"Mock response to: {prompt}. Temperature is {temperature}."
    words = response_text.split()
    for i, word in enumerate(words):
        if not i:
            resp = {
                "choices": [
                    {
                        "delta": {"role": "assistant", "content": word + " "},
                        "finish_reason": None,
                    }
                ],
            }
            print("Sending resp", resp)
            yield json.dumps(resp)
        else:
            resp = {
                "choices": [
                    {
                        "delta": {"content": word + " "},
                        "finish_reason": None,
                    }
                ],
            }
            print("Sending resp", resp)
            yield json.dumps(resp)
        await asyncio.sleep(0.1)

    resp = {
        "choices": [
            {
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    print("Sending resp", resp)
    yield json.dumps(resp)
    yield "[DONE]"



async def chat(request: Request) -> StreamingResponse:
    """
    Handles the /v1/chat/completions endpoint for streaming.
    """
    iface = request.app.state.llama_interface
    try:
        body = await request.json()
        messages = body["messages"]
        stream = body.get("stream", False)
        temperature = body.get("temperature", 0.2)
    except Exception as e:
        async def error_generator(e):
            err = {'error': e}
            yield f"data: {json.dumps(err)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(error_generator(e), media_type="text/event-stream")

    if not stream:
        async def non_stream_generator():
            full_response_text = ""
            async for chunk in process_chat(iface, messages, temperature):
                chunk_data = json.loads(chunk)
                if chunk_data["choices"][0]["finish_reason"] == "stop":
                    break
                if "content" in chunk_data["choices"][0]["delta"]:
                    full_response_text += chunk_data["choices"][0]["delta"]["content"]
            yield json.dumps({
                "choices": [{"message": {"role": "assistant", "content": full_response_text},
                             "finish_reason": "stop"}]
            })
        return StreamingResponse(non_stream_generator(), media_type="application/json")

    async def generate() -> AsyncGenerator[str, None]:
        async for chunk in process_chat(iface, messages, temperature):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")


# async def complete(request: Request) -> StreamingResponse:
#     """
#     Endpoint that streams tokens from the Llama model.
#     """
#     message = await request.json()
#     #  get the llama interface from the app state.
#     iface = request.app.state.llama_interface
#     request_id = iface.eval_message(message, stream=False)
#     # TODO: It's an int right now
#     if request_id is None:
#         raise Exception("eval_message failed to return a request ID for streaming")

#     async def generate_tokens() -> AsyncGenerator[str, None]:
#         try:
#             async for token in iface.receive_tokens():
#                 sys.stdout.flush() # ADDED THIS LINE
#                 yield token + "\n"  # Add a newline for easier client handling
#         except KeyError as e:
#             yield f"KeyError: {e}"
#         except Exception as e:
#             yield f"Exception: {e}"

#     return StreamingResponse(generate_tokens(), media_type="text/plain")



async def create_app(config, mock_llama_interface=None) -> Starlette:
    """
    Create the Starlette application.
    """
    app = Starlette(routes=[
        Route("/stream", stream_response, methods=["POST"]),
        Route("/completions", chat, methods=["POST"]),
        Route("/chat/completions", chat, methods=["POST"]),
    ])

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
