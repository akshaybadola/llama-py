from typing import Optional, AsyncGenerator
import asyncio
import time
import json
import sys

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse
from starlette.routing import Route

from .gemma_iface import GemmaInterface


async def stream_response(request: Request) -> StreamingResponse:
    """
    Endpoint that streams tokens from the Llama model.
    """
    message = await request.json()
    #  get the llama interface from the app state.
    print(f"Got message {message}")
    iface: GemmaInterface = request.app.state.llama_interface
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


def get_prompt_from_messages(messages):
    if isinstance(messages[-1]["content"], list):
        for m in messages[-1]["content"]:
            if m.keys() - {"type", "text", "images"}:
                raise NotImplementedError("Only text and images implemented for now")
        prompt = {"text": "\n\n".join([x["text"] for x in messages[-1]["content"]
                                       if x["type"] == "text"]),
                  "images": [x["image"] for x in messages[-1]["content"]
                             if x["type"] == "image"]}
    elif isinstance(messages[-1]["content"], dict):
        if messages[-1]["content"].keys() - {"type", "text", "images"}:
            raise NotImplementedError("Only text and images implemented for now")
        prompt = messages[-1]["content"]
    elif isinstance(messages[-1]["content"], str):
        prompt = {"text": messages[-1]["content"], "images": []}
    else:
        raise NotImplementedError(f"Got bad message f{messages[-1]['content']}")
    reset = len(messages) == 1
    return prompt, reset


async def stream_chat(iface: GemmaInterface, messages: list[dict[str, str]],
                      stop_strings: Optional[list[str]] = None,
                      reset: bool = False,
                      sampler_params: Optional[dict] = None) -> AsyncGenerator[str, None]:
    """
    Generates a mock streaming response.  Replace with your model logic.
    """
    prompt, _reset = get_prompt_from_messages(messages)
    reset = _reset or reset
    add_bos = False
    if reset:
        iface.reset_context()
        add_bos = True
    print(f"prompt {prompt}, add_bos {add_bos}")
    sys.stdout.flush()
    # Need to format the prompt here
    iface.eval_message(prompt, stream=True, add_bos=add_bos,
                       stop_strings=stop_strings,
                       sampler_params=sampler_params)
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


def complete_chat(iface: GemmaInterface, messages: list[dict[str, str]],
                  stop_strings: Optional[list[str]] = None,
                  reset: bool = False,
                  sampler_params: Optional[dict] = None) -> str:
    prompt, _reset = get_prompt_from_messages(messages)
    reset = _reset or reset
    add_bos = False
    if reset:
        iface.reset_context()
        add_bos = True
    sys.stdout.flush()
    return str(iface.eval_message(prompt, stream=False,
                                  add_bos=add_bos,
                                  stop_strings=stop_strings,
                                  sampler_params=sampler_params))


def get_usage_timings(iface: GemmaInterface):
    info = iface.info()
    process_time = iface.generation_start_time - iface.process_start_time
    generation_time = time.time() - iface.generation_start_time
    prompt_n = info.prompt_n
    predicted_n = info.predicted_n
    return {
        "usage": {
            "completion_tokens": predicted_n,
            "prompt_tokens": prompt_n,
            "total_tokens": prompt_n+predicted_n,
        },
        "timings": {
            "prompt_n": prompt_n,
            "prompt_ms": process_time*1000,
            "prompt_per_token_ms": process_time/prompt_n*1000,
            "prompt_per_second": 1/process_time*prompt_n,
            "predicted_n": predicted_n,
            "predicted_ms": generation_time*1000,
            "predicted_per_token_ms": generation_time/predicted_n*1000,
            "predicted_per_second": 1/generation_time*predicted_n,
        }
    }


async def chat(request: Request) -> StreamingResponse | JSONResponse:
    """
    Handles the chat endpoints
    :code:`/completions`
    :code:`/chat/completions`
    :code:`/v1/chat/completions`

    """
    iface: GemmaInterface = request.app.state.llama_interface
    try:
        body = await request.json()
        messages = body["messages"]
        stream = body.get("stream", False)
        stop_strings = body.get("stop", [])
        reset = body.get("reset", False)
    except Exception as e:
        async def error_generator(e):
            err = {'error': e}
            yield f"data: {json.dumps(err)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(error_generator(e), media_type="text/event-stream")

    sampler_params = {k: body.get(k)
                      for k in ["temperature", "top_k", "top_p", "min_p", "top_n_sigma"]
                      if body.get(k)}
    if "temperature" in sampler_params:
        sampler_params["temp"] = sampler_params.pop("temperature")

    async def generate() -> AsyncGenerator[str, None]:
        async for chunk in stream_chat(iface, messages,
                                       reset=reset,
                                       stop_strings=stop_strings,
                                       sampler_params=sampler_params):
            yield f"data: {chunk}\n\n"
    if stream:
        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        result = complete_chat(iface, messages,
                               reset=reset,
                               stop_strings=stop_strings,
                               sampler_params=sampler_params)
        return JSONResponse({"role": "assistant",
                             "choices": [
                                 {"message": {"content": result},
                                  "finish_reason": "stop",
                                  "index": 0,
                                  "logprobs": None,
                                  "refusal": None,
                                  "role": "assistant",
                                  "annotations": None,
                                  "audio": None,
                                  "function_call": None,
                                  "tool_calls": None}],
                             **get_usage_timings(iface)},
                            status_code=200)


async def reset_context(request: Request) -> JSONResponse:
    iface: GemmaInterface = request.app.state.llama_interface
    result = iface.reset_context()
    if not result:
        return JSONResponse({"message": "Successfully reset"}, status_code=200)
    else:
        return JSONResponse({"message": "Could not reset"}, status_code=500)


async def interrupt(request: Request) -> JSONResponse:
    request.app.state.llama_interface.interrupt()
    return JSONResponse({"message": "Interrupted"})


async def is_generating(request: Request) -> JSONResponse:
    val = request.app.state.llama_interface.is_generating()
    return JSONResponse({"message": val})


async def create_app(config, mock_llama_interface=None) -> Starlette:
    """
    Create the Starlette application.
    """
    app = Starlette(routes=[
        Route("/stream", stream_response, methods=["POST"]),
        Route("/completions", chat, methods=["POST"]),
        Route("/chat/completions", chat, methods=["POST"]),
        Route("/v1/chat/completions", chat, methods=["POST"]),
        Route("/reset_context", reset_context, methods=["GET"]),
        Route("/interrupt", interrupt, methods=["GET"]),
        Route("/is_generating", is_generating, methods=["GET"]),
    ], debug=True)

    async def startup():
        if mock_llama_interface is not None:
            app.state.llama_interface = mock_llama_interface
        else:
            loop = asyncio.get_running_loop()
            app.state.llama_interface = GemmaInterface(
                loop=loop,
                **config
            )

    app.add_event_handler("startup", startup)
    return app
