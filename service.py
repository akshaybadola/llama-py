import asyncio
import json
from typing import AsyncGenerator, Optional, Any
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.routing import Route

import llama


class Service:
    def __init__(self):
        self.llama = llama
        self.routes = [
            Route("/v1/chat/completions", endpoint=self.chat, methods=["POST"]),
        ]
        self.app = Starlette(routes=self.routes)

    # Assuming model_callback is a globally defined function
    async def model_callback(self, token: str) -> None:
        """
        Mock function to simulate token generation.
        """
        await asyncio.sleep(0.01)
        # print(f"Callback: {token}")
        return


    async def process_chat(self, messages: list[dict[str, str]],
                           temperature: Optional[float] = None) -> AsyncGenerator[str, None]:
        """
        Generates a mock streaming response.  Replace with your model logic.
        """
        # Need to format the prompt here
        prompt = " ".join([m["content"] for m in messages])
        response_text = f"Mock response to: {prompt}. Temperature is {temperature}."
        words = response_text.split()
        for i, word in enumerate(words):
            await model_callback(word)
            if not i:
                yield json.dumps({
                    "choices": [
                        {
                            "delta": {"role": "assistant", "content": word},
                            "finish_reason": None,
                        }
                    ],
                })
            else:
                yield json.dumps({
                    "choices": [
                        {
                            "delta": {"content": word},
                            "finish_reason": None,
                        }
                    ],
                })
            await asyncio.sleep(0.1)

        yield json.dumps({
            "choices": [
                {
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        })
        yield "[DONE]"

    async def chat(self, request: Request) -> StreamingResponse:
        """
        Handles the /v1/chat/completions endpoint for streaming.
        """
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
                async for chunk in self.process_chat(messages, temperature):
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
            async for chunk in self.process_chat(messages, temperature):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")

    def start(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8000, reload=True)
