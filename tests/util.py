import asyncio
from typing import Optional, AsyncGenerator
import json

from starlette.requests import Request
from starlette.responses import StreamingResponse, Response


class MockLlamaInterface:
    def __init__(self, *args, **kwargs):
        self.called_with = []

    def eval_message(self, message: dict[str, str | list[str]], stream=False):
        self.called_with.append(message)
        return 1

    async def receive_tokens(self) -> AsyncGenerator[str, None]:
        tokens = ["This", " ", "is", " ", "a", " ", "test", ".", "[EOS]"]
        for token in tokens:
            await asyncio.sleep(0.1)  # Simulate token delay
            yield token


async def fake_process_chat(iface, messages: list[dict[str, str]],
                            temperature: Optional[float] = None,
                            reset: bool = False) -> AsyncGenerator[str, None]:
    """
    Generates a mock streaming response.  Replace with your model logic.
    """
    print(f"Got messages: {messages}, temp: {temperature}, reset: {reset}")
    if isinstance(messages[-1]["content"], dict):
        if messages[-1]["content"].keys() - {"text", "images"}:
            raise NotImplementedError("Only text and images implemented for now")
        prompt = messages[-1]["content"]
    elif isinstance(messages[-1]["content"], str):
        prompt = {"text": messages[-1]["content"], "images": []}
    else:
        raise NotImplementedError(f"Got bad message f{messages[-1]['content']}")
    print("PROMPT_TEXT", prompt["text"])
    response_text = f"Mock response to: {prompt['text']}. Temperature is {temperature}."
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
