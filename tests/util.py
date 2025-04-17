import asyncio
from typing import AsyncGenerator


class MockLlamaInterface:
    def __init__(self):
        self.called_with = []

    def eval_message(self, message: dict[str, str | list[str]], stream=False):
        self.called_with.append(message)
        return 1

    async def receive_tokens(self) -> AsyncGenerator[str, None]:
        tokens = ["This", " ", "is", " ", "a", " ", "test", ".", "[EOS]"]
        for token in tokens:
            await asyncio.sleep(0.1)  # Simulate token delay
            yield token
