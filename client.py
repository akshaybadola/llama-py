import asyncio
import base64
import time
import re

from PIL import Image
import httpx


msg_txt = """I am having trouble with some of the math in this (file:///home/joe/test_nn_mat_1.png).
This is a about NonNegative Matrices. This is the next page (file:///home/joe/test_nn_mat_1.png).

Can you extract the text from the images and explain the math? YOU MUST FORMAT your answer in LaTeX
for all the math symbols."""


async def test_client():
    """
    Simple client to test the streaming endpoint.
    """
    url = "http://192.168.1.101:8000/stream"  # Adjust if needed
    imgs = re.findall(r"\(file://(/.+)\)", msg_txt)
    msg = re.sub(r"\(file://(/.+)\)", "<__image__>", msg_txt)
    message = {
        "text": msg,
        "images": imgs,
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=message, timeout=None)  # Use timeout=None for streaming
        if response.status_code == 200:
            start_time = time.time()
            count = 0
            async for chunk in response.aiter_text():
                print(f"Received chunk: {chunk.strip()}")  # Print each chunk
                count += 1
            end_time = time.time()
            duration = end_time - start_time
            print(f"Received {count} chunks in {duration:.2f} seconds")

        else:
            print(f"Error: {response.status_code} - {response.text}")



asyncio.run(test_client())
