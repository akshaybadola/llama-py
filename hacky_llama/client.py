import asyncio
import time
import re
import base64
from io import BytesIO
from PIL import Image
import sys

import httpx

msg_txt = """I am having trouble with some of the math in this (file:///home/joe/nn_mat_test_1.png).
This is a about NonNegative Matrices. This is the next page (file:///home/joe/nn_mat_test_2.png).

Can you extract the text from the images and explain the math? YOU MUST FORMAT your answer in LaTeX
for all the math symbols."""

url = "http://192.168.1.101:8000/stream"


async def test_client(url):
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
    message = {
        "text": msg,
        "images": imgs_data,
    }

    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, json=message, timeout=None) as response:
            if response.status_code == 200:
                start_time = time.time()
                count = 0
                async for chunk in response.aiter_text():
                    print(chunk.replace('\n', '', 1), end="")
                    sys.stdout.flush()
                    count += 1
                end_time = time.time()
                duration = end_time - start_time
                print(f"Received {count} chunks in {duration:.2f} seconds")
            else:
                print(f"Error: {response.status_code} - {(await response.aread()).decode()}")


if __name__ == '__main__':
    asyncio.run(test_client(url=url))
