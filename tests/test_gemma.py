import os
import asyncio
from PIL import Image
from io import BytesIO
import base64
from hacky_llama import gemma_iface


def init_iface(lib_path):
    return gemma_iface.GemmaInterface(lib_path,
                                      os.path.expanduser("~/gemma-3-4b-it-q4_0.gguf"),
                                      os.path.expanduser("~/mmproj-model-f16-4B.gguf"),
                                      overrides={"n_gpu_layers": 100, "n_ctx": 8192, "n_batch": 8192},
                                      loop=None)


def eval_message(iface, n_predict, stream, stop_strings):
    msg_txt = """I am having trouble with some of the math in this <__image__>.
This is a about NonNegative Matrices. This is the next page <__image__>.

Can you extract the text from the images and explain the math? YOU MUST FORMAT your answer in LaTeX
for all the math symbols."""
    img_paths = [os.path.expanduser("~/nn_mat_test_1.png"),
                 os.path.expanduser("~/nn_mat_test_2.png")]
    message = {"text": msg_txt,
               "images": []}
    for img_path in img_paths:
        img = Image.open(img_path)
        img_bytes = BytesIO()
        img.save(img_bytes, format=img.format)
        img_str = base64.b64encode(img_bytes.getvalue())
        message["images"].append(img_str.decode())

    result = iface.eval_message(message, stream, add_bos=True, stop_strings=stop_strings)
    return result


async def _stream(iface):
    loop = asyncio.get_running_loop()
    iface.loop = loop

    async def receive_tokens(iface):
        while True:
            token = await iface.q.get()
            print(token, end="")

    asyncio.create_task(receive_tokens(iface))
    eval_message(iface, 1024, True, stop_strings=["```"])
    await asyncio.sleep(2)


def test_stream_with_images(iface):
    asyncio.run(_stream(iface))


def test_collect_with_images(iface):
    result = eval_message(iface, 1024, False, stop_strings=["```"])
    print(result)


def test_smol_prompt(iface, temp=0.2):
    with open(os.path.abspath(__file__).rsplit("/", 1)[0] + "/smol_prompt.md") as f:
        msg_text = f.read()
    message = {"text": msg_text,
               "images": []}
    stop_strings = ['<end_code>', 'Observation:', 'Calling tools:']
    result = iface.eval_message(message, False, add_bos=True, stop_strings=stop_strings,
                                sampler_params={"temp": temp})
    return result
