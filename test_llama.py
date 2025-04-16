import asyncio
import llama


def test_eval_message(iface, n_predict):
    msg_txt = """I am having trouble with some of the math in this <__image__>.
This is a about NonNegative Matrices. This is the next page <__image__>.

Can you extract the text from the images and explain the math? YOU MUST FORMAT your answer in LaTeX
for all the math symbols."""
    message = {"text": msg_txt,
               "images": ["/home/joe/nn_mat_test_1.png", "/home/joe/nn_mat_test_2.png"]}
    iface.eval_message(message, True)


async def main():
    loop = asyncio.get_running_loop()

    iface = llama.LlamaInterface("/home/joe/gemma-3-4b-it-q4_0.gguf",
                                 "/home/joe/mmproj-model-f16-4B.gguf",
                                 overrides={"n_gpu_layers": 100},
                                 loop=loop)

    async def receive_tokens(iface):
        while True:
            token = await iface.q.get()
            print(token, end="")

    asyncio.create_task(receive_tokens(iface))
    test_eval_message(iface, 1024)
    await asyncio.sleep(2)


asyncio.run(main())

