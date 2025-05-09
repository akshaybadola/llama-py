from typing import Optional, AsyncGenerator
import ctypes
from ctypes import c_int, create_string_buffer, POINTER, c_ubyte, cast
import time
from io import BytesIO
import json
import asyncio
import base64
import sys

from PIL import Image

from .lib import init_lib, TOKEN_CALLBACK


class LlamaInterface:
    def __init__(self, lib_path: str, model_path: str, mmproj_path: Optional[str] = None,
                 overrides: Optional[dict] = None, n_predict: int = 8192, loop=None):
        print("Loading library", lib_path)
        self.lib = init_lib(lib_path)
        overrides = overrides or {}
        # self.queues: dict[str, asyncio.Queue[str]] = {}
        self.q: asyncio.Queue[str] = asyncio.Queue()
        self.c_callback = TOKEN_CALLBACK(self.python_token_callback)
        self.is_multimodal = True
        self.temperature = 0.2
        if not mmproj_path:
            print("mmproj path not given. Only text input will be supported")
            self.is_multimodal = False
            mmproj_path = ""
        self.ctx = self.lib.gemma3_static_initialize(
            model_path.encode(),
            mmproj_path.encode(),
            json.dumps(overrides).encode()
        )
        self.n_predict = n_predict
        try:
            self.loop = loop or asyncio.get_running_loop()
        except Exception:
            print("Could not get event loop will run in sync mode")

    def interrupt(self):
        self.lib.gemma3_static_interrupt()

    def is_generating(self):
        return self.lib.gemma3_is_generating()

    def python_token_callback(self, token_ptr):
        token = ctypes.string_at(token_ptr).decode('utf-8')
        future = asyncio.run_coroutine_threadsafe(self.q.put(token), self.loop)
        future.add_done_callback(lambda f: f.exception() and print("Put failed:", f.exception()))

    def eval_message(self, message: dict[str, str | list[str]], stream=False, add_bos=False,
                     temperature: float = 0.2, stop_strings=None) -> int | str:
        if self.temperature != temperature:
            self.lib.re_init_sampler(json.dumps({"temperature": temperature}))
            self.temperature = temperature
        self.q = asyncio.Queue()
        msg_text = message["text"]
        msg_imgs = message["images"]
        stop_strings = stop_strings or []
        c_strings = (ctypes.c_char_p * len(stop_strings))()
        c_strings[:] = [s.encode('utf-8') for s in stop_strings]  # Encode to bytes
        if not self.is_multimodal:
            _ = self.lib.gemma3_static_eval_message_text_only(
                msg_text.encode(),  # type: ignore
                add_bos
            )
        else:
            image_data = []
            image_sizes = []
            if msg_imgs:
                for m in msg_imgs:
                    data = base64.b64decode(m)
                    image_data.append((c_ubyte * len(data)).from_buffer_copy(data))
                    image_sizes.append(len(data))
                num_images = len(image_data)
            else:
                num_images = 0
            # Create arrays for ctypes
            image_data_pointers_array_type = POINTER(c_ubyte) * num_images
            image_data_pointers = image_data_pointers_array_type(*(cast(img_data, POINTER(c_ubyte))
                                                                   for img_data in image_data))
            image_sizes_array_type = c_int * num_images
            image_sizes_array = image_sizes_array_type(*image_sizes)
            _ = self.lib.gemma3_static_eval_message_with_images(
                msg_text.encode(),  # type: ignore
                image_data_pointers,
                image_sizes_array,
                num_images,
                add_bos
            )
        if stream:
            self.loop.run_in_executor(
                None,
                lambda: self.lib.gemma3_static_stream_response(self.c_callback, self.n_predict,
                                                               c_strings,
                                                               c_int(len(c_strings)))
            )
            return 0
        buffer = create_string_buffer(self.n_predict)
        _ = self.lib.gemma3_static_collect_response(c_int(self.n_predict),
                                                    buffer,
                                                    c_int(self.n_predict * 8),
                                                    c_strings,
                                                    c_int(len(c_strings)))
        return buffer.value.decode()

    async def receive_tokens(self) -> AsyncGenerator[str, None]:
        """Receive tokens"""
        while True:
            token = await self.q.get()
            if token == "[EOS]":  # End-of-stream token
                print("Got [EOS] token")
                sys.stdout.flush()
                break
            yield token

    def reset_context(self):
        return self.lib.gemma3_static_reset()
