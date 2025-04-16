from typing import Optional, AsyncGenerator
import ctypes
from ctypes import (cdll, c_void_p, c_char_p, c_int, create_string_buffer, CFUNCTYPE,
                    POINTER, c_ubyte, cast, Structure, Array, c_voidp)
import time
from io import BytesIO
import json
import asyncio
import base64

from PIL import Image

lib = cdll.LoadLibrary("/home/joe/projects/llama.cpp/build/bin/libgemma3.so")

lib.gemma3_create_params.argtypes = []
lib.gemma3_create_params.restype = c_void_p

lib.gemma3_create_params_with_overrides.argtypes = [
    c_char_p                    # json data of dict
]
lib.gemma3_create_params_with_overrides.restype = c_void_p


lib.gemma3_create_sampler.argtypes = [
    c_void_p,                   # ctx_ptr
    c_void_p                    # params_ptr
]
lib.gemma3_create_sampler.restype = c_void_p

lib.gemma3_create_context.argtypes = [
    c_char_p,                   # model_path
    c_char_p                    # mmproj_path
]
lib.gemma3_create_context.restype = c_void_p

lib.gemma3_eval_message.argtypes = [
    c_void_p,                   # ctx_ptr
    c_char_p                    # msg_str
]
lib.gemma3_eval_message.restype = c_int

lib.gemma3_generate_response.argtypes = [
    c_void_p,                   # ctx_ptr
    c_void_p,                   # msg_str
    c_int                       # n_predict
]
lib.gemma3_generate_response.restype = c_int

lib.gemma3_collect_response.argtypes = [
    c_void_p,                   # ctx_ptr
    c_void_p,                   # msg_str
    c_int,                      # n_predict
    c_char_p,                   # result buffer
    c_int                       # n_generations
]
lib.gemma3_collect_response.restype = c_int

TOKEN_CALLBACK = CFUNCTYPE(None, c_char_p)

lib.gemma3_stream_response.argtypes = [
    c_void_p,                   # ctx_ptr
    c_void_p,                   # msg_str
    c_int,                      # n_predict
    TOKEN_CALLBACK              # Callback function
]
lib.gemma3_stream_response.restype = c_int


lib.gemma3_eval_message_with_images.argtypes = [
    c_void_p,                   # ctx_ptr
    c_char_p,                   # msg_str
    POINTER(POINTER(c_ubyte)),  # Array of pointers to image data
    POINTER(c_int),             # Array of image data sizes
    c_int                       # Number of images
]
lib.gemma3_eval_message_with_images.restype = c_int

# static initialize
lib.gemma3_static_initialize.argtypes = [
    c_char_p,                   # model_path
    c_char_p,                   # mmproj_path
    c_char_p                    # overrides
]
lib.gemma3_static_initialize.restype = c_void_p

# static eval message
lib.gemma3_static_eval_message.argtypes = [
    c_char_p                   # model_path
]
lib.gemma3_static_eval_message.restype = c_int

# static eval message with images
lib.gemma3_static_eval_message_with_images.argtypes = [
    c_char_p,                   # msg_str
    POINTER(POINTER(c_ubyte)),  # Array of pointers to image data
    POINTER(c_int),             # Array of image data sizes
    c_int                       # Number of images
]
lib.gemma3_eval_message_with_images.restype = c_int

# static generate response
lib.gemma3_static_generate_response.argtypes = [
    c_int                       # model_path
]
lib.gemma3_static_generate_response.restype = c_int

# static stream response
lib.gemma3_static_stream_response.argtypes = [
    TOKEN_CALLBACK,             # Callback function
    c_int                       # n_predict
]
lib.gemma3_static_stream_response.restype = c_int


class LlamaInterface:
    def __init__(self, model_path: str, mmproj_path: str,
                 overrides: Optional[dict] = None, loop=None):
        overrides = overrides or {}
        # self.queues: dict[str, asyncio.Queue[str]] = {}
        self.q: asyncio.Queue[str] = asyncio.Queue()
        self.c_callback = TOKEN_CALLBACK(self.python_token_callback)
        self.ctx = lib.gemma3_static_initialize(
            model_path.encode(),
            mmproj_path.encode(),
            json.dumps(overrides).encode()
        )
        self.n_predict = 1024
        self.loop = loop or asyncio.get_running_loop()


    def python_token_callback(self, token_ptr):
        token = ctypes.string_at(token_ptr).decode('utf-8')
        asyncio.run_coroutine_threadsafe(self.q.put(token), self.loop)

    def eval_message(self, message: dict[str, str | list[str]], stream=False):
        msg_text = message["text"]
        msg_imgs = message["images"]
        image_data = []
        image_sizes = []
        if msg_imgs:
            for file_or_data in msg_imgs:
                if isinstance(file_or_data, str):
                    try:
                        img = Image.open(file_or_data)
                        img_bytes = BytesIO()
                        img.save(img_bytes, format=img.format)
                    except FileNotFoundError:
                        print(f"Error: Image file not found: {file_or_data}")
                        exit()
                else:
                    img_bytes = base64.b64decode(file_or_data)
                data = img_bytes.getvalue()
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
        result = lib.gemma3_static_eval_message_with_images(
            msg_text.encode(),
            image_data_pointers,
            image_sizes_array,
            num_images
        )
        if stream:
            result = lib.gemma3_static_stream_response(self.c_callback, self.n_predict)
        else:
            result = lib.gemma3_static_generate_response(self.n_predict)

        if result == 0:
            print("Message with multiple raw images evaluated successfully.")
        else:
            print(f"Error evaluating message with multiple raw images: {result}")
        return result

    async def receive_tokens(self) -> AsyncGenerator[str, None]:
        """Receive tokens"""
        while True:
            token = await self.q.get()
            if token == "[EOS]":  # End-of-stream token
                break
            yield token

    # async def receive_tokens(self, request_id: str) -> AsyncGenerator[str, None]:
    #     """Receive tokens for a specific request."""
    #     if request_id not in self.queues:
    #         raise KeyError(f"No queue for request ID: {request_id}")
    #     try:
    #         while True:
    #             token = await self.queues[request_id].get()
    #             if token == "[EOS]":  # End-of-stream token
    #                 break
    #             yield token
    #     except Exception as e:
    #          yield f"Error in receive_tokens: {e}" # Yield the error
    #     finally:
    #         del self.queues[request_id]  # Clean up the queue when done
    #         print(f"Queue for request ID {request_id} cleaned up")
