from typing import Optional, AsyncGenerator
import ctypes
from ctypes import (cdll, c_void_p, c_char_p, c_int, create_string_buffer, CFUNCTYPE,
                    POINTER, c_ubyte, cast, Structure, Array, c_voidp, c_bool)
import time
from io import BytesIO
import json
import asyncio
import base64

from PIL import Image


TOKEN_CALLBACK = CFUNCTYPE(None, c_char_p)


def init_lib(dll_path: str):
    """Initialize Gemma3 C API lib and return

    Args:
        dll_path: Library file path


    """
    lib = cdll.LoadLibrary(dll_path)

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

    # static eval message text only. Uses different C function
    lib.gemma3_static_eval_message_text_only.argtypes = [
        c_char_p,                   # msg str
        c_bool                      # add_bos
    ]
    lib.gemma3_static_eval_message_text_only.restype = c_int

    # static eval message with images
    lib.gemma3_static_eval_message_with_images.argtypes = [
        c_char_p,                   # msg_str
        POINTER(POINTER(c_ubyte)),  # Array of pointers to image data
        POINTER(c_int),             # Array of image data sizes
        c_int,                      # Number of images
        c_bool                      # add bos
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

    lib.gemma3_static_reset.argtypes = []
    lib.gemma3_static_reset.restype = c_int

    lib.gemma3_is_generating.argtypes = []
    lib.gemma3_is_generating.restype = c_bool

    lib.gemma3_static_interrupt.argtypes = []
    lib.gemma3_static_interrupt.restype = c_voidp

    return lib
