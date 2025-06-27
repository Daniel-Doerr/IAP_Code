# config
import requests
import base64
import time
import BytesIO
import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from PIL import Image
import numpy as np
import tempfile
import io


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()

def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()

from nodes import NODE_CLASS_MAPPINGS

import_custom_nodes()
with torch.inference_mode():
    checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
    checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
        ckpt_name="sd_xl_base_1.0.safetensors"
    )

    emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
    emptylatentimage_5 = emptylatentimage.generate(
        width=1024, height=1024, batch_size=1
    )

    janusmodelloader = NODE_CLASS_MAPPINGS["JanusModelLoader"]()
    janusmodelloader_130 = janusmodelloader.load_model(
        model_name="deepseek-ai/Janus-Pro-1B"
    )

    loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()

    janusimageunderstanding = NODE_CLASS_MAPPINGS["JanusImageUnderstanding"]()


    loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
    loraloader_68 = loraloader.load_lora(
        lora_name="xraylorasdxl.safetensors",
        strength_model=1.0000000000000002,
        strength_clip=1,
        model=get_value_at_index(checkpointloadersimple_4, 0),
        clip=get_value_at_index(checkpointloadersimple_4, 1),
    )

    cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()


    cliptextencode_7 = cliptextencode.encode(
        text="low quality, blurry, noisy, text, watermark, cartoon, drawing, sketch, painting, anime, 3D render, plush toy, fabric, stuffing, colorful, toy bones, plastic bones, cartoon bones, unrealistic skeleton, bad anatomy, deformed, mutated, extra limbs, fused bones, skin, fur, organs, clutter, multiple animals",
        clip=get_value_at_index(loraloader_68, 1),
    )

    checkpointloadersimple_12 = checkpointloadersimple.load_checkpoint(
        ckpt_name="SDXL/sd_xl_refiner_1.0.safetensors"
    )


    cliptextencode_16 = cliptextencode.encode(
        text="low quality, blurry, noisy, text, watermark, cartoon, drawing, sketch, painting, anime, 3D render, plush toy, fabric, stuffing, colorful, toy bones, plastic bones, cartoon bones, unrealistic skeleton, bad anatomy, deformed, mutated, extra limbs, fused bones, skin, fur, organs, clutter, multiple animals",
        clip=get_value_at_index(checkpointloadersimple_12, 1),
    )

    controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
    controlnetloader_52 = controlnetloader.load_controlnet(
        control_net_name="SDXL/controlnet-union-sdxl-1.0/diffusion_pytorch_model_promax.safetensors"
    )

    ipadapterencoder = NODE_CLASS_MAPPINGS["IPAdapterEncoder"]()
    ipadapterunifiedloader = NODE_CLASS_MAPPINGS["IPAdapterUnifiedLoader"]()
    ipadapterunifiedloader_63 = ipadapterunifiedloader.load_models(
        preset="PLUS (high strength)", model=get_value_at_index(loraloader_68, 0)
    )

    ipadaptercombineembeds = NODE_CLASS_MAPPINGS["IPAdapterCombineEmbeds"]()
    ipadapterembeds = NODE_CLASS_MAPPINGS["IPAdapterEmbeds"]()
    imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
    image_rembg_remove_background = NODE_CLASS_MAPPINGS[
        "Image Rembg (Remove Background)"
    ]()
    depthanythingpreprocessor = NODE_CLASS_MAPPINGS["DepthAnythingPreprocessor"]()
    controlnetapplyadvanced = NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]()
    ksampleradvanced = NODE_CLASS_MAPPINGS["KSamplerAdvanced"]()
    vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
    imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
    textonimage = NODE_CLASS_MAPPINGS["TextOnImage"]()