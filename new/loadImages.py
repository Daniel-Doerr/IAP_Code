#load_images
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
import tempfile

from config import *

class load_image():
    def __init__(self, animal_type: str, image_bytes: bytes):
        self.image_bytes = image_bytes
        self.animal_type = animal_type

        def load_animal_image(base_path, animal, view):
            # Normalize folder names
            folder_name = f"{animal.capitalize()}_done"
            fallback_folder = animal.capitalize()  # If _done doesn't exist

            folder_path = os.path.join(base_path, folder_name)
            if not os.path.exists(folder_path):
                folder_path = os.path.join(base_path, fallback_folder)

            # Create filename
            filename = f"{animal.lower()}_{view}.png"
            image_path = os.path.join(folder_path, filename)

            if os.path.exists(image_path):
                return image_path  
            else:
                raise FileNotFoundError(f"Image not found: {image_path}") 


        with torch.inference_mode():
            base_path = "/mnt/data/tbkh2025_dk/ComfyUI/Input_animals"

            self.loadimage_60 = loadimage.load_image(image=load_animal_image(base_path, animal_type, "front"))
            self.loadimage_64 = loadimage.load_image(image=load_animal_image(base_path, animal_type, "back"))
            self.loadimage_71 = loadimage.load_image(image=load_animal_image(base_path, animal_type, "side"))
            self.loadimage_72 = loadimage.load_image(image=load_animal_image(base_path, animal_type, "front2"))

            self.loadimage_111 = loadimage.load_image(image="/mnt/data/tbkh2025_dk/ComfyUI/input/pasted/image.png")


            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name
            self.loadimage_148 = loadimage.load_image(image=tmp_path)

            self.janusimageunderstanding_129 = janusimageunderstanding.analyze_image(
                question="Generate a prompt for an image model to create a clinical X-ray-style image of the animal's skeleton (no head/skull), showing only bones (spine, limbs, digits, pelvis, ribs, joints) in a high-resolution, neutral, medical illustration style. No soft tissue, horror, or fantasy. Use terms like radiographic scan, scientific rendering, semi-transparent glowing bones, dark background.",
                seed=random.randint(1, 2**64),
                temperature=0.7000000000000001,
                top_p=0.9,
                max_new_tokens=2048,
                model=get_value_at_index(janusmodelloader_130, 0),
                processor=get_value_at_index(janusmodelloader_130, 1),
                image=get_value_at_index(self.loadimage_148, 0),
            )

            self.cliptextencode_6 = cliptextencode.encode(
            text=get_value_at_index(self.janusimageunderstanding_129, 0),
            clip=get_value_at_index(loraloader_68, 1),
            )

            self.cliptextencode_15 = cliptextencode.encode(
                text=get_value_at_index(self.janusimageunderstanding_129, 0),
                clip=get_value_at_index(checkpointloadersimple_12, 1),
            )

            self.ipadapterencoder_136 = ipadapterencoder.encode(
                weight=1.0000000000000002,
                ipadapter=get_value_at_index(ipadapterunifiedloader_63, 1),
                image=get_value_at_index(self.loadimage_60, 0),
            )

            self.ipadapterencoder_139 = ipadapterencoder.encode(
                weight=1,
                ipadapter=get_value_at_index(ipadapterunifiedloader_63, 1),
                image=get_value_at_index(self.loadimage_64, 0),
            )

            self.ipadapterencoder_140 = ipadapterencoder.encode(
                weight=1.0000000000000002,
                ipadapter=get_value_at_index(ipadapterunifiedloader_63, 1),
                image=get_value_at_index(self.loadimage_72, 0),
            )

            self.ipadapterencoder_141 = ipadapterencoder.encode(
                weight=1,
                ipadapter=get_value_at_index(ipadapterunifiedloader_63, 1),
                image=get_value_at_index(self.loadimage_71, 0),
            )

    
    def __getitem__(self, key):
        return getattr(self, key)

    def __getattr__(self, key):
        raise AttributeError(f"'LoadImage' object has no attribute '{key}'")