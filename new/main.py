# main
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


from workflow import generateImage



WEB_SERVER = "http://localhost:8001" 


def get_access_token(password: str) -> str:
    """Authenticate with the backend and get a JWT access token."""
    url = f"{WEB_SERVER}/token"
    data = {"password": password}
    response = requests.post(url, data=data)
    response.raise_for_status()
    return response.json()["access_token"]


def poll_job():
    index = 0
    while True:
        try:
            path = "/mnt/data/tbkh2025_dk/ComfyUI/Input_animals/Dog_done/dog_original.png"
            # extract image data
            with open(path, "rb") as f:
                image_bytes = f.read()

            # Header-Metadata
            job_id = index
            first_name = "hallo"
            last_name = "neuer Workflow"
            animal_name = "test"
            animal_type = "dog"

            print(f"Job received: {job_id}")
            print(f"Patient: {first_name} {last_name}, Animal: {animal_name}, AnimalType: {animal_type}")
            
            img_buffer = generateImage(image_bytes, animal_type, first_name, last_name, animal_name)

            index += 1

            # Save the output image as a PNG in a directory
            output_dir = "output_images"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{animal_name}_{job_id}.png")

            # Assuming img_buffer is a BytesIO or PIL Image object
            if isinstance(img_buffer, bytes):
                img = Image.open(io.BytesIO(img_buffer))
                img.save(output_path, format="PNG")
            elif isinstance(img_buffer, io.BytesIO):
                img_buffer.seek(0)
                img = Image.open(img_buffer)
                img.save(output_path, format="PNG")
            elif isinstance(img_buffer, Image.Image):
                img_buffer.save(output_path, format="PNG")
            elif isinstance(img_buffer, torch.Tensor):
                arr = img_buffer.cpu().numpy()
                arr = np.squeeze(arr)
                if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
                    arr = np.transpose(arr, (1, 2, 0))
                arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
                img = Image.fromarray(arr)
                img.save(output_path, format="PNG")
            else:
                print(f"Unknown image buffer type: {type(img_buffer)}")


        except Exception as e:
            print("Error:", e)
            time.sleep(3)



if __name__ == "__main__":
    poll_job()



"""
arr = image.cpu().numpy()

# Remove extra dimensions if present
arr = np.squeeze(arr)  # removes dimensions of size 1

# If shape is (H, W, 4) or (H, W, 3), continue. If (4, H, W), transpose.
if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
    arr = np.transpose(arr, (1, 2, 0))  # (C, H, W) -> (H, W, C)

arr = np.clip(arr * 255, 0, 255).astype(np.uint8)

img = Image.fromarray(arr)

img_buffer = io.BytesIO()
img.save(img_buffer, format="PNG")
img_buffer.seek(0)
"""