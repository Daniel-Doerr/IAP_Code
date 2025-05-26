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

WEB_SERVER = "http://localhost:8000"


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
import io

import_custom_nodes()
with torch.inference_mode():
    checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
    emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
    loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
    cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
    loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
    controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
    ipadapterunifiedloader = NODE_CLASS_MAPPINGS["IPAdapterUnifiedLoader"]()
    imagebatch = NODE_CLASS_MAPPINGS["ImageBatch"]()
    ipadapter = NODE_CLASS_MAPPINGS["IPAdapter"]()
    imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
    image_rembg_remove_background = NODE_CLASS_MAPPINGS["Image Rembg (Remove Background)"]()
    depthanythingpreprocessor = NODE_CLASS_MAPPINGS["DepthAnythingPreprocessor"]()
    controlnetapplyadvanced = NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]()
    ksampleradvanced = NODE_CLASS_MAPPINGS["KSamplerAdvanced"]()
    vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
    imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
    textonimage = NODE_CLASS_MAPPINGS["TextOnImage"]()
    saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

    checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
        ckpt_name="sd_xl_base_1.0.safetensors"
    )

    emptylatentimage_5 = emptylatentimage.generate(
        width=1024, height=1024, batch_size=1
    )
    
    loraloader_68 = loraloader.load_lora(
        lora_name="xraylorasdxl.safetensors",
        strength_model=1.0000000000000002,
        strength_clip=1,
        model=get_value_at_index(checkpointloadersimple_4, 0),
        clip=get_value_at_index(checkpointloadersimple_4, 1),
    )

    cliptextencode_7 = cliptextencode.encode(
        text="worst quality, low quality, blurry, noisy, text, signature, watermark, UI, cartoon, drawing, illustration, sketch, painting, anime, 3D render, (photorealistic plush toy), (visible fabric texture), (visible stuffing), colorful, vibrant colors, toy bones, plastic bones, cartoon bones, unrealistic skeleton, bad anatomy, deformed skeleton, disfigured, mutated limbs, extra limbs, fused bones, skin, fur, organs, background clutter, multiple animals",
        clip=get_value_at_index(loraloader_68, 1),
    )

    checkpointloadersimple_12 = checkpointloadersimple.load_checkpoint(
        ckpt_name="SDXL/sd_xl_refiner_1.0.safetensors"
    )

    cliptextencode_16 = cliptextencode.encode(
        text="worst quality, low quality, blurry, noisy, text, signature, watermark, UI, cartoon, drawing, illustration, sketch, painting, anime, 3D render, (photorealistic plush toy), (visible fabric texture), (visible stuffing), colorful, vibrant colors, toy bones, plastic bones, cartoon bones, unrealistic skeleton, bad anatomy, deformed skeleton, disfigured, mutated limbs, extra limbs, fused bones, skin, fur, organs, background clutter, multiple animals",
        clip=get_value_at_index(checkpointloadersimple_12, 1),
    )

    controlnetloader_52 = controlnetloader.load_controlnet(
        control_net_name="SDXL/controlnet-union-sdxl-1.0/diffusion_pytorch_model_promax.safetensors"
    )
    ## Watermark images
    loadimage_111 = loadimage.load_image(image="pasted/image.png")

    ipadapterunifiedloader_63 = ipadapterunifiedloader.load_models(
        preset="STANDARD (medium strength)",
        model=get_value_at_index(loraloader_68, 0),
    )


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
    

def load_IPAdapter(animal):
    with torch.inference_mode():
        base_path = "/mnt/data/tbkh2025_dk/ComfyUI/Input_animals"

        # IPAdapter images 
        loadimage_60 = loadimage.load_image(image=load_animal_image(base_path, animal, "front"))
        loadimage_64 = loadimage.load_image(image=load_animal_image(base_path, animal, "back"))
        loadimage_71 = loadimage.load_image(image=load_animal_image(base_path, animal, "side"))
        loadimage_72 = loadimage.load_image(image=load_animal_image(base_path, animal, "front2"))

        imagebatch_119 = imagebatch.batch(
            image1=get_value_at_index(loadimage_64, 0),
            image2=get_value_at_index(loadimage_60, 0),
        )

        imagebatch_120 = imagebatch.batch(
            image1=get_value_at_index(loadimage_71, 0),
            image2=get_value_at_index(imagebatch_119, 0),
        )

        imagebatch_121 = imagebatch.batch(
            image1=get_value_at_index(loadimage_72, 0),
            image2=get_value_at_index(imagebatch_120, 0),
        )

        ipadapter_58 = ipadapter.apply_ipadapter(
            weight=1,
            start_at=0,
            end_at=1,
            weight_type="standard",
            model=get_value_at_index(ipadapterunifiedloader_63, 0),
            ipadapter=get_value_at_index(ipadapterunifiedloader_63, 1),
            image=get_value_at_index(imagebatch_121, 0),
        )

        return ipadapter_58


def load_input_image(image):
    with torch.inference_mode():
        if isinstance(image, bytes):
            # Save bytes to a temporary PNG file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(image)
                tmp_path = tmp.name
            loadimage_50 = loadimage.load_image(
                image=tmp_path
            )
        else:
            loadimage_50 = loadimage.load_image(
                image=image
            )
        
        imageresizekj_82 = imageresizekj.resize(
            width=1024,
            height=1024,
            upscale_method="nearest-exact",
            keep_proportion=False,
            divisible_by=2,
            crop="center",
            image=get_value_at_index(loadimage_50, 0),
        )
        return imageresizekj_82


def load_Prompt(animal):
    with torch.inference_mode():
        cliptextencode_6 = cliptextencode.encode(
            text=f"masterpiece, best quality, ultra-detailed, high resolution, sharp focus, pseudo-x-ray scan of a {animal} plush toy, revealing an anatomically plausible, realistic skeleton with detailed bones inside, medical imaging style, monochrome, grayscale, black and white, xray style",
            clip=get_value_at_index(loraloader_68, 1),
        )

        cliptextencode_15 = cliptextencode.encode(
            text=f"masterpiece, best quality, ultra-detailed, high resolution, sharp focus, pseudo-x-ray scan of a {animal} plush toy, revealing an anatomically plausible, realistic skeleton with detailed bones inside, medical imaging style, monochrome, grayscale, black and white, xray style",
            clip=get_value_at_index(checkpointloadersimple_12, 1),
        )
        return cliptextencode_6, cliptextencode_15


last_type = ""

def poll_job():
    while True:
        try:
            response = requests.get(f"{WEB_SERVER}/job")
            if response.status_code == 204:
                print("No job received...")
                time.sleep(2)
                continue

            # extract image data
            image_bytes = response.content

            # Header-Metadata
            job_id = response.headers.get("img_id")
            first_name = response.headers.get("first_name")
            last_name = response.headers.get("last_name")
            animal_name = response.headers.get("animal_name")
            animal_type = response.headers.get("animal_type")
            ## not jet implemented
            bone_broken = response.headers.get("bone_broken")

            print(f"Job received: {job_id}")
            print(f"Patient: {first_name} {last_name}, Animal: {animal_name}, AnimalType: {animal_type}")




            # Process image here 
            with torch.inference_mode():
                if animal_type != last_type:
                    cliptextencode_6, cliptextencode_15 = load_Prompt(animal_type)
                    ipadapter_58 = load_IPAdapter(animal_type)
                imageresizekj_82 = load_input_image(image_bytes)


                for q in range(4):
                    image_rembg_remove_background_86 = (
                        image_rembg_remove_background.image_rembg(
                            transparency=False,
                            model="u2netp",
                            post_processing=False,
                            only_mask=False,
                            alpha_matting=True,
                            alpha_matting_foreground_threshold=240,
                            alpha_matting_background_threshold=10,
                            alpha_matting_erode_size=10,
                            background_color="white",
                            images=get_value_at_index(imageresizekj_82, 0),
                        )
                    )

                    depthanythingpreprocessor_55 = depthanythingpreprocessor.execute(
                        ckpt_name="depth_anything_vitb14.pth",
                        resolution=1024,
                        image=get_value_at_index(image_rembg_remove_background_86, 0),
                    )

                    image_rembg_remove_background_96 = (
                        image_rembg_remove_background.image_rembg(
                            transparency=False,
                            model="u2net",
                            post_processing=False,
                            only_mask=False,
                            alpha_matting=True,
                            alpha_matting_foreground_threshold=240,
                            alpha_matting_background_threshold=10,
                            alpha_matting_erode_size=10,
                            background_color="black",
                            images=get_value_at_index(depthanythingpreprocessor_55, 0),
                        )
                    )

                    controlnetapplyadvanced_54 = controlnetapplyadvanced.apply_controlnet(
                        strength=0.9000000000000001,
                        start_percent=0,
                        end_percent=1,
                        positive=get_value_at_index(cliptextencode_6, 0),
                        negative=get_value_at_index(cliptextencode_7, 0),
                        control_net=get_value_at_index(controlnetloader_52, 0),
                        image=get_value_at_index(image_rembg_remove_background_96, 0),
                    )

                    ksampleradvanced_10 = ksampleradvanced.sample(
                        add_noise="enable",
                        noise_seed=random.randint(1, 2**64),
                        steps=30,
                        cfg=9.200000000000001,
                        sampler_name="euler",
                        scheduler="sgm_uniform",
                        start_at_step=0,
                        end_at_step=20,
                        return_with_leftover_noise="enable",
                        model=get_value_at_index(ipadapter_58, 0),
                        positive=get_value_at_index(controlnetapplyadvanced_54, 0),
                        negative=get_value_at_index(controlnetapplyadvanced_54, 1),
                        latent_image=get_value_at_index(emptylatentimage_5, 0),
                    )

                    ksampleradvanced_11 = ksampleradvanced.sample(
                        add_noise="disable",
                        noise_seed=random.randint(1, 2**64),
                        steps=30,
                        cfg=7.2,
                        sampler_name="euler",
                        scheduler="sgm_uniform",
                        start_at_step=20,
                        end_at_step=908,
                        return_with_leftover_noise="disable",
                        model=get_value_at_index(checkpointloadersimple_12, 0),
                        positive=get_value_at_index(cliptextencode_15, 0),
                        negative=get_value_at_index(cliptextencode_16, 0),
                        latent_image=get_value_at_index(ksampleradvanced_10, 0),
                    )

                    vaedecode_17 = vaedecode.decode(
                        samples=get_value_at_index(ksampleradvanced_11, 0),
                        vae=get_value_at_index(checkpointloadersimple_12, 2),
                    )

                    imagecompositemasked_112 = imagecompositemasked.composite(
                        x=0,
                        y=0,
                        resize_source=True,
                        destination=get_value_at_index(loadimage_111, 0),
                        source=get_value_at_index(vaedecode_17, 0),
                        mask=get_value_at_index(loadimage_111, 1),
                    )

                    textonimage_115 = textonimage.apply_text(
                        text=f"{first_name}\n{last_name}\n{animal_name}",
                        x=853,
                        y=898,
                        font_size=16,
                        text_color="#d3c7b6",
                        text_opacity=1,
                        use_gradient=False,
                        start_color="#ff0000",
                        end_color="#0000ff",
                        angle=0,
                        stroke_width=0,
                        stroke_color="#000000",
                        stroke_opacity=1,
                        shadow_x=0,
                        shadow_y=0,
                        shadow_color="#000000",
                        shadow_opacity=1,
                        font_file="en-AllRoundItalic.ttf",
                        image=get_value_at_index(imagecompositemasked_112, 0),
                    )
                    image = get_value_at_index(textonimage_115, 0)
                    arr = image.cpu().numpy()

                    # Remove extra dimensions if present
                    arr = np.squeeze(arr)  # removes dimensions of size 1

                    # If shape is (H, W, 4) or (H, W, 3), continue. If (4, H, W), transpose.
                    if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
                        arr = np.transpose(arr, (1, 2, 0))  # (C, H, W) -> (H, W, C)

                    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)

                    img = Image.fromarray(arr)

                    print("Image processed.")

                    # send result
                    files = {
                        "result": ("result.png", BytesIO(image_bytes), "image/png")
                    }
                    headers = {
                        "image_id": job_id
                    }

                    res = requests.post(f"{WEB_SERVER}/job", files=files, headers=headers)
                    print("Result sent:", res.status_code, res.text)
            last_type = animal_type

        except Exception as e:
            print("Error:", e)
            time.sleep(3)



if __name__ == "__main__":
    poll_job()



"""
def poll_job():
    while True:
        try:
            response = requests.get(f"{WEB_SERVER}/job")
            if response.status_code == 204:
                print("No job recieved...")
                time.sleep(2)
                continue

            job = response.json()
            job_id = job["job_id"]
            print(f"Job recieved: {job_id}")
            print(f"Job details: {job["text"]}")
            print(f"bone broken: {job["bone_broken"]}")

            # Processing (for demo: sleeping)
            time.sleep(5)
            print("picture processed")

            # Return result image (for demo: same picture)
            result = {
                "result_image_base64": job["image_base64"]
            }
            res = requests.post(f"{WEB_SERVER}/job/{job_id}/result", json=result)
            print("Result sent:", res.status_code)

        except Exception as e:
            print("Error:", e)
            time.sleep(3)
"""