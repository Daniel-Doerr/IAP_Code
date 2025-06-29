import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


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


def main():
    import_custom_nodes()
    with torch.inference_mode():
        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_17 = loadimage.load_image(image="IMG-20250422-WA0003.jpg")

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_32 = vaeloader.load_vae(
            vae_name="diffusion_pytorch_model.safetensors"
        )

        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        checkpointloadersimple_38 = checkpointloadersimple.load_checkpoint(
            ckpt_name="flux1-kontext-dev.safetensors"
        )

        janusmodelloader = NODE_CLASS_MAPPINGS["JanusModelLoader"]()
        janusmodelloader_51 = janusmodelloader.load_model(
            model_name="deepseek-ai/Janus-Pro-1B"
        )

        imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
        imageresizekj_37 = imageresizekj.resize(
            width=1024,
            height=1024,
            upscale_method="nearest-exact",
            keep_proportion=False,
            divisible_by=2,
            crop="center",
            image=get_value_at_index(loadimage_17, 0),
        )

        janusimageunderstanding = NODE_CLASS_MAPPINGS["JanusImageUnderstanding"]()
        janusimageunderstanding_52 = janusimageunderstanding.analyze_image(
            question="Clearly identify the type of animal (e.g., plush bear, plush rabbit), and specify whether it is shown from the front, side, or back. Keep the description concise and factual.",
            seed=random.randint(1, 2**64),
            temperature=0.30000000000000004,
            top_p=0.9,
            max_new_tokens=128,
            model=get_value_at_index(janusmodelloader_51, 0),
            processor=get_value_at_index(janusmodelloader_51, 1),
            image=get_value_at_index(imageresizekj_37, 0),
        )

        text_multiline = NODE_CLASS_MAPPINGS["Text Multiline"]()
        text_multiline_56 = text_multiline.text_multiline(
            text="\nGenerate the ainimal depicted in a clean, clinical X-ray scan style. The internal bone structure is detailed and anatomically plausible, resembling simplified mammalian bones, including a visible spine with vertebrae, ribcage, arms, legs, joints, pelvis, and digits — all proportioned to the animals plush body. The bones are semi-transparent and softly glowing in white and pale blue, rendered with subtle radiographic shadows. The background is dark and neutral to mimic a real X-ray scan. The style is medical, technical, and illustrative — no horror elements, no visible skull, no face or eyes, no soft tissue, no fur, no fabric seams. The overall mood is scientific and clean, not emotional or creepy. High-resolution, radiographic rendering, suitable for veterinary illustration or educational imaging."
        )

        stringconcatenate = NODE_CLASS_MAPPINGS["StringConcatenate"]()
        stringconcatenate_54 = stringconcatenate.execute(
            string_a=get_value_at_index(janusimageunderstanding_52, 0),
            string_b=get_value_at_index(text_multiline_56, 0),
            delimiter="",
        )

        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        dualcliploader_45 = dualcliploader.load_clip(
            clip_name1="clip_l.safetensors",
            clip_name2="t5/t5xxl_fp16.safetensors",
            type="flux",
            device="default",
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_44 = cliptextencode.encode(
            text=get_value_at_index(stringconcatenate_54, 0),
            clip=get_value_at_index(dualcliploader_45, 0),
        )

        controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
        controlnetloader_47 = controlnetloader.load_controlnet(
            control_net_name="FLUX.1/Shakker-Labs-ControlNet-Union-Pro/diffusion_pytorch_model.safetensors"
        )

        image_rembg_remove_background = NODE_CLASS_MAPPINGS[
            "Image Rembg (Remove Background)"
        ]()
        image_rembg_remove_background_62 = image_rembg_remove_background.image_rembg(
            transparency=False,
            model="u2netp",
            post_processing=False,
            only_mask=False,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
            background_color="white",
            images=get_value_at_index(imageresizekj_37, 0),
        )

        vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
        vaeencode_49 = vaeencode.encode(
            pixels=get_value_at_index(image_rembg_remove_background_62, 0),
            vae=get_value_at_index(vaeloader_32, 0),
        )

        loadimage_58 = loadimage.load_image(image="pasted/image.png")

        cliptextencode_66 = cliptextencode.encode(
            text="low quality, blurry, out of focus, noisy, distorted anatomy, deformed limbs, missing bones, broken joints, horror elements, scary, creepy, disturbing, grotesque, blood, gore, flesh, skin texture, visible eyes, open mouth, facial expression, exposed skull, colorful background, vivid colors, fantasy style, surreal, painterly, cartoon, anime, watercolor, oil painting, overexposed, underexposed, strong shadows, photo artifacts, grain, chromatic aberration, double exposure, body horror, glowing eyes, nightmare style, unsettling, low resolution, soft rendering, plastic texture, shiny surface, incorrect perspective, unrealistic proportions, extra limbs, anatomical errors, fantasy bones, melted shapes, glitch effects, artistic filter, cinematic lighting, emotional tone",
            clip=get_value_at_index(dualcliploader_45, 0),
        )

        depthanythingpreprocessor = NODE_CLASS_MAPPINGS["DepthAnythingPreprocessor"]()
        controlnetapplyadvanced = NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]()
        fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
        textonimage = NODE_CLASS_MAPPINGS["TextOnImage"]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        for q in range(1):
            depthanythingpreprocessor_36 = depthanythingpreprocessor.execute(
                ckpt_name="depth_anything_vitl14.pth",
                resolution=1024,
                image=get_value_at_index(image_rembg_remove_background_62, 0),
            )

            controlnetapplyadvanced_40 = controlnetapplyadvanced.apply_controlnet(
                strength=0.8500000000000002,
                start_percent=0,
                end_percent=1,
                positive=get_value_at_index(cliptextencode_44, 0),
                negative=get_value_at_index(cliptextencode_66, 0),
                control_net=get_value_at_index(controlnetloader_47, 0),
                image=get_value_at_index(depthanythingpreprocessor_36, 0),
                vae=get_value_at_index(vaeloader_32, 0),
            )

            fluxguidance_43 = fluxguidance.append(
                guidance=7,
                conditioning=get_value_at_index(controlnetapplyadvanced_40, 0),
            )

            fluxguidance_42 = fluxguidance.append(
                guidance=1,
                conditioning=get_value_at_index(controlnetapplyadvanced_40, 1),
            )

            ksampler_3 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=1,
                sampler_name="euler",
                scheduler="normal",
                denoise=1,
                model=get_value_at_index(checkpointloadersimple_38, 0),
                positive=get_value_at_index(fluxguidance_43, 0),
                negative=get_value_at_index(fluxguidance_42, 0),
                latent_image=get_value_at_index(vaeencode_49, 0),
            )

            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(ksampler_3, 0),
                vae=get_value_at_index(vaeloader_32, 0),
            )

            imagecompositemasked_57 = imagecompositemasked.composite(
                x=0,
                y=0,
                resize_source=True,
                destination=get_value_at_index(loadimage_58, 0),
                source=get_value_at_index(vaedecode_8, 0),
                mask=get_value_at_index(loadimage_58, 1),
            )

            textonimage_59 = textonimage.apply_text(
                text="TBKH2025\nFLUX Kontext\nTest\n",
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
                image=get_value_at_index(imagecompositemasked_57, 0),
            )

            saveimage_9 = saveimage.save_images(
                filename_prefix="ComfyUI", images=get_value_at_index(textonimage_59, 0)
            )


if __name__ == "__main__":
    main()
