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
        ## loads Stable Diffusion model 
        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name="sd_xl_base_1.0.safetensors"
        )

        ## sets size of output image
        emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
        emptylatentimage_5 = emptylatentimage.generate(
            width=1024, height=1024, batch_size=1
        )

        ## load LoRA with strength of model and clip 
        loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
        loraloader_68 = loraloader.load_lora(
            lora_name="xraylorasdxl.safetensors",
            strength_model=1.0000000000000002,
            strength_clip=1,
            model=get_value_at_index(checkpointloadersimple_4, 0),
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        ## load cliptextencode model
        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()

        ## load positive prompt 
        cliptextencode_6 = cliptextencode.encode(
            text="masterpiece, best quality, ultra-detailed, high resolution, sharp focus, pseudo-x-ray scan of a plush toy, revealing an anatomically plausible, realistic skeleton with detailed bones inside, medical imaging style, monochrome, grayscale, black and white, xray style",
            clip=get_value_at_index(loraloader_68, 1),
        )

        ## load negative prompt 
        cliptextencode_7 = cliptextencode.encode(
            text="worst quality, low quality, blurry, noisy, text, signature, watermark, UI, cartoon, drawing, illustration, sketch, painting, anime, 3D render, (photorealistic plush toy), (visible fabric texture), (visible stuffing), colorful, vibrant colors, toy bones, plastic bones, cartoon bones, unrealistic skeleton, bad anatomy, deformed skeleton, disfigured, mutated limbs, extra limbs, fused bones, skin, fur, organs, background clutter, multiple animals",
            clip=get_value_at_index(loraloader_68, 1),
        )

        ## loads Stable Diffusion model (there are two, could be one)
        checkpointloadersimple_12 = checkpointloadersimple.load_checkpoint(
            ckpt_name="sd_xl_base_1.0.safetensors"
        )

        ## load positive prompt refiner (same as normal prompt)
        cliptextencode_15 = cliptextencode.encode(
            text="masterpiece, best quality, ultra-detailed, high resolution, sharp focus, pseudo-x-ray scan of a plush toy, revealing an anatomically plausible, realistic skeleton with detailed bones inside, medical imaging style, monochrome, grayscale, black and white, xray style",
            clip=get_value_at_index(checkpointloadersimple_12, 1),
        )

        ## load negative prompt refiner (same as normal prompt)
        cliptextencode_16 = cliptextencode.encode(
            text="worst quality, low quality, blurry, noisy, text, signature, watermark, UI, cartoon, drawing, illustration, sketch, painting, anime, 3D render, (photorealistic plush toy), (visible fabric texture), (visible stuffing), colorful, vibrant colors, toy bones, plastic bones, cartoon bones, unrealistic skeleton, bad anatomy, deformed skeleton, disfigured, mutated limbs, extra limbs, fused bones, skin, fur, organs, background clutter, multiple animals",
            clip=get_value_at_index(checkpointloadersimple_12, 1),
        )

        ## define loadimage
        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        
        ## input image
        loadimage_50 = loadimage.load_image(
            image="f693e789-a5a9-4430-b33e-fe2597e9cb9e.jpg"
        )

        ## unconnected vae loader (unnecessary)?
        vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
        vaeencode_51 = vaeencode.encode(
            pixels=get_value_at_index(loadimage_50, 0),
            vae=get_value_at_index(checkpointloadersimple_4, 2),
        )

        ## controlnet loader
        controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
        controlnetloader_52 = controlnetloader.load_controlnet(
            control_net_name="SDXL/controlnet-union-sdxl-1.0/diffusion_pytorch_model_promax.safetensors"
        )

        ## load reference image 
        loadimage_60 = loadimage.load_image(
            image="4139a1d1-9769-45f6-a96e-646496617f24.jpg"
        )

        ## load reference image 
        loadimage_64 = loadimage.load_image(
            image="f6fcd01d-3e55-40ed-adb0-8e17e1a7531a.jpg"
        )


        ## load reference image 
        loadimage_71 = loadimage.load_image(
            image="f0dc608d-3848-4fd4-83c4-a7e1dddc1d9b.jpg"
        )

        ## load reference image 
        loadimage_72 = loadimage.load_image(
            image="cce1254a-26e0-47b9-886e-1f7f6447251e.jpg"
        )

        ## load reference image 
        loadimage_74 = loadimage.load_image(
            image="90a78a86-a2e5-43ec-ad0c-e0d2a72cfb6e.jpg"
        )

        ## load reference image 
        loadimage_77 = loadimage.load_image(
            image="63d047d3-6544-477a-8188-c3a5904db5a4.jpg"
        )

        ## load ipadapter unifiedloader
        ipadapterunifiedloader = NODE_CLASS_MAPPINGS["IPAdapterUnifiedLoader"]()

        ## load iamge batch (for combinding reference images)
        imagebatch = NODE_CLASS_MAPPINGS["ImageBatch"]()

        ## load ipadapter (image processing of reference image)
        ipadapter = NODE_CLASS_MAPPINGS["IPAdapter"]()

        ## style of controlnet (depth map)
        depthanythingpreprocessor = NODE_CLASS_MAPPINGS["DepthAnythingPreprocessor"]()

        ## load controlnetapplyadvanced
        controlnetapplyadvanced = NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]()

        ## load ksampleradvanced 
        ksampleradvanced = NODE_CLASS_MAPPINGS["KSamplerAdvanced"]()

        ## load vaedecode
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

        ## load saveimage (unnecessary if I save them separately)
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        ## load second style of controlnet (unnecessary)
        cannyedgepreprocessor = NODE_CLASS_MAPPINGS["CannyEdgePreprocessor"]()

        for q in range(1):
            ## ipadapter unified loader
            ipadapterunifiedloader_63 = ipadapterunifiedloader.load_models(
                preset="VIT-G (medium strength)",
                model=get_value_at_index(loraloader_68, 0),
            )

            ## image batch (combine the input images)
            imagebatch_65 = imagebatch.batch(
                image1=get_value_at_index(loadimage_64, 0),
                image2=get_value_at_index(loadimage_60, 0),
            )
            ## ""
            imagebatch_70 = imagebatch.batch(
                image1=get_value_at_index(imagebatch_65, 0),
                image2=get_value_at_index(loadimage_71, 0),
            )
            ## ""
            imagebatch_73 = imagebatch.batch(
                image1=get_value_at_index(imagebatch_70, 0),
                image2=get_value_at_index(loadimage_72, 0),
            )
            ## ""
            imagebatch_75 = imagebatch.batch(
                image1=get_value_at_index(imagebatch_73, 0),
                image2=get_value_at_index(loadimage_74, 0),
            )
            ## ""
            imagebatch_76 = imagebatch.batch(
                image1=get_value_at_index(imagebatch_75, 0),
                image2=get_value_at_index(loadimage_77, 0),
            )

            ## load the combined images in the ipadapter
            ipadapter_58 = ipadapter.apply_ipadapter(
                weight=1,
                start_at=0,
                end_at=1,
                weight_type="standard",
                model=get_value_at_index(ipadapterunifiedloader_63, 0),
                ipadapter=get_value_at_index(ipadapterunifiedloader_63, 1),
                image=get_value_at_index(imagebatch_76, 0),
            )

            ## creating depth image for controlnet
            depthanythingpreprocessor_55 = depthanythingpreprocessor.execute(
                ckpt_name="depth_anything_vitb14.pth",
                resolution=512,
                image=get_value_at_index(loadimage_50, 0),
            )


            ## applies the controlnet 
            controlnetapplyadvanced_54 = controlnetapplyadvanced.apply_controlnet(
                strength=0.7000000000000002,
                start_percent=0,
                end_percent=1,
                positive=get_value_at_index(cliptextencode_6, 0),
                negative=get_value_at_index(cliptextencode_7, 0),
                control_net=get_value_at_index(controlnetloader_52, 0),
                image=get_value_at_index(depthanythingpreprocessor_55, 0),
            )

            ## first ksampler advanced with noise
            ksampleradvanced_10 = ksampleradvanced.sample(
                add_noise="enable",
                noise_seed=random.randint(1, 2**64),
                steps=50,
                cfg=9,
                sampler_name="euler",
                scheduler="sgm_uniform",
                start_at_step=0,
                end_at_step=30,
                return_with_leftover_noise="enable",
                model=get_value_at_index(ipadapter_58, 0),
                positive=get_value_at_index(controlnetapplyadvanced_54, 0),
                negative=get_value_at_index(controlnetapplyadvanced_54, 1),
                latent_image=get_value_at_index(emptylatentimage_5, 0),
            )

            ## second ksampler advanced without noise
            ksampleradvanced_11 = ksampleradvanced.sample(
                add_noise="disable",
                noise_seed=random.randint(1, 2**64),
                steps=50,
                cfg=8,
                sampler_name="euler",
                scheduler="normal",
                start_at_step=30,
                end_at_step=10000,
                return_with_leftover_noise="disable",
                model=get_value_at_index(checkpointloadersimple_12, 0),
                positive=get_value_at_index(cliptextencode_15, 0),
                negative=get_value_at_index(cliptextencode_16, 0),
                latent_image=get_value_at_index(ksampleradvanced_10, 0),
            )

            ## vae decode image (output image)
            vaedecode_17 = vaedecode.decode(
                samples=get_value_at_index(ksampleradvanced_11, 0),
                vae=get_value_at_index(checkpointloadersimple_12, 2),
            )

            ## save output image
            saveimage_19 = saveimage.save_images(
                filename_prefix="ComfyUI", images=get_value_at_index(vaedecode_17, 0)
            )

            ## different style of controlnet (unnecessary)
            cannyedgepreprocessor_56 = cannyedgepreprocessor.execute(
                low_threshold=100,
                high_threshold=200,
                resolution=512,
                image=get_value_at_index(loadimage_50, 0),
            )


if __name__ == "__main__":
    main()
