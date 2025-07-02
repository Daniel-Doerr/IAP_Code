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
        loadimage_50 = loadimage.load_image(
            image="dd35e9ae-9c87-4db3-85b8-709cbb2d89cf.webp"
        )

        janusimageunderstanding = NODE_CLASS_MAPPINGS["JanusImageUnderstanding"]()
        janusimageunderstanding_129 = janusimageunderstanding.analyze_image(
            question="Generate a descriptive text prompt intended for use in an image generation model (e.g., Stable Diffusion) to create an X-ray-style image of the given subject. This prompt should focus entirely on the skeletal structure, while intentionally avoiding any mention of the skull, face, or head to maintain a neutral and non-creepy aesthetic.\n\nStructure the prompt in the following way:\n\nSpecies and anatomical context: Begin by identifying the subject and state that it is being represented in X-ray form, focusing on internal bone structures.\n\nDetailed skeletal description (excluding head):\nDescribe key bone structures such as:\n\nSpine and vertebrae\n\nLimbs (e.g., elongated hind legs, forelimbs)\n\nDigits or toes\n\nPelvis, ribs (if applicable)\n\nJoints and connections between bones\nBe anatomically accurate and emphasize proportions and layout.\n\nVisual appearance and rendering style:\nDefine the visual style using phrases like:\n\n“semi-transparent bones glowing in white or blue”\n\n“clean medical X-ray look”\n\n“set against a dark or neutral background”\n\n“no visible soft tissue details unless subtle”\n\nStylistic tone and exclusions:\nMake it clear that the output should:\n\nBe clinical, technical, or illustrative\n\nAvoid all horror, fantasy, or emotionally charged interpretations\n\nExplicitly exclude any depiction or focus on the head or skull\n\nOptional enhancement terms:\nEncourage inclusion of terms such as:\n\n“high resolution”\n\n“medical illustration”\n\n“radiographic scan”\n\n“scientific rendering”\n\nThe result should be a clean, anatomical-style image prompt focused on skeletal anatomy below the neck, suitable for generating an X-ray-style output that is medically inspired and visually neutral.",
            seed=random.randint(1, 2**64),
            temperature=0.7000000000000001,
            top_p=0.9,
            max_new_tokens=2048,
            model=get_value_at_index(janusmodelloader_130, 0),
            processor=get_value_at_index(janusmodelloader_130, 1),
            image=get_value_at_index(loadimage_50, 0),
        )

        loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
        loraloader_68 = loraloader.load_lora(
            lora_name="xraylorasdxl.safetensors",
            strength_model=1.0000000000000002,
            strength_clip=1,
            model=get_value_at_index(checkpointloadersimple_4, 0),
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_6 = cliptextencode.encode(
            text=get_value_at_index(janusimageunderstanding_129, 0),
            clip=get_value_at_index(loraloader_68, 1),
        )

        cliptextencode_7 = cliptextencode.encode(
            text="worst quality, low quality, blurry, noisy, text, signature, watermark, UI, cartoon, drawing, illustration, sketch, painting, anime, 3D render, (photorealistic plush toy), (visible fabric texture), (visible stuffing), colorful, vibrant colors, toy bones, plastic bones, cartoon bones, unrealistic skeleton, bad anatomy, deformed skeleton, disfigured, mutated limbs, extra limbs, fused bones, skin, fur, organs, background clutter, multiple animals",
            clip=get_value_at_index(loraloader_68, 1),
        )

        checkpointloadersimple_12 = checkpointloadersimple.load_checkpoint(
            ckpt_name="SDXL/sd_xl_refiner_1.0.safetensors"
        )

        cliptextencode_15 = cliptextencode.encode(
            text=get_value_at_index(janusimageunderstanding_129, 0),
            clip=get_value_at_index(checkpointloadersimple_12, 1),
        )

        cliptextencode_16 = cliptextencode.encode(
            text="worst quality, low quality, blurry, noisy, text, signature, watermark, UI, cartoon, drawing, illustration, sketch, painting, anime, 3D render, (photorealistic plush toy), (visible fabric texture), (visible stuffing), colorful, vibrant colors, toy bones, plastic bones, cartoon bones, unrealistic skeleton, bad anatomy, deformed skeleton, disfigured, mutated limbs, extra limbs, fused bones, skin, fur, organs, background clutter, multiple animals",
            clip=get_value_at_index(checkpointloadersimple_12, 1),
        )

        controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
        controlnetloader_52 = controlnetloader.load_controlnet(
            control_net_name="SDXL/controlnet-union-sdxl-1.0/diffusion_pytorch_model_promax.safetensors"
        )

        loadimage_60 = loadimage.load_image(image="Cat_back.png")

        loadimage_64 = loadimage.load_image(image="Cat_front2.png")

        loadimage_71 = loadimage.load_image(image="Cat_front2.png")

        loadimage_72 = loadimage.load_image(image="Cat_side.png")

        loadimage_111 = loadimage.load_image(image="pasted/image.png")

        ipadapterunifiedloader = NODE_CLASS_MAPPINGS["IPAdapterUnifiedLoader"]()
        ipadapterunifiedloader_63 = ipadapterunifiedloader.load_models(
            preset="PLUS (high strength)", model=get_value_at_index(loraloader_68, 0)
        )

        ipadapterencoder = NODE_CLASS_MAPPINGS["IPAdapterEncoder"]()
        ipadapterencoder_136 = ipadapterencoder.encode(
            weight=1.0000000000000002,
            ipadapter=get_value_at_index(ipadapterunifiedloader_63, 1),
            image=get_value_at_index(loadimage_60, 0),
        )

        ipadapterencoder_139 = ipadapterencoder.encode(
            weight=1,
            ipadapter=get_value_at_index(ipadapterunifiedloader_63, 1),
            image=get_value_at_index(loadimage_64, 0),
        )

        ipadapterencoder_140 = ipadapterencoder.encode(
            weight=1.0000000000000002,
            ipadapter=get_value_at_index(ipadapterunifiedloader_63, 1),
            image=get_value_at_index(loadimage_72, 0),
        )

        ipadapterencoder_141 = ipadapterencoder.encode(
            weight=1,
            ipadapter=get_value_at_index(ipadapterunifiedloader_63, 1),
            image=get_value_at_index(loadimage_71, 0),
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
        easy_showanything = NODE_CLASS_MAPPINGS["easy showAnything"]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        for q in range(1):
            ipadaptercombineembeds_138 = ipadaptercombineembeds.batch(
                method="concat",
                embed1=get_value_at_index(ipadapterencoder_139, 0),
                embed2=get_value_at_index(ipadapterencoder_136, 0),
                embed3=get_value_at_index(ipadapterencoder_140, 0),
                embed4=get_value_at_index(ipadapterencoder_141, 0),
            )

            ipadaptercombineembeds_143 = ipadaptercombineembeds.batch(
                method="concat",
                embed1=get_value_at_index(ipadapterencoder_139, 1),
                embed2=get_value_at_index(ipadapterencoder_136, 1),
                embed3=get_value_at_index(ipadapterencoder_140, 1),
                embed4=get_value_at_index(ipadapterencoder_141, 1),
            )

            ipadapterembeds_137 = ipadapterembeds.apply_ipadapter(
                weight=1.0000000000000002,
                weight_type="linear",
                start_at=0,
                end_at=1,
                embeds_scaling="V only",
                model=get_value_at_index(ipadapterunifiedloader_63, 0),
                ipadapter=get_value_at_index(ipadapterunifiedloader_63, 1),
                pos_embed=get_value_at_index(ipadaptercombineembeds_138, 0),
                neg_embed=get_value_at_index(ipadaptercombineembeds_143, 0),
            )

            imageresizekj_82 = imageresizekj.resize(
                width=1024,
                height=1152,
                upscale_method="nearest-exact",
                keep_proportion=False,
                divisible_by=2,
                crop="center",
                image=get_value_at_index(loadimage_50, 0),
            )

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
                strength=1.0000000000000002,
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
                steps=40,
                cfg=15.5,
                sampler_name="euler",
                scheduler="sgm_uniform",
                start_at_step=0,
                end_at_step=35,
                return_with_leftover_noise="enable",
                model=get_value_at_index(ipadapterembeds_137, 0),
                positive=get_value_at_index(controlnetapplyadvanced_54, 0),
                negative=get_value_at_index(controlnetapplyadvanced_54, 1),
                latent_image=get_value_at_index(emptylatentimage_5, 0),
            )

            ksampleradvanced_11 = ksampleradvanced.sample(
                add_noise="disable",
                noise_seed=random.randint(1, 2**64),
                steps=40,
                cfg=14,
                sampler_name="euler",
                scheduler="sgm_uniform",
                start_at_step=35,
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
                text="TBKH2025\n",
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

            easy_showanything_134 = easy_showanything.log_input(
                text="**Species and Anatomical Context:**\n\n**Detailed Skeletal Description (Excluding Head):**\n\n- **Spine and Vertebrae:**\n  - **Cervical vertebrae** (C1-C7)\n  - **Thoracic vertebrae** (T1-T12)\n  - **Lumbar vertebrae** (L1-L5)\n\n- **Limbs:**\n  - **Hind legs**\n  - **Forelimbs** (knee, wrist, elbow)\n\n- **Digits or Toes:**\n  - **Foot metatarsals**\n  - **Phalanges**\n\n- **Pelvis, Ribs (If Applicable):**\n  - **Coccyx**\n  - **Pubic symphysis**\n  - **Pubic tubercle**\n\n- **Joints and Connections:**\n  - **Hip joint**\n  - **Knee joint**\n  - **Elbow joint**\n\n- **Anatomical Structure:**\n  - **Spine**\n  - **Lumbar spine**\n  - **Pelvic and hip bones**\n  - **Knee joint**\n  - **Wrist joint**\n  - **Elbow joint**\n\n- **Visual Appearance and Rendering Style:**\n  - **Semi-transparent bones glowing in white or blue**\n  - **Clean medical X-ray look**\n  - **set against a dark or neutral background**\n  - **no visible soft tissue details unless subtle**\n\n**Optional Enhancement Terms:**\n\n- **High Resolution:**\n- **Medical Illustration:**\n- **Radiographic Scan:**\n- **Scientific Rendering:**\n- **Low-Fidelity:**\n- **High-Fidelity:**\n\n**Stylistic Tone and Exclusions:**\n\n- **Clinically, Technical, or Illustrative**\n- **Free from horror, fantasy, or emotionally charged interpretations**\n- **No depiction of the head or skull**",
                anything=get_value_at_index(janusimageunderstanding_129, 0),
                unique_id=2550299730956210994,
            )

            saveimage_135 = saveimage.save_images(
                filename_prefix="ComfyUI", images=get_value_at_index(textonimage_115, 0)
            )


if __name__ == "__main__":
    main()
