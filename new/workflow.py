# workflow
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

from config import *
from loadImages import load_image

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



def generateImage(image_bytes: bytes, animal_type: str, first_name: str, last_name: str, animal_name: str) -> io.BytesIO:
    with torch.inference_mode():

        img = load_image(animal_type, image_bytes)

        loadimage_111 = img.loadimage_111
        loadimage_148 = img.loadimage_148
        cliptextencode_6 = img.cliptextencode_6
        cliptextencode_15 = img.cliptextencode_15
        ipadapterencoder_136 = img.ipadapterencoder_136
        ipadapterencoder_139 = img.ipadapterencoder_139
        ipadapterencoder_140 = img.ipadapterencoder_140
        ipadapterencoder_141 = img.ipadapterencoder_141

        del img

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
            image=get_value_at_index(loadimage_148, 0),
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

        def format_text_for_field(first_name, last_name, animal_name, line_length=13, lines=3):
            import textwrap
            text = f"{first_name} {last_name} {animal_name}"
            wrapped = textwrap.wrap(text, width=line_length)
            # Ensure exactly 'lines' lines (pad with empty strings if needed)
            wrapped = wrapped[:lines] + [""] * (lines - len(wrapped))
            return "\n".join(wrapped)

        textonimage_115 = textonimage.apply_text(
            text=format_text_for_field(first_name, last_name, animal_name),
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
        return image
