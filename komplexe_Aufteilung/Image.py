import random
import tempfile
import torch

from config import *

def loadImage(image_bytes):
    with torch.inference_mode():
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        loadimage_17 = loadimage.load_image(image=tmp_path)
        ##

        imageresizekj_37 = imageresizekj.resize(
            width=1024,
            height=1024,
            upscale_method="nearest-exact",
            keep_proportion=False,
            divisible_by=2,
            crop="center",
            image=get_value_at_index(loadimage_17, 0),
        )

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

        stringconcatenate_54 = stringconcatenate.execute(
            string_a=get_value_at_index(janusimageunderstanding_52, 0),
            string_b=get_value_at_index(text_multiline_56, 0),
            delimiter="",
        )


        cliptextencode_44 = cliptextencode.encode(
            text=get_value_at_index(stringconcatenate_54, 0),
            clip=get_value_at_index(dualcliploader_45, 0),
        )


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


        vaeencode_49 = vaeencode.encode(
            pixels=get_value_at_index(image_rembg_remove_background_62, 0),
            vae=get_value_at_index(vaeloader_32, 0),
        )
        return image_rembg_remove_background_62, cliptextencode_44, vaeencode_49