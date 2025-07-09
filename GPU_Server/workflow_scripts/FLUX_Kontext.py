import os
import sys
import io
from typing import Any, Union, Sequence, Mapping
import torch
from PIL import Image  
import random

# Add the parent directory to sys.path to handle relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from functions import Functions

# Change the class name to your workflow name, e.g. "FLUX_Kontext"
class FLUX_Kontext:
    # You don't need to change this function
    def __init__(self, arg_function):
        self.functions = arg_function

    def start_load_once(self):
        self.config = self.load_once()


    def load_once(self):
        """Safe time by loading surtain nodes only once."""
        # follow the steps in the generate function

        NODE_CLASS_MAPPINGS = self.NODE_CLASS_MAPPINGS
        get_value_at_index = self.functions.get_value_at_index

        with torch.inference_mode():
            loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
            vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
            checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
            janusmodelloader = NODE_CLASS_MAPPINGS["JanusModelLoader"]()
            imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
            janusimageunderstanding = NODE_CLASS_MAPPINGS["JanusImageUnderstanding"]()
            text_multiline = NODE_CLASS_MAPPINGS["Text Multiline"]()
            stringconcatenate = NODE_CLASS_MAPPINGS["StringConcatenate"]()
            dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
            cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
            image_rembg_remove_background = NODE_CLASS_MAPPINGS["Image Rembg (Remove Background)"]()
            vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
            depthanythingpreprocessor = NODE_CLASS_MAPPINGS["DepthAnythingPreprocessor"]()
            controlnetapplyadvanced = NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]()
            fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
            ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
            vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
            imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
            textonimage = NODE_CLASS_MAPPINGS["TextOnImage"]()
            saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()
            
            vaeloader_32 = vaeloader.load_vae(
                vae_name="diffusion_pytorch_model.safetensors"
            )

            checkpointloadersimple_38 = checkpointloadersimple.load_checkpoint(
                ckpt_name="flux1-kontext-dev.safetensors"
            )

            janusmodelloader_51 = janusmodelloader.load_model(
                model_name="deepseek-ai/Janus-Pro-1B"
            )

            dualcliploader_45 = dualcliploader.load_clip(
                clip_name1="clip_l.safetensors",
                clip_name2="t5/t5xxl_fp16.safetensors",
                type="flux",
                device="default",
            )

            controlnetloader_47 = controlnetloader.load_controlnet(
                control_net_name="FLUX.1/Shakker-Labs-ControlNet-Union-Pro/diffusion_pytorch_model.safetensors"
            )

        return {k: v for k, v in locals().items() if k != "self"}






    def generate(self, workflow_name: str, image_bytes: bytes, animal_type: str, first_name: str, last_name: str, animal_name: str) -> io.BytesIO:
        # Update local variables from self attributes (you don't need to write self. in front of the variables)
        globals().update(self.config)
        # Make the functions available in the local scope (no self. prefix needed)
        NODE_CLASS_MAPPINGS = self.NODE_CLASS_MAPPINGS
        get_value_at_index = self.functions.get_value_at_index
        get_path_from_bytes = self.functions.get_path_from_bytes
        converte_image = self.functions.converte_image
        format_text_for_field = self.functions.format_text_for_field
        # Load the image from bytes
        tmp_path = get_path_from_bytes(image_bytes)

        with torch.inference_mode():
            # How to put your own workflow here 
            # Step 1: You need to export yor own workflow as a Python skript from ComfyUI 
            # Step 2: Copy all code from the scripts main function into this function
            # Step 3: Using "Ctrl + f" search for "NODE_CLASS_MAPPINGS" and move all lines of code into the "load_once" function
            # Step 4: Now search for ".safetensors" and move all lines of code into the "load_once" function
            # Step 5: Now search for "model_name" or "modelloader" and move all lines of code into the "load_once" function
            # Optional: You can also move static Prompts or images into the "load_once" function, just nothing that depends directly or indirectly on the input image
            # Step 6: Locate your input image and change the code to "loadimage_X = loadimage.load_image(image=tmp_path)" (X = a number, e.g. 17)
            # If you use an IP-Adapter or watermark check that the path is correct
            # If you use text on the image, you can use "format_text_for_field(your_text_input)" to format the text correctly for the field
            # Step 7: Remove "saveimge_X" because we don't need to save the image in the workflow, we return it as a buffer
            # Step 8: Change in "converte_image(generatedImage)" generatedImage to the last generated image, e.g. "converte_image(textonimage_59)"
            # Step 9 : Change the name of the class to the name of your workflow, e.g. "FLUX_Kontext" and change the name of the file
            # Step 10: Go into the dispatcher.py and follow the steps to add your workflow to the dispatcher

            text_multiline_56 = text_multiline.text_multiline(
                text="\nGenerate the ainimal depicted in a clean, clinical X-ray scan style. The internal bone structure is detailed and anatomically plausible, resembling simplified mammalian bones, including a visible spine with vertebrae, ribcage, arms, legs, joints, pelvis, and digits — all proportioned to the animals plush body. The bones are semi-transparent and softly glowing in white and pale blue, rendered with subtle radiographic shadows. The background is dark and neutral to mimic a real X-ray scan. The style is medical, technical, and illustrative — no horror elements, no visible skull, no face or eyes, no soft tissue, no fur, no fabric seams. The overall mood is scientific and clean, not emotional or creepy. High-resolution, radiographic rendering, suitable for veterinary illustration or educational imaging."
            )

            cliptextencode_66 = cliptextencode.encode(
                text="low quality, blurry, out of focus, noisy, distorted anatomy, deformed limbs, missing bones, broken joints, horror elements, scary, creepy, disturbing, grotesque, blood, gore, flesh, skin texture, visible eyes, open mouth, facial expression, exposed skull, colorful background, vivid colors, fantasy style, surreal, painterly, cartoon, anime, watercolor, oil painting, overexposed, underexposed, strong shadows, photo artifacts, grain, chromatic aberration, double exposure, body horror, glowing eyes, nightmare style, unsettling, low resolution, soft rendering, plastic texture, shiny surface, incorrect perspective, unrealistic proportions, extra limbs, anatomical errors, fantasy bones, melted shapes, glitch effects, artistic filter, cinematic lighting, emotional tone",
                clip=get_value_at_index(dualcliploader_45, 0),
            )

            # watermark image change to right path 
            loadimage_58 = loadimage.load_image(image="pasted/image.png")
            
            # change the static image to the input image
            loadimage_17 = loadimage.load_image(image=tmp_path)

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
                text=format_text_for_field(first_name + " " + last_name + " " + animal_name), ####### Custom Text #######
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

            result = converte_image(textonimage_59)
            
            return result