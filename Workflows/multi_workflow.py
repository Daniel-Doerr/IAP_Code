import io
import random
from typing import Any
import numpy as np
from PIL import Image
from typing import Union, Sequence, Mapping
import torch
import tempfile

from multi_config import load_config_based_on_workflow


class load_workflow:
    
    def converte_image(self, generatedImage: Any) -> io.BytesIO:
        global config_obj
        image = config_obj.get_value_at_index(generatedImage, 0)
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
        return img_buffer


    def format_text_for_field(self, first_name, last_name, animal_name, line_length=13, lines=3):
        import textwrap
        text = f"{first_name} {last_name} {animal_name}"
        wrapped = textwrap.wrap(text, width=line_length)
        # Ensure exactly 'lines' lines (pad with empty strings if needed)
        wrapped = wrapped[:lines] + [""] * (lines - len(wrapped))
        return "\n".join(wrapped)
    
    
    def get_path_from_bytes(self, image_bytes: bytes) -> str:     
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(image_bytes)
            return tmp.name



    # Global variable to hold the config object
    conig_obj = None

    # Initialize the workflow with the command
    def __init__(self, workfolw_name : str):
        # If you add a new workflow, you need to add the name and the name of the function here
        self.befehle = {
            "AIprompt_FLUX_Kontext": self.AIprompt_FLUX_Kontext_workflow,
            "IP_Adapter_SDXL": self.IP_Adapter_SDXL_workflow,
            # "The name on which you want to call the workflow": self.name_of_the_function,
        }
        global config_obj
        # create a config object 
        config_obj = load_config_based_on_workflow()
        # Load the configuration for the specified workflow
        self.config_vars = config_obj.load_config(workfolw_name)
        
        # Ensure config_vars is not None to prevent errors
        if self.config_vars is None:
            print(f"Warning: No configuration found for workflow '{workfolw_name}'. Using empty configuration.")
            self.config_vars = {}


    # by calling generate with the workflow_name, the spessific function will be called
    def generate(self, workflow_name: str, image_bytes: bytes, animal_type: str, first_name: str, last_name: str, animal_name: str):
        funktion = self.befehle.get(workflow_name)
        if funktion:
            return funktion(workflow_name, image_bytes, animal_type, first_name, last_name, animal_name)
        else:
            print(f"Unbekannter Befehl: {workflow_name}")
            return None


    # AIprompt_FLUX_Kontext_workflow function
    def AIprompt_FLUX_Kontext_workflow(self, workflow_name, image_bytes: bytes, animal_type: str, first_name: str, last_name: str, animal_name: str) -> io.BytesIO:
        # Ensure the global config_obj is accessible
        global config_obj
        # Check if config_vars is a dictionary
        if not isinstance(self.config_vars, dict):
            raise ValueError(f"Configuration not properly loaded for workflow '{workflow_name}'. Expected dict, got {type(self.config_vars)}")
        # Update the global namespace with the config_vars
        globals().update(self.config_vars)
        
        # Access the config_obj methods and own functions
        get_value_at_index = config_obj.get_value_at_index
        converte_image = self.converte_image
        format_text_for_field = self.format_text_for_field
        get_path_from_bytes = self.get_path_from_bytes
        
        with torch.inference_mode(): 
            # Load the image from bytes
            tmp_path = get_path_from_bytes(image_bytes)
            
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
                text=format_text_for_field(first_name, last_name, animal_name), ####### Custom Text #######
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






    # IP_Adapter_SDXL_workflow function
    def IP_Adapter_SDXL_workflow(self, workflow_name, image_bytes: bytes, animal_type: str, first_name: str, last_name: str, animal_name: str) -> io.BytesIO:
        # Ensure the global config_obj is accessible
        global config_obj
        # Check if config_vars is a dictionary
        if not isinstance(self.config_vars, dict):
            raise ValueError(f"Configuration not properly loaded for workflow '{workflow_name}'. Expected dict, got {type(self.config_vars)}")
        # Update the global namespace with the config_vars
        globals().update(self.config_vars)
        
        # Access the config_obj methods and own functions
        get_value_at_index = config_obj.get_value_at_index
        converte_image = self.converte_image
        format_text_for_field = self.format_text_for_field
        get_path_from_bytes = self.get_path_from_bytes
        

        with torch.inference_mode(): 
            # Load the image from bytes
            tmp_path = get_path_from_bytes(image_bytes)
            # Put the rest of the workflow here starting from the line below "with torch.inference_mode():" 

            loadimage_50 = loadimage.load_image(
                image="dd35e9ae-9c87-4db3-85b8-709cbb2d89cf.webp"
            )

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

            cliptextencode_6 = cliptextencode.encode(
                text=get_value_at_index(janusimageunderstanding_129, 0),
                clip=get_value_at_index(loraloader_68, 1),
            )

            cliptextencode_15 = cliptextencode.encode(
                text=get_value_at_index(janusimageunderstanding_129, 0),
                clip=get_value_at_index(checkpointloadersimple_12, 1),
            )


            loadimage_60 = loadimage.load_image(image="Cat_back.png")

            loadimage_64 = loadimage.load_image(image="Cat_front2.png")

            loadimage_71 = loadimage.load_image(image="Cat_front2.png")

            loadimage_72 = loadimage.load_image(image="Cat_side.png")

            loadimage_111 = loadimage.load_image(image="pasted/image.png")

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

            # remove the save image and return the last variable (which is a image)
            # don't return somthing like "saveimage_XX" or "easy_showanything_XX" (XX = some number)
            result = converte_image(textonimage_115) # change last_image to the last variable which is a image
            return result




"""
    def Name_of_workflow(self, workflow_name, image_bytes: bytes, animal_type: str, first_name: str, last_name: str, animal_name: str) -> io.BytesIO:
        # Ensure the global config_obj is accessible
        global config_obj
        # Check if config_vars is a dictionary
        if not isinstance(self.config_vars, dict):
            raise ValueError(f"Configuration not properly loaded for workflow '{workflow_name}'. Expected dict, got {type(self.config_vars)}")
        # Update the global namespace with the config_vars
        globals().update(self.config_vars)
        
        # Access the config_obj methods and own functions
        get_value_at_index = config_obj.get_value_at_index
        converte_image = self.converte_image
        format_text_for_field = self.format_text_for_field
        get_path_from_bytes = self.get_path_from_bytes
        

        with torch.inference_mode(): 
            # Load the image from bytes
            tmp_path = get_path_from_bytes(image_bytes)
            # Put the rest of the workflow here starting from the line below "with torch.inference_mode():" 


        
            # remove the save image and return the last variable (which is a image)
            # don't return somthing like "saveimage_XX" or "easy_showanything_XX" (XX = some number)
            result = converte_image(last_image) # change last_image to the last variable which is a image
            return result
"""