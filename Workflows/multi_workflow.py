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
    conig_obj = None
    def __init__(self, befehl : str):
        self.befehle = {
            "AIprompt_FLUX_Kontext": self.AIprompt_FLUX_Kontext_workflow,
            "IP_Adapter_SDXL": self.IP_Adapter_SDXL_workflow,
        }
        global config_obj
        config_obj = load_config_based_on_workflow()
        self.config_vars = config_obj.load_config(befehl)
        
        # Ensure config_vars is not None to prevent errors
        if self.config_vars is None:
            print(f"Warning: No configuration found for workflow '{befehl}'. Using empty configuration.")
            self.config_vars = {}


    def generate(self, befehl: str, image_bytes: bytes, animal_type: str, first_name: str, last_name: str, animal_name: str):
        funktion = self.befehle.get(befehl)
        if funktion:
            return funktion(befehl, image_bytes, animal_type, first_name, last_name, animal_name)
        else:
            print(f"Unbekannter Befehl: {befehl}")
            return None
    
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




    def AIprompt_FLUX_Kontext_workflow(self, befehl, image_bytes: bytes, animal_type: str, first_name: str, last_name: str, animal_name: str) -> io.BytesIO:
        global config_obj
        
        # Ensure config_vars is properly loaded
        if not isinstance(self.config_vars, dict):
            raise ValueError(f"Configuration not properly loaded for workflow '{befehl}'. Expected dict, got {type(self.config_vars)}")
        
        # Automatisch alle Variablen aus der Konfiguration laden
        # Wir verwenden globals() um die Variablen in den globalen Namespace zu setzen
        globals().update(self.config_vars)
        
        get_value_at_index = config_obj.get_value_at_index
        converte_image = self.converte_image
        format_text_for_field = self.format_text_for_field
        


        with torch.inference_mode(): 
            tmp_path = self.get_path_from_bytes(image_bytes)
            
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

    def IP_Adapter_SDXL_workflow(self, befehl, image_bytes: bytes, animal_type: str, first_name: str, last_name: str, animal_name: str) -> io.BytesIO:
        return "IP_Adapter_SDXL is not implemented yet."