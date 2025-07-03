import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import io
import numpy as np

class load_config_based_on_workflow:  # only for NODE_CLASS_MAPPINGS and models
    
    # functions needed for loading the ComfyUI environment and custom nodes
    def get_value_at_index(self, obj: Union[Sequence, Mapping], index: int) -> Any:
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

    def find_path(self, name: str, path: str = None) -> str:
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
        return self.find_path(name, parent_directory)

    def add_comfyui_directory_to_sys_path(self) -> None:
        """
        Add 'ComfyUI' to the sys.path
        """
        comfyui_path = self.find_path("ComfyUI")
        if comfyui_path is not None and os.path.isdir(comfyui_path):
            if comfyui_path not in sys.path:
                sys.path.append(comfyui_path)
                print(f"'{comfyui_path}' added to sys.path")

    def add_extra_model_paths(self) -> None:
        """
        Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
        """
        try:
            from main import load_extra_path_config
        except ImportError:
            print(
                "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
            )
            try:
                from utils.extra_config import load_extra_path_config
            except ImportError:
                print("Could not import load_extra_path_config from utils.extra_config.")
                return

        extra_model_paths = self.find_path("extra_model_paths.yaml")

        if extra_model_paths is not None:
            load_extra_path_config(extra_model_paths)
        else:
            print("Could not find the extra_model_paths config file.")

    def import_custom_nodes(self) -> None:
        """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

        This function sets up a new asyncio event loop, initializes the PromptServer,
        creates a PromptQueue, and initializes the custom nodes.
        """
        try:
            import asyncio
            import execution
            from nodes import init_extra_nodes
            import server
        except ImportError as e:
            print(f"Could not import required modules for custom nodes: {e}")
            return

        # Creating a new event loop and setting it as the default loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Creating an instance of PromptServer with the loop
        server_instance = server.PromptServer(loop)
        execution.PromptQueue(server_instance)

        # Initializing custom nodes
        init_extra_nodes()
    # end of functions needed for loading the ComfyUI environment and custom nodes 


    # This is the constructor of the class, which will be called when an instance of the class is created (in multi_workflow.py)
    def __init__(self):
        # This will set up the environment for ComfyUI
        self.add_comfyui_directory_to_sys_path()
        self.add_extra_model_paths()
        try:
            from nodes import NODE_CLASS_MAPPINGS
            self.NODE_CLASS_MAPPINGS = NODE_CLASS_MAPPINGS
        except ImportError:
            print("Could not import NODE_CLASS_MAPPINGS from nodes.")
            self.NODE_CLASS_MAPPINGS = {}
        self.import_custom_nodes()

        # If you add a new workflow, you need to add the name and the name of the function here
        self.befehle = {
            "AIprompt_FLUX_Kontext": self.AIprompt_FLUX_Kontext,
            "IP_Adapter_SDXL": self.IP_Adapter_SDXL,
            # "The name on which you want to call the workflow": self.name_of_the_function,
        }


    # by calling load_config(workflow_name), you can load the config for the desired workflow
    def load_config(self, workflow_name: str):
        funktion = self.befehle.get(workflow_name)
        if funktion:
            return funktion()
        else:
            print(f"Unbekannter Befehl: {workflow_name}")
            return None

    def AIprompt_FLUX_Kontext(self):
        NODE_CLASS_MAPPINGS = self.NODE_CLASS_MAPPINGS
        get_value_at_index = self.get_value_at_index
        with torch.inference_mode():
            print("AIprompt_FLUX_Kontext workflow started")

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

            text_multiline_56 = text_multiline.text_multiline(
                text="\nGenerate the ainimal depicted in a clean, clinical X-ray scan style. The internal bone structure is detailed and anatomically plausible, resembling simplified mammalian bones, including a visible spine with vertebrae, ribcage, arms, legs, joints, pelvis, and digits — all proportioned to the animals plush body. The bones are semi-transparent and softly glowing in white and pale blue, rendered with subtle radiographic shadows. The background is dark and neutral to mimic a real X-ray scan. The style is medical, technical, and illustrative — no horror elements, no visible skull, no face or eyes, no soft tissue, no fur, no fabric seams. The overall mood is scientific and clean, not emotional or creepy. High-resolution, radiographic rendering, suitable for veterinary illustration or educational imaging."
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


            cliptextencode_66 = cliptextencode.encode(
                text="low quality, blurry, out of focus, noisy, distorted anatomy, deformed limbs, missing bones, broken joints, horror elements, scary, creepy, disturbing, grotesque, blood, gore, flesh, skin texture, visible eyes, open mouth, facial expression, exposed skull, colorful background, vivid colors, fantasy style, surreal, painterly, cartoon, anime, watercolor, oil painting, overexposed, underexposed, strong shadows, photo artifacts, grain, chromatic aberration, double exposure, body horror, glowing eyes, nightmare style, unsettling, low resolution, soft rendering, plastic texture, shiny surface, incorrect perspective, unrealistic proportions, extra limbs, anatomical errors, fantasy bones, melted shapes, glitch effects, artistic filter, cinematic lighting, emotional tone",
                clip=get_value_at_index(dualcliploader_45, 0),
            )

            loadimage_58 = loadimage.load_image(image="pasted/image.png")

            # Return all local variables except 'self'
            return {k: v for k, v in locals().items() if k != "self"}





    def IP_Adapter_SDXL(self):
        NODE_CLASS_MAPPINGS = self.NODE_CLASS_MAPPINGS
        get_value_at_index = self.get_value_at_index
        with torch.inference_mode():
            print("IP_Adapter_SDXL workflow started")
            # put your workflow models here
            # search for "NODE_CLASS_MAPPINGS" in the comfyui script and put all of them here 
            # then put all variables befor the for loop here, which have nothing to do with the input image (somthing like a watermark or a text can be put here)
            # TIPS: Coppy all variables befor the for loop 
            # then cut the ones which use "loadimage_XX" (XX = some number) 
            # and then cut all variables which are unknown marked 
            # all cut variables will be put back befor the for loop 

            checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
            emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
            janusmodelloader = NODE_CLASS_MAPPINGS["JanusModelLoader"]()
            loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
            janusimageunderstanding = NODE_CLASS_MAPPINGS["JanusImageUnderstanding"]()
            loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
            cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
            ipadapterunifiedloader = NODE_CLASS_MAPPINGS["IPAdapterUnifiedLoader"]()
            ipadapterencoder = NODE_CLASS_MAPPINGS["IPAdapterEncoder"]()
            ipadaptercombineembeds = NODE_CLASS_MAPPINGS["IPAdapterCombineEmbeds"]()
            ipadapterembeds = NODE_CLASS_MAPPINGS["IPAdapterEmbeds"]()
            imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
            image_rembg_remove_background = NODE_CLASS_MAPPINGS["Image Rembg (Remove Background)"]()
            depthanythingpreprocessor = NODE_CLASS_MAPPINGS["DepthAnythingPreprocessor"]()
            controlnetapplyadvanced = NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]()
            ksampleradvanced = NODE_CLASS_MAPPINGS["KSamplerAdvanced"]()
            vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
            imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
            textonimage = NODE_CLASS_MAPPINGS["TextOnImage"]()
            easy_showanything = NODE_CLASS_MAPPINGS["easy showAnything"]()
            saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

            checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
                ckpt_name="sd_xl_base_1.0.safetensors"
            )

            emptylatentimage_5 = emptylatentimage.generate(
                width=1024, height=1024, batch_size=1
            )

            janusmodelloader_130 = janusmodelloader.load_model(
                model_name="deepseek-ai/Janus-Pro-1B"
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

            ipadapterunifiedloader_63 = ipadapterunifiedloader.load_models(
                preset="PLUS (high strength)", model=get_value_at_index(loraloader_68, 0)
            )

        # will return all local variables except 'self'
        return {k: v for k, v in locals().items() if k != "self"}
    


"""
    def Workflow_name(self):
        NODE_CLASS_MAPPINGS = self.NODE_CLASS_MAPPINGS
        get_value_at_index = self.get_value_at_index
        with torch.inference_mode():
            print("Workflow_name started")
            # put your workflow models here
            # search for "NODE_CLASS_MAPPINGS" in the comfyui script and put all of them here 
            # then put all variables befor the for loop here, which have nothing to do with the input image (somthing like a watermark or a text can be put here)
            # TIPS: Coppy all variables befor the for loop 
            # then cut the ones which use "loadimage_XX" (XX = some number) 
            # and then cut all variables which are unknown marked 
            # all cut variables will be put back befor the for loop 


        # will return all local variables except 'self'
        return {k: v for k, v in locals().items() if k != "self"}
"""