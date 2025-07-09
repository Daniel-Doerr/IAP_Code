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
class IP_Adapter_SDXL:
    # You don't need to change this function
    def __init__(self,arg_function: Functions, gpu_id: int = None):
        self.gpu_id = gpu_id
        self.device = self._setup_device()
        self.functions = arg_function
        

    def _setup_device(self):
        """Setup the device for this workflow instance."""
        if self.gpu_id is not None and self.gpu_id >= 0:
            device = f"cuda:{self.gpu_id}"
            try:
                torch.cuda.set_device(self.gpu_id)
                # Set device before any model loading
                torch.cuda.empty_cache()  # Clear cache
                print(f"IP_Adapter_SDXL workflow bound to GPU {self.gpu_id}")
                return device
            except Exception as e:
                print(f"Failed to set GPU {self.gpu_id}, falling back to CPU: {e}")
                return "cpu"
        else:
            print("IP_Adapter_SDXL workflow running on CPU")
            return "cpu"

    def start_load_once(self):
        """Load workflow with proper device handling"""
        try:
            # Ensure we're on the correct device
            if self.device.startswith("cuda") and torch.cuda.is_available():
                device_id = int(self.device.split(":")[-1]) if ":" in self.device else 0
                torch.cuda.set_device(device_id)
                print(f"IP_Adapter_SDXL: Loading models on {self.device}")
            self.config = self.load_once()
        except Exception as e:
            print(f"Error in start_load_once: {e}")
            print("Falling back to CPU...")
            self.device = "cpu"
            self.config = self.load_once()


    def load_once(self):
        """Safe time by loading surtain nodes only once."""
        # follow the steps in the generate function

        NODE_CLASS_MAPPINGS = self.NODE_CLASS_MAPPINGS
        get_value_at_index = self.functions.get_value_at_index

        with torch.inference_mode():
            print(f"Loading IP_Adapter_SDXL nodes on {self.device}...")
            # follow the steps 3, 4, 5 and Optional from the generate function
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

        return {k: v for k, v in locals().items() if k != "self"}


    def generate(self, workflow_name: str, image_bytes: bytes, animal_type: str, first_name: str, last_name: str, animal_name: str) -> io.BytesIO:
        try:
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
                # Ensure we're on the correct device with proper error handling
                try:
                    if self.device.startswith("cuda") and torch.cuda.is_available():
                        device_id = int(self.device.split(":")[-1]) if ":" in self.device else 0
                        if device_id < torch.cuda.device_count():
                            torch.cuda.set_device(device_id)
                            # Also force all operations to use this device
                            device = torch.device(self.device)
                            print(f"IP_Adapter_SDXL: Set device to {self.device}")
                        else:
                            print(f"Warning: Device {self.device} not available, falling back to cuda:0")
                            torch.cuda.set_device(0)
                            device = torch.device("cuda:0")
                            self.device = "cuda:0"
                    else:
                        print(f"IP_Adapter_SDXL: Using CPU mode")
                        device = torch.device("cpu")
                        self.device = "cpu"
                except Exception as e:
                    print(f"Warning: Could not set device {self.device}, falling back to CPU: {e}")
                    device = torch.device("cpu")
                    self.device = "cpu"
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

                loadimage_50 = loadimage.load_image(image=tmp_path)

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
                        image=get_value_at_index(imagecompositemasked_112, 0),
                    )
                

                result = converte_image(textonimage_115)
                
                return result

        except Exception as e:
            print(f"Error in IP_Adapter_SDXL generate method: {e}")
            import traceback
            traceback.print_exc()
            raise
