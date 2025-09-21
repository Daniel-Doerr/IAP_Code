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
class ChromaV44:
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
            print("Loading nodes...")
            # follow the steps 3, 4, 5 and Optional from the generate function
            emptysd3latentimage = NODE_CLASS_MAPPINGS["EmptySD3LatentImage"]()
            emptysd3latentimage_69 = emptysd3latentimage.generate(
                width=1024, height=1024, batch_size=1
            )

            cliploader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
            cliploader_78 = cliploader.load_clip(
                clip_name="t5/t5xxl_fp16.safetensors", type="chroma", device="default"
            )

            t5tokenizeroptions = NODE_CLASS_MAPPINGS["T5TokenizerOptions"]()
            t5tokenizeroptions_82 = t5tokenizeroptions.set_options(
                min_padding=1, min_length=0, clip=get_value_at_index(cliploader_78, 0)
            )

            cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            cliptextencode_75 = cliptextencode.encode(
                text="llustration, anime, drawing, artwork, bad hands, blurry, low quality, out of focus, deformed, smudged, red",
                clip=get_value_at_index(t5tokenizeroptions_82, 0),
            )

            unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
            unetloader_76 = unetloader.load_unet(
                unet_name="chroma-unlocked-v44-detail-calibrated.safetensors",
                weight_dtype="default",
            )

            vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
            vaeloader_80 = vaeloader.load_vae(
                vae_name="diffusion_pytorch_model.safetensors"
            )

            loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
            imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
            image_rembg_remove_background = NODE_CLASS_MAPPINGS["Image Rembg (Remove Background)"]()
            depthanythingpreprocessor = NODE_CLASS_MAPPINGS["DepthAnythingPreprocessor"]()
            alphachanelasmask = NODE_CLASS_MAPPINGS["AlphaChanelAsMask"]()
            masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
            multiplynode = NODE_CLASS_MAPPINGS["MultiplyNode"]()
            invertimagenode = NODE_CLASS_MAPPINGS["InvertImageNode"]()
            addnode = NODE_CLASS_MAPPINGS["AddNode"]()
            vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
            loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
            ollamaconnectivityv2 = NODE_CLASS_MAPPINGS["OllamaConnectivityV2"]()
            ollamageneratev2 = NODE_CLASS_MAPPINGS["OllamaGenerateV2"]()
            ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
            vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
            imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
            textonimage = NODE_CLASS_MAPPINGS["TextOnImage"]()
            saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

            loraloadermodelonly_159 = loraloadermodelonly.load_lora_model_only(
                lora_name="Hyper-Chroma-Turbo-Alpha-16steps-lora.safetensors",
                strength_model=0.4900000000000001,
                model=get_value_at_index(unetloader_76, 0),
            )

        return {k: v for k, v in locals().items() if k != "self"}






    def generate(self, workflow_name: str, image_bytes: bytes, animal_type: str, first_name: str, last_name: str, animal_name: str) -> io.BytesIO:
        # Update local variables from self attributes (you don't need to write self. in front of the variables)
        globals().update(self.config)
        # Make the functions available in the local scope (no self. prefix needed)
        NODE_CLASS_MAPPINGS = self.NODE_CLASS_MAPPINGS
        get_value_at_index = self.functions.get_value_at_index
        get_path_from_bytes = self.functions.get_path_from_bytes
        convert_image = self.functions.converte_image
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
            # Step 8: Change in "convert_image(generatedImage)" generatedImage to the last generated image, e.g. "convert_image(textonimage_59)"
            # Step 9 : Change the name of the class to the name of your workflow, e.g. "FLUX_Kontext" and change the name of the file
            # Step 10: Go into the dispatcher.py and follow the steps to add your workflow to the dispatcher

            
            loadimage_89 = loadimage.load_image(image=tmp_path)

            
            imageresizekj_164 = imageresizekj.resize(
                width=1024,
                height=1024,
                upscale_method="nearest-exact",
                keep_proportion=False,
                divisible_by=2,
                crop="center",
                image=get_value_at_index(loadimage_89, 0),
            )
            
            image_rembg_remove_background_91 = image_rembg_remove_background.image_rembg(
                transparency=False,
                model="u2net",
                post_processing=False,
                only_mask=False,
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=10,
                background_color="black",
                images=get_value_at_index(imageresizekj_164, 0),
            )
            
            depthanythingpreprocessor_86 = depthanythingpreprocessor.execute(
                ckpt_name="depth_anything_vitl14.pth",
                resolution=1024,
                image=get_value_at_index(image_rembg_remove_background_91, 0),
            )

            image_rembg_remove_background_109 = image_rembg_remove_background.image_rembg(
                transparency=True,
                model="isnet-general-use",
                post_processing=False,
                only_mask=False,
                alpha_matting=False,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=10,
                background_color="none",
                images=get_value_at_index(image_rembg_remove_background_91, 0),
            )

            
            alphachanelasmask_110 = alphachanelasmask.node(
                method="invert",
                images=get_value_at_index(image_rembg_remove_background_109, 0),
            )
            
            masktoimage_113 = masktoimage.mask_to_image(
                mask=get_value_at_index(alphachanelasmask_110, 0)
            )
            
            multiplynode_120 = multiplynode.multiply(
                input1=get_value_at_index(depthanythingpreprocessor_86, 0),
                input2=get_value_at_index(masktoimage_113, 0),
            )

            loadimage_116 = loadimage.load_image(image="pasted/image (1).png")

            imageresizekj_165 = imageresizekj.resize(
                width=1024,
                height=1024,
                upscale_method="nearest-exact",
                keep_proportion=False,
                divisible_by=2,
                crop="center",
                image=get_value_at_index(loadimage_116, 0),
            )

            invertimagenode_119 = invertimagenode.invert_image(
                image=get_value_at_index(masktoimage_113, 0)
            )

            multiplynode_122 = multiplynode.multiply(
                input1=get_value_at_index(imageresizekj_165, 0),
                input2=get_value_at_index(invertimagenode_119, 0),
            )

            addnode_126 = addnode.add(
                input1=get_value_at_index(multiplynode_120, 0),
                input2=get_value_at_index(multiplynode_122, 0),
            )

            vaeencode_97 = vaeencode.encode(
                pixels=get_value_at_index(addnode_126, 0),
                vae=get_value_at_index(vaeloader_80, 0),
            )

            loadimage_140 = loadimage.load_image(image="Watermark1.png")

            ollamaconnectivityv2_160 = ollamaconnectivityv2.ollama_connectivity(
                url="http://127.0.0.1:11435",
                model="mistral-small3.1:24b",
                keep_alive=5,
                keep_alive_unit="minutes",
            )

            ollamageneratev2_146 = ollamageneratev2.ollama_generate_v2(
                system="",
                prompt="You are a visual analysis and prompt-engineering specialist. You are shown a single, clear, frontal image of a plush toy animal. Your goal is to:\n\nAnalyze the image carefully and describe the plush animal's external anatomical features in exhaustive detail, including:\n\nThe type of animal it represents (e.g., monkey, bear, rabbit).\n\nThe posture and orientation (e.g., sitting, standing, crouching, head facing forward or tilted).\n\nProportions of the limbs (length of arms vs. legs, relative size of hands and feet).\n\nSize and positioning of ears, eyes, nose, mouth, and tail (if visible).\n\nAny notable stylized features (e.g., exaggerated hands, large eyes, round head, oversized feet). Do not mention colors of the original image.\n\nBased solely on this image description, construct a FLUX prompt for generating a realistic, medically plausible X-ray image of the plush animal as if it had a biological internal structure.\n\nThe FLUX prompt must meet the following criteria:\n\nAccurately reflect the external anatomy, proportions, and posture of the plush animal.\n\nDepict a detailed, friendly skeletal system corresponding to the animal’s body shape and pose. The bones should appear realistic but adapted to the exaggerated or cartoonish proportions of the plush.\n\nLimbs, hands, feet, ears, and tail (if present) must have anatomically plausible bone structures, adjusted to match the stylized features seen in the image.\n\nInclude only bones and soft-tissue glow; no internal organs or disturbing anatomical details.\n\nSoft-tissue glow should create a gentle, non-creepy X-ray effect, emphasizing bone contrast while allowing for a subtle outline of the body and limbs.\n\nPresent the X-ray in a clean, clinical radiographic style with a neutral or black background, without any horror elements or unsettling features.\n\nYour output must be only the final FLUX prompt, written in natural language, descriptive, precise, and fully self-contained.\n\nExample output structure (you must replace placeholders with accurate descriptions from the image):\n\n“A realistic medical-style X-ray image of a [detailed animal type and description], with its [head facing direction], [pose], [detailed limb proportions], and [specific features like ear size, hand shape, tail presence]. The X-ray reveals a biologically plausible skeletal structure matching its proportions, with elongated bones in the [arms/legs], defined phalanges in [hands/feet], a simplified ribcage, vertebral column following the posture, and structural support in the [ears/tail if applicable]. The soft tissue appears as a gentle, semi-transparent glow outlining the body and limbs. The image is set on a clean, black radiographic background, realistic and educational in style, without any creepy or unsettling features.”",
                filter_thinking=True,
                keep_context=False,
                format="text",
                connectivity=get_value_at_index(ollamaconnectivityv2_160, 0),
                images=get_value_at_index(image_rembg_remove_background_91, 0),
            )

            cliptextencode_163 = cliptextencode.encode(
                text=get_value_at_index(ollamageneratev2_146, 0),
                clip=get_value_at_index(t5tokenizeroptions_82, 0),
            )

            for q in range(1):
                ksampler_94 = ksampler.sample(
                    seed=random.randint(1, 2**64),
                    steps=15,
                    cfg=4,
                    sampler_name="euler",
                    scheduler="beta",
                    denoise=0.8000000000000002,
                    model=get_value_at_index(loraloadermodelonly_159, 0),
                    positive=get_value_at_index(cliptextencode_163, 0),
                    negative=get_value_at_index(cliptextencode_75, 0),
                    latent_image=get_value_at_index(vaeencode_97, 0),
                )

                vaedecode_79 = vaedecode.decode(
                    samples=get_value_at_index(ksampler_94, 0),
                    vae=get_value_at_index(vaeloader_80, 0),
                )

                imagecompositemasked_139 = imagecompositemasked.composite(
                    x=0,
                    y=0,
                    resize_source=False,
                    destination=get_value_at_index(loadimage_140, 0),
                    source=get_value_at_index(vaedecode_79, 0),
                    mask=get_value_at_index(loadimage_140, 1),
                )

                textonimage_142 = textonimage.apply_text(
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
                    image=get_value_at_index(imagecompositemasked_139, 0),
                )

            result = convert_image(textonimage_142)
            
            return result