import click
import os
import math
import random
import sys
import torch
from typing import Sequence, Mapping, Any, Union

@click.command()
@click.option('--cfg-1', type=str, prompt="Enter cfg_1 values as 'start,end,increment'", 
              default="8.0,10.0,1.0", show_default=True, 
              help="Comma-separated values for cfg_1: start,end,increment (e.g., 0.0,100.0,1.0)")

@click.option('--cfg-2', type=str, prompt="Enter cfg_2 values as 'start,end,increment'", 
              default="6.0,8.0,1.0", show_default=True, 
              help="Comma-separated values for cfg_2: start,end,increment (e.g., 0.0,100.0,1.0)")

@click.option('--total-steps', type=int, prompt="Enter the total number of steps", 
              default=50, show_default=True, help="Total number of steps")

@click.option('--first-steps', type=int, prompt="Enter the number of steps for the first ksampler", 
              default=30, show_default=True, help="Number of steps for the first ksampler")


@click.option('--lora-strength', type=str, prompt="Enter lora strength values as 'start,end,increment' max:100 (if increment is 0, only one value will be used)", 
              default="0.7,0.0,0.0", show_default=True, 
              help="Comma-separated values for lora strenght: start,end,increment (e.g., 0.7,0.0,0.0)")


@click.option('--controlnet-strength', type=str, prompt="Enter controlnet strength values as 'start,end,increment' max:10 (if increment is 0, only one value will be used)", 
              default="0.7,0.0,0.0", show_default=True, 
              help="Comma-separated values for controlnet strenght: start,end,increment (e.g., 0.7,0.0,0.0)")


@click.option('--output-dir', type=str, prompt="Enter the output directory name (default: output_images)", 
              default="output_images", show_default=True, help="Output directory name")


def main(cfg_1, cfg_2, total_steps, first_steps, lora_strength, controlnet_strength, output_dir):
    cfg_1_start, cfg_1_end, cfg_1_increment = map(float, cfg_1.split(','))
    cfg_2_start, cfg_2_end, cfg_2_increment = map(float, cfg_2.split(','))
    lora_start, lora_end, lora_increment = map(float, lora_strength.split(','))
    controlnet_start, controlnet_end, controlnet_increment = map(float, controlnet_strength.split(','))

    ## if the increment is 0, only go through the loop once
    def check_if_zero(start: float, end: float, increment: float):
        if increment == 0:
            increment = 1.0
            end = start + 0.5
        return start, end, increment
        
    cfg_1_start, cfg_1_end, cfg_1_increment = check_if_zero(cfg_1_start, cfg_1_end, cfg_1_increment)
    cfg_2_start, cfg_2_end, cfg_2_increment = check_if_zero(cfg_2_start, cfg_2_end, cfg_2_increment)
    lora_start, lora_end, lora_increment = check_if_zero(lora_start, lora_end, lora_increment)
    controlnet_start, controlnet_end, controlnet_increment = check_if_zero(controlnet_start, controlnet_end, controlnet_increment)


    ## Check if the start and end values are in the right order
    def swap_if_needed(start: float, end: float):
        if start > end:
            tmp = start
            start = end
            end = tmp
        return start, end
    
    cfg_1_start, cfg_1_end = swap_if_needed(cfg_1_start, cfg_1_end)
    cfg_2_start, cfg_2_end = swap_if_needed(cfg_2_start, cfg_2_end)
    lora_start, lora_end = swap_if_needed(lora_start, lora_end)
    controlnet_start, controlnet_end = swap_if_needed(controlnet_start, controlnet_end)

    ## Check if the increments are negative
    if cfg_1_increment < 0:
        cfg_1_increment = -cfg_1_increment
    if cfg_2_increment < 0:
        cfg_2_increment = -cfg_2_increment
    if lora_increment < 0:
        lora_increment = -lora_increment
    if controlnet_increment < 0:
        controlnet_increment = -controlnet_increment


    # Calculate total folders and images
    cfg_folders = (math.floor((cfg_1_end - cfg_1_start) / cfg_1_increment)+1) * (math.floor((cfg_2_end - cfg_2_start) / cfg_2_increment)+1)
    lora_folders = math.floor((lora_end - lora_start) / lora_increment)+1
    controlnet_folders = math.floor((controlnet_end - controlnet_start) / controlnet_increment)+1
    total_folders = cfg_folders * lora_folders * controlnet_folders
    total_images = total_folders * 4
    total_size = total_images  # Assuming each image is 1 MB
    total_time = total_images * total_steps * 0.27 + 15  # Estimated time calculation

    # Print summary
    click.echo(f"Total folders to be created: {total_folders}")
    click.echo(f"Total images to be generated: {total_images}")
    click.echo(f"Estimated size of the output folder: {total_size} MB")
    click.echo(f"Estimated time to generate the images: {total_time:.2f} seconds ({total_time/60:.2f} minutes, {total_time/3600:.2f} hours)")

    continue_choice = input("Do you want to continue? (yes/no): ").strip().lower()
    if continue_choice != "yes":
        print("Exiting the program.")
        exit()

    ## Create output directory
    index = 1
    base_output_dir = output_dir
    while os.path.exists(output_dir):
        output_dir = f"{base_output_dir}_{index}"
        index += 1
    os.makedirs(output_dir)
    click.echo(f"Output directory created: {output_dir}")

    ## Save settings to a file
    settings_file_path = os.path.join(output_dir, "settings.txt")
    with open(settings_file_path, "w") as settings_file:
        settings_file.write(f"cfg_1_start: {cfg_1_start}\n")
        settings_file.write(f"cfg_1_end: {cfg_1_end}\n")
        settings_file.write(f"cfg_1_increment: {cfg_1_increment}\n")
        settings_file.write(f"cfg_2_start: {cfg_2_start}\n")
        settings_file.write(f"cfg_2_end: {cfg_2_end}\n")
        settings_file.write(f"cfg_2_increment: {cfg_2_increment}\n")
        settings_file.write(f"total_steps: {total_steps}\n")
        settings_file.write(f"first_steps: {first_steps}\n")
        settings_file.write(f"lora_start: {lora_start}\n")
        settings_file.write(f"lora_end: {lora_end}\n")
        settings_file.write(f"lora_increment: {lora_increment}\n")
        settings_file.write(f"controlnet_start: {controlnet_start}\n")
        settings_file.write(f"controlnet_end: {controlnet_end}\n")
        settings_file.write(f"controlnet_increment: {controlnet_increment}\n")
        settings_file.write(f"total_folders: {total_folders}\n")
        settings_file.write(f"total_images: {total_images}\n")
        settings_file.write(f"total_estimated_time: {total_time} seconds\n")
        settings_file.write(f"output_dir: {output_dir}\n")
    click.echo(f"Settings saved to: {settings_file_path}")


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

    
    with torch.inference_mode():  
        import_custom_nodes()  
        ###### Load all Nodes ######
        ## loads Stable Diffusion node
        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()

        ## load cliptextencode node
        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()

        ## load ipadapter unified loader node
        ipadapterunifiedloader = NODE_CLASS_MAPPINGS["IPAdapterUnifiedLoader"]()

        ## load iamge batch node (for combinding reference images)
        imagebatch = NODE_CLASS_MAPPINGS["ImageBatch"]()

        ## load ipadapter node (image processing of reference image)
        ipadapter = NODE_CLASS_MAPPINGS["IPAdapter"]()

        ## style of controlnet node (depth map)
        depthanythingpreprocessor = NODE_CLASS_MAPPINGS["DepthAnythingPreprocessor"]()

        ## load controlnetapplyadvanced node
        controlnetapplyadvanced = NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]()

        ## load ksampleradvanced node
        ksampleradvanced = NODE_CLASS_MAPPINGS["KSamplerAdvanced"]()

        ## load vaedecode node
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

        ## load saveimage node (unnecessary if I save them separately)
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        ## load second style of controlnet node (unnecessary)
        cannyedgepreprocessor = NODE_CLASS_MAPPINGS["CannyEdgePreprocessor"]()

        ## load loadimage node
        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()    
        
        ## unconnected vae loader (unnecessary)
        vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()

        ## load controlnet node 
        controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()

        ## load controlnet node 
        controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()

        ## load image size node
        emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()

        ## load LoRA node
        loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
        ############



        ###### Load Models ######
        ## loads Stable Diffusion model
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name="sd_xl_base_1.0.safetensors"
        )

        ## loads Stable Diffusion model (there are two, could be one)
        checkpointloadersimple_12 = checkpointloadersimple.load_checkpoint(
            ckpt_name="sd_xl_base_1.0.safetensors"
        )

        ## load controlnet model 
        controlnetloader_52 = controlnetloader.load_controlnet(
            control_net_name="SDXL/controlnet-union-sdxl-1.0/diffusion_pytorch_model_promax.safetensors"
        )
        ############
        


        ###### Load output size ######
        ## sets size of output image
        emptylatentimage_5 = emptylatentimage.generate(
            width=1024, height=1024, batch_size=1
        )
        ############


        ###### Load prompts ######
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
        ############



        ###### Load and batch reference iamges ######
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
        ############



        image = ["image1.png", "image2.png", "image3.png", "image4.png"]
        index = 0

    
        lora_strength = lora_start
        while lora_strength <= lora_end:
            ## load LoRA with strength of model and clip 
            loraloader_68 = loraloader.load_lora(
                lora_name="xraylorasdxl.safetensors",
                strength_model=lora_strength,
                strength_clip=1,
                model=get_value_at_index(checkpointloadersimple_4, 0),
                clip=get_value_at_index(checkpointloadersimple_4, 1),
            )

            ## load ipadapter strength model
            ipadapterunifiedloader_63 = ipadapterunifiedloader.load_models(
                preset="VIT-G (medium strength)",
                model=get_value_at_index(loraloader_68, 0),
            )
            
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
            

            controlnet_strength = controlnet_start
            while controlnet_strength <= controlnet_end:
                sub_dir = os.path.join(output_dir, f"lora_{lora_strength}_controlnet_{controlnet_strength}")
                if not os.path.exists(sub_dir):
                    os.makedirs(sub_dir)

                cfg_1 = cfg_1_start
                while cfg_1 <= cfg_1_end:
                    
                    cfg_2 = cfg_2_start
                    while cfg_2 <= cfg_2_end:
                        
                        ## Create a subdirectory for each combination of cfg_1 and cfg_2
                        sub_sub_dir = os.path.join(sub_dir, f"cfg_{cfg_1}_{cfg_2}")
                        if not os.path.exists(sub_sub_dir):
                            os.makedirs(sub_sub_dir)

                        for i in range(4):
                            ## input image
                            loadimage_50 = loadimage.load_image(
                                image=image[i]
                            )

                            ## load unconnected vae loader (unnecessary)
                            vaeencode_51 = vaeencode.encode(
                                pixels=get_value_at_index(loadimage_50, 0),
                                vae=get_value_at_index(checkpointloadersimple_4, 2),
                            )

                            ## creating depth image for controlnet
                            depthanythingpreprocessor_55 = depthanythingpreprocessor.execute(
                                ckpt_name="depth_anything_vitb14.pth",
                                resolution=512,
                                image=get_value_at_index(loadimage_50, 0),
                            )

                            ## applies the controlnet 
                            controlnetapplyadvanced_54 = controlnetapplyadvanced.apply_controlnet(
                                strength=controlnet_strength,
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
                                steps=total_steps,
                                cfg=cfg_1,
                                sampler_name="euler",
                                scheduler="sgm_uniform",
                                start_at_step=0,
                                end_at_step=first_steps,
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
                                steps=total_steps,
                                cfg=cfg_2,
                                sampler_name="euler",
                                scheduler="normal",
                                start_at_step=first_steps,
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

                            # Save the images in the respective subdirectory
                            saveimage.save_images(
                                filename_prefix=os.path.join(sub_sub_dir, f"ComfyUI_{index}"),
                                images=get_value_at_index(vaedecode_17, 0),
                            )
                            index += 1
                        cfg_2 += cfg_2_increment
                    cfg_1+= cfg_1_increment
                controlnet_strength += controlnet_increment
            lora_strength += lora_increment



if __name__ == "__main__":
    main()