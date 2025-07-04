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
class YourWorkflowName:
    # You don't need to change this function
    def __init__(self):
        self.functions = Functions()
        self.functions.add_comfyui_directory_to_sys_path()
        self.functions.add_extra_model_paths()
        try:
            from nodes import NODE_CLASS_MAPPINGS
            self.NODE_CLASS_MAPPINGS = NODE_CLASS_MAPPINGS
        except ImportError:
            print("Could not import NODE_CLASS_MAPPINGS from nodes.")
            self.NODE_CLASS_MAPPINGS = {}
        self.functions.import_custom_nodes()
        self.config = self.load_once()


    def load_once(self):
        """Safe time by loading surtain nodes only once."""
        # follow the steps in the generate function

        NODE_CLASS_MAPPINGS = self.NODE_CLASS_MAPPINGS
        get_value_at_index = self.functions.get_value_at_index

        with torch.inference_mode():
            print("Loading nodes...")
            # follow the steps 3, 4, 5 and Optional from the generate function



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

            

            result = converte_image(generatedImage)
            
            return result