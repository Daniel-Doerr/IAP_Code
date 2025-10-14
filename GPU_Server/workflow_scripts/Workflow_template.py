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


# TODO: Change the class name to your workflow name, e.g. "FLUX_Kontext"
class YourWorkflowName:
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
            # How to implement your own workflow:
            # You need a ComfyUI workflow and a tool to export it as a Python script (e.g., https://github.com/pydn/ComfyUI-to-Python-Extension).
            #
            # 1. Export the workflow into a Python script.
            # 2. Open the exported file, go to the 'main' function, and copy all the code from inside the 'with torch.inference_mode():' block.
            #
            # To optimize performance by loading models only once (recommended):
            # 3. Search for "NODE_CLASS_MAPPINGS" and move all related initializations to the 'load_once' function.
            # 4. Search for ".safetensors" and move all code lines that load these files into the 'load_once' function.
            # 5. Search for "model_name" or "modelloader" and move those lines into the 'load_once' function as well.
            #    Optionally, move any static assets (e.g., watermarks) that don't depend on the input image to 'load_once'.
            #
            # 6. In the 'generate' function, find the input image loading step (e.g., 'loadimage_X') and change it to:
            #    loadimage_X = loadimage.load_image(image=tmp_path)
            #    Ensure paths for constant images (like IP-Adapter inputs or watermarks) are correct.
            #    For text overlays, use format_text_for_field(your_text_input, line_length, num_lines) from 'functions.py'.
            #
            # 7. Remove the 'saveimage_Y' step at the end, as the image is returned as a buffer, not saved.
            #
            # 8. In the 'convert_image(generatedImage)' call, replace 'generatedImage' with the variable holding the final processed image
            #    (e.g., 'convert_image(textonimage_142)').
            #
            # 9. Rename the class and the file to match your workflow's name (e.g., "FLUX_Kontext").
            #
            # Final steps in 'dispatcher.py':
            # 10. Import your new workflow: from workflow_scripts.[workflow_name] import [ClassName]
            # 11. Add your workflow to the 'workflow_class' dictionary: "[workflow_name]": [ClassName]
            #
            # You can also set your new workflow as the default in 'main.py'.

            

 
            result = convert_image(generatedImage)
            
            return result