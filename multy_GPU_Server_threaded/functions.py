import io
import tempfile
from typing import Any
import numpy as np
from PIL import Image
import os
import sys
from typing import Union, Sequence, Mapping


class Functions:
    def __init__(self):
        """
        Initialize the Functions class.
        This class contains utility functions for various operations.
        """
        pass

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




    def converte_image(self, generatedImage: Any) -> io.BytesIO:
        image = self.get_value_at_index(generatedImage, 0)
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


    def format_text_for_field(self, text, line_length=13, lines=3):
        """
        Formats a single string into a fixed number of lines, each with a maximum length.

        Args:
            text (str): The input text to format.
            line_length (int): Maximum number of characters per line.
            lines (int): Total number of lines to return.

        Returns:
            str: The formatted text with line breaks, padded with empty lines if needed.
        """
        import textwrap
        wrapped = textwrap.wrap(text, width=line_length)
        # Ensure exactly 'lines' lines (pad with empty strings if needed)
        wrapped = wrapped[:lines] + [""] * (lines - len(wrapped))
        return "\n".join(wrapped)
    
    
    def get_path_from_bytes(self, image_bytes: bytes) -> str:     
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(image_bytes)
            return tmp.name
