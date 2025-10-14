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
        Initializes the Functions class.
        This class bundles together various utility functions used across the application,
        such as file path manipulation, ComfyUI integration, and data conversion.
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
            # Standard access for lists or simple dictionaries
            return obj[index]
        except KeyError:
            # Fallback for nested results, common in some custom nodes
            return obj["result"][index]


    def find_path(self, name: str, path: str = None) -> str:
        """
        Recursively looks at parent folders starting from the given path until it finds the given name.
        Returns the path as a Path object if found, or None otherwise.
        """
        # If no starting path is provided, use the current working directory
        if path is None:
            path = os.getcwd()

        # Check if the target name exists in the current directory
        if name in os.listdir(path):
            path_name = os.path.join(path, name)
            print(f"{name} found: {path_name}")
            return path_name

        # Move up to the parent directory for the next iteration
        parent_directory = os.path.dirname(path)

        # If the parent is the same as the current path, we have reached the filesystem root
        if parent_directory == path:
            return None

        # Continue the search in the parent directory
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
            # Attempt to import the configuration loader from the main script
            from main import load_extra_path_config
        except ImportError:
            # Fallback if the main script structure changes or is not available
            print(
                "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
            )
            try:
                from utils.extra_config import load_extra_path_config
            except ImportError:
                print("Could not import load_extra_path_config from utils.extra_config.")
                return

        # Find the configuration file
        extra_model_paths = self.find_path("extra_model_paths.yaml")

        if extra_model_paths is not None:
            # Load the paths if the file is found
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

        # A new asyncio event loop is required for the server and queue setup
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # These instances are necessary for the node initialization process to succeed
        server_instance = server.PromptServer(loop)
        execution.PromptQueue(server_instance)

        # This function discovers and registers the custom nodes
        init_extra_nodes()




    def converte_image(self, generatedImage: Any) -> io.BytesIO:
        """
        Converts a generated image tensor into a PNG byte buffer.

        Args:
            generatedImage: The raw output from a ComfyUI node, typically a PyTorch tensor.

        Returns:
            An in-memory binary buffer (io.BytesIO) containing the PNG image data.
        """
        # Extract the tensor from the potentially nested result
        image = self.get_value_at_index(generatedImage, 0)
        # Move tensor to CPU and convert to a NumPy array
        arr = image.cpu().numpy()

        # Remove singleton dimensions (e.g., [1, H, W, C] -> [H, W, C])
        arr = np.squeeze(arr)

        # Transpose if the channel is the first dimension (e.g., [C, H, W] -> [H, W, C])
        if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
            arr = np.transpose(arr, (1, 2, 0))

        # Normalize from [0, 1] float to [0, 255] uint8 range
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)

        # Create a PIL Image from the array
        img = Image.fromarray(arr)

        # Save the image to an in-memory buffer as PNG
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        img_buffer.seek(0)  # Rewind the buffer to the beginning
        return img_buffer


    def format_text_for_field(self, text, line_length=13, lines=3):
        """
        Formats a string into a fixed number of lines with a maximum length per line.
        Useful for displaying text in constrained UI elements.

        Args:
            text: The input string to format.
            line_length: The maximum number of characters allowed per line.
            lines: The exact number of lines the output should have.

        Returns:
            A single string with newlines, padded with empty lines if necessary.
        """
        import textwrap
        # Wrap the text to the specified line length
        wrapped = textwrap.wrap(text, width=line_length)
        # Truncate or pad the list of lines to ensure it has the exact desired length
        wrapped = wrapped[:lines] + [""] * (lines - len(wrapped))
        # Join the lines back into a single string
        return "\n".join(wrapped)
    
    
    def get_path_from_bytes(self, image_bytes: bytes) -> str:
        """
        Saves image bytes to a temporary file and returns the file path.
        This is useful when a node or function requires a file path instead of in-memory data.

        Args:
            image_bytes: The raw bytes of the image.

        Returns:
            The absolute path to the newly created temporary image file.
        """
        # Create a temporary file that is not deleted on close, with a .png suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(image_bytes)
            return tmp.name
