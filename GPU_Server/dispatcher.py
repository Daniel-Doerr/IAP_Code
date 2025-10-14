import os
import sys

# Ensure the project's root directory is in the system path.
# This allows for consistent module resolution (e.g., importing `functions`).
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# --- Workflow Class Imports ---
# Each new workflow must be imported here to be recognized by the dispatcher.
from workflow_scripts.FLUX_Kontext import FLUX_Kontext
from workflow_scripts.IP_Adapter_SDXL import IP_Adapter_SDXL
from workflow_scripts.ChromaV44 import ChromaV44
from functions import Functions


# Final steps in 'dispatcher.py':
# 10. Import your new workflow: from workflow_scripts.[workflow_name] import [ClassName]
# 11. Add your workflow to the 'workflow_class' dictionary: "[workflow_name]": [ClassName]

class WorkflowDispatcher:
    """
    Manages the lifecycle of different ComfyUI workflow classes.
    
    This class is responsible for:
    - Setting up the necessary paths for ComfyUI and its custom nodes.
    - Registering all available workflow classes.
    - Instantiating workflow objects upon request.
    """
    
    def __init__(self):
        """
        Initializes the dispatcher and prepares the environment for ComfyUI.
        """
        self.functions = Functions()
        
        # Set up the environment by adding required paths for ComfyUI to function correctly.
        self.functions.add_comfyui_directory_to_sys_path()
        self.functions.add_extra_model_paths()
        
        # Import ComfyUI's node mappings, which are essential for running workflows.
        try:
            from nodes import NODE_CLASS_MAPPINGS
            self.NODE_CLASS_MAPPINGS = NODE_CLASS_MAPPINGS
        except ImportError:
            print("Warning: Could not import NODE_CLASS_MAPPINGS from ComfyUI.")
            self.NODE_CLASS_MAPPINGS = {}
            
        # Discover and load any custom nodes.
        self.functions.import_custom_nodes()

        # --- Workflow Registration ---
        # Add an entry to this dictionary mapping a unique string name to the workflow's class definition.
        self.workflow_class = {
            "FLUX_Kontext": FLUX_Kontext, 
            "IP_Adapter_SDXL": IP_Adapter_SDXL,
            "ChromaV44": ChromaV44,
            # Example:
            # "YourWorkflowName": YourWorkflowClassName,
        }


    def create_workflow_obj(self):
        """
        Creates an instance of every registered workflow class.

        Returns:
            dict: A dictionary mapping workflow names to their instantiated objects.
        """
        workflows = {}
        for name, cls in self.workflow_class.items():
            workflow_instance = cls(self.functions)
            # Inject the node mappings into the instance for its use.
            workflow_instance.NODE_CLASS_MAPPINGS = self.NODE_CLASS_MAPPINGS
            workflows[name] = workflow_instance
        return workflows
        

    def create_single_workflow_obj(self, workflow: str):
        """
        Creates an instance of a single, specified workflow class.

        Args:
            workflow (str): The name of the workflow to create.

        Returns:
            An instance of the requested workflow class.
            
        Raises:
            ValueError: If the requested workflow name is not registered.
        """
        if workflow not in self.workflow_class:
            raise ValueError(f"Workflow '{workflow}' is not defined in the dispatcher.")
        
        cls = self.workflow_class[workflow]
        workflow_instance = cls(self.functions)
        # Inject the node mappings into the instance.
        workflow_instance.NODE_CLASS_MAPPINGS = self.NODE_CLASS_MAPPINGS
        return workflow_instance    