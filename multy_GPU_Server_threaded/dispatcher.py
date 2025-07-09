import os
import sys

# Add the parent directory to sys.path to handle relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# import the workflow class you want to add
from functions import Functions
from workflow_scripts.FLUX_Kontext import FLUX_Kontext
from workflow_scripts.IP_Adapter_SDXL import IP_Adapter_SDXL


class WorkflowDispatcher:
    """Dispatcher that creates and manages workflow objects."""

    # Step 1: Put the name of your workflow and the name of the class into workflow_class (can be the same name)
    # Step 2: Import your workflow class at the top of this file
    # Step 3: Copy the name of your new workflow and go to main.py to the main function
    
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

        # Map workflow names to their corresponding classes
        self.workflow_class = {
            "FLUX_Kontext": FLUX_Kontext, 
            "IP_Adapter_SDXL": IP_Adapter_SDXL
            # example 
            # "name_of_your_workflow": YourWorkflowClassName,
        }


    def create_workflow_obj(self, gpu_id: int = None):
        """Create and return all workflow objects bound to a specific GPU."""
        workflows = {}
        for name, cls in self.workflow_class.items():
            workflow_instance = cls(self.functions, gpu_id=gpu_id)
            workflow_instance.NODE_CLASS_MAPPINGS = self.NODE_CLASS_MAPPINGS  # Add NODE_CLASS_MAPPINGS
            workflows[name] = workflow_instance
        return workflows
        
    def create_single_workflow_obj(self, gpu_id : int, workflow : str):
        """Create and return a single workflow object bound to a specific GPU."""
        if workflow not in self.workflow_class:
            raise ValueError(f"Workflow '{workflow}' is not defined in the dispatcher.")
        
        cls = self.workflow_class[workflow]
        workflow_instance = cls(self.functions, gpu_id=gpu_id)
        workflow_instance.NODE_CLASS_MAPPINGS = self.NODE_CLASS_MAPPINGS  # Add NODE_CLASS_MAPPINGS
        return workflow_instance     