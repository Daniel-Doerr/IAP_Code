import os
import sys

# Add the parent directory to sys.path to handle relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# import the workflow class you want to add
from workflow_scripts.FLUX_Kontext import FLUX_Kontext
from workflow_scripts.IP_Adapter_SDXL import IP_Adapter_SDXL


class WorkflowDispatcher:
    """Dispatcher that creates and manages workflow objects."""

    # Step 1: Put the name of your workflow and the name of the class into workflow_class (can be the same name)
    # Step 2: Import your workflow class at the top of this file
    # Step 3: Copy the name of your new workflow and go to main.py to the main function
    
    def __init__(self):
        # Map workflow names to their corresponding classes
        self.workflow_class = {
            "FLUX_Kontext": FLUX_Kontext, 
            "IP_Adapter_SDXL": IP_Adapter_SDXL
            # example 
            # "name_of_your_workflow": YourWorkflowClassName,
        }
        self.current_workflow = None

    def create_workflow(self, workflow_name: str):
        """Create and return a workflow object."""
        workflow_class = self.workflow_class.get(workflow_name)
        if workflow_class:
            # Create an instance of the workflow class
            workflow_instance = workflow_class()
            self.current_workflow = workflow_instance
            return workflow_instance
        else:
            print(f"Unknown workflow: {workflow_name}")
            return None