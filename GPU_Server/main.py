import requests
import time
import os
import click
import toml
import sys
import torch
import signal

from dispatcher import WorkflowDispatcher

# Global flag to handle clean shutdown
shutdown_requested = False


def signal_handler(sig, frame):
    global shutdown_requested
    print("\n" + "="*60)
    print("SHUTDOWN REQUESTED - Stopping gracefully...")
    print("="*60)
    shutdown_requested = True


# Load configuration from config.toml file
# This function reads the configuration file and returns the server settings
def load_config():
    """Load configuration from config.toml file."""
    config_path = os.path.join(os.path.dirname(__file__), "config.toml")
    try:
        with open(config_path, 'r') as f:
            config_data = toml.load(f)
            return config_data.get("server", {})
    except FileNotFoundError:
        print(f"Config file not found at {config_path}. Shutting down.")
        exit(1)
    except toml.TomlDecodeError:
        print(f"Invalid TOML in config file {config_path}. Shutting down.")
        exit(1)

# Load configuration
config = load_config()
WEB_SERVER = config.get("WEB_SERVER") # Load web server URL from configuration file
password = config.get("password") # Load password from configuration file


def restart_program():
    """Restart the entire program to completely clear GPU memory"""
    global shutdown_requested
    
    # Check if shutdown was requested - don't restart if user wants to quit
    if shutdown_requested:
        print("Shutdown requested - NOT restarting program")
        exit(0)
    
    try:
        import subprocess
        print("=" * 60)
        print("RESTARTING PROGRAM TO CLEAR GPU MEMORY")
        print("=" * 60)
        
        # Get current script path and arguments
        script_path = sys.argv[0]
        script_args = sys.argv[1:]
        
        # Start new process
        subprocess.Popen([sys.executable, script_path] + script_args)
        
        # Exit current process
        print("New process started. Exiting current process...")
        exit(0)
        
    except Exception as e:
        print(f"Failed to restart program: {e}")
        print("Continuing with current process...")


def cleanup_gpu_memory():
    try:
        if torch.cuda.is_available():
            print("Starting GPU memory cleanup...")
            
            # First try to access ComfyUI's model management
            try:
                # Import ComfyUI's model management
                import comfy.model_management
                
                # Unload all models from GPU
                print("Unloading all models from ComfyUI model management...")
                comfy.model_management.unload_all_models()

                # Free as much memory as possible
                try:
                    device = comfy.model_management.get_torch_device()
                    comfy.model_management.free_memory(1e30, device)
                except TypeError:
                    # Fallback for older ComfyUI versions where signature differed
                    comfy.model_management.free_memory()

                comfy.model_management.soft_empty_cache()

                # Clear model cache completely
                if hasattr(comfy.model_management, 'cleanup_models'):
                    comfy.model_management.cleanup_models()

                print("ComfyUI models unloaded")
                
            except ImportError:
                print("ComfyUI model_management not available, skipping model unload")
            except Exception as e:
                print(f"Error unloading ComfyUI models: {e}")
            
            # Clear PyTorch cache multiple times
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear cache again after garbage collection
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            print("GPU memory cleanup completed")
            
    except Exception as e:
        print(f"Error during GPU cleanup: {e}")


def get_access_token(password: str) -> str:
    """Authenticate with the backend and get a JWT access token."""
    url = f"{WEB_SERVER}/token"
    data = {"password": password}
    response = requests.post(url, data=data)
    response.raise_for_status()
    return response.json()["access_token"]


def poll_job():
    global password, shutdown_requested

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Try to get access token 
    token = None
    while token is None and not shutdown_requested:
        try:
            token = get_access_token(password)
            print("Successfully obtained access token")
        except Exception as e:
            print(f"Failed to get access token: {e}")
            print("Retrying in 10 seconds...")
            time.sleep(10)
            
    if shutdown_requested:
        print("Shutdown requested during token acquisition")
        return

    # Set up headers for authentication
    headers = {"Authorization": f"Bearer {token}"}

    # Create a WorkflowDispatcher instance to manage workflows
    dispatcher = WorkflowDispatcher()
    # Create workflow objects for each workflow type without loading the models yet
    workflow_objects = dispatcher.create_workflow_obj()

    last_workflow = None # Variable to keep track of the last used workflow object

    no_job_count = 1

    while not shutdown_requested:
        try:
            # Poll for a job from the web server
            response = requests.get(f"{WEB_SERVER}/job", headers=headers)
            # Handle unauthorized access
            if response.status_code == 401:
                print("Unauthorized, refreshing token...")
                token = get_access_token(password)
                headers["Authorization"] = f"Bearer {token}"
                time.sleep(2)
                continue
            # Handle no job available
            if response.status_code == 204:
                print("No job received...")
                no_job_count += 1

                if no_job_count >= 960: # no jobs for 1 hour
                    # Check if any workflow was loaded and clean up
                    print("Total sleep mode reached! Polling again in 1 minute.")

                    if last_workflow is not None:
                        print("Resetting program to free GPU memory.")
                        print("(Press Ctrl+C to terminate the program)")
                        restart_program()
                        
                    for _ in range(2):  # 1 minute sleep with shutdown check
                        if shutdown_requested:
                            return
                        time.sleep(30)

                elif no_job_count >= 900: # no jobs for 30 minutes
                    print("Going into sleep mode! Polling again in 30 seconds.")
                    if last_workflow is not None:
                        cleanup_gpu_memory()
                    for _ in range(3):  # 30 seconds sleep with shutdown check
                        if shutdown_requested:
                            return
                        time.sleep(10)

                else:
                    if shutdown_requested: #shutdown check
                        return
                    time.sleep(2)
                continue

            no_job_count = 1

            # extract image data
            image_bytes = response.content

            # Header-Metadata
            job_id = response.headers.get("img_id")
            first_name = response.headers.get("first_name")
            last_name = response.headers.get("last_name")
            animal_name = response.headers.get("animal_name")
            animal_type = response.headers.get("animal_type")
            workflow = response.headers.get("workflow")

            # Check if all required data is present
            if not job_id or not image_bytes:
                print("No valid job data received, skipping...")
                time.sleep(2)
                continue

            # Check if the workflow is known
            if workflow not in workflow_objects:
                print(f"Unknown workflow: {workflow}. Available: {list(workflow_objects.keys())}")
                workflow = "FLUX_Kontext"  # Default to a known workflow
                # time.sleep(2)
                # continue
            
            # Handle animal type "other" for prompting
            if animal_type == "other":
                animal_type = "stuffed animal"

            # Select the appropriate workflow object
            if workflow != last_workflow: 
                # Delete the last workflow object to free memory
                if last_workflow is not None:
                    print(f"Switching from {last_workflow} to {workflow}, cleaning GPU memory...")
                    try:
                        # Clear the workflow object's internal state
                        workflow_objects[last_workflow].__dict__.clear()
                        del workflow_objects[last_workflow]
                        
                        # Use the dedicated cleanup function
                        cleanup_gpu_memory()
                        
                        # Re-create the last_workflow object and update the dictionary
                        workflow_objects[last_workflow] = dispatcher.create_single_workflow_obj(last_workflow)
                        print(f"GPU memory cleaned and {last_workflow} workflow recreated")
                    except Exception as e:
                        print(f"Error during workflow cleanup: {e}")
                        # Re-create anyway
                        workflow_objects[last_workflow] = dispatcher.create_single_workflow_obj(last_workflow)
                
                # Load the new workflow
                print(f"Loading workflow: {workflow}")
                workflow_objects[workflow].start_load_once()

            # set the last_workflow object to the current workflow
            last_workflow = workflow
            
            print(f"Job received: {job_id}; Using workflow: {workflow}")
            print(f"Patient: {first_name} {last_name}, Animal: {animal_name}, AnimalType: {animal_type}")

            for i in range(1):
                start_time = time.time()
                img_buffer = workflow_objects[workflow].generate(workflow, image_bytes, animal_type, first_name, last_name, animal_name)
                elapsed_time = time.time() - start_time
                print(f"Time taken to generate image: {elapsed_time:.2f} seconds")

                files = {
                    "result": ("result.png", img_buffer.getvalue(), "image/png"),
                }
                data = {
                    "image_id": job_id,
                }

                # Send the result back to the web server
                res = requests.post(f"{WEB_SERVER}/job", headers={"Authorization": f"Bearer {token}"}, files=files, data=data)
                print("Result sent:", res.status_code, res.text)

        # Handle exceptions during job processing
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received - shutting down gracefully...")
            shutdown_requested = True
            break
        except Exception as e:
            print("Error:", e)
            if shutdown_requested:
                break
            # Clean up GPU memory on error using dedicated function
            cleanup_gpu_memory()
            time.sleep(3)
    
    # Clean shutdown
    print("Program terminated gracefully")



@click.command()
@click.option('-test', '-t', is_flag=True, help='Run in test mode')
def main(test):
    # Remove the -test flag from sys.argv so ComfyUI doesn't see it
    # This preserves the original argv for ComfyUI while allowing our script to process -test
    original_argv = sys.argv.copy()
    if '-test' in sys.argv:
        sys.argv.remove('-test')
    if '-t' in sys.argv:
        sys.argv.remove('-t')
    
    # for testing start the test_server in the testing directory and run the main.py script with the -test or -t flag
    # Test mode for local testing, without the need for an external backend server
    if test: 
        print("Running in test mode...")
        global WEB_SERVER, password
        WEB_SERVER = "http://localhost:8001"
        password = "Password"
        poll_job()
    else:
        # Normal mode, connect to the configured backend server
        # make sure the config.toml file is set up correctly
        poll_job()


if __name__ == "__main__":
    main()