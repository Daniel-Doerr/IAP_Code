import requests
import time
import os
import click
import toml
import sys
import torch
import signal

from dispatcher import WorkflowDispatcher

# Global flag to handle clean shutdown via signals like Ctrl+C
shutdown_requested = False


def signal_handler(sig, frame):
    """Signal handler to set the global shutdown flag."""
    global shutdown_requested
    print("\n" + "="*60)
    print("SHUTDOWN REQUESTED - Stopping gracefully...")
    print("="*60)
    shutdown_requested = True


def load_config():
    """Load server configuration from the 'config.toml' file."""
    config_path = os.path.join(os.path.dirname(__file__), "config.toml")
    try:
        with open(config_path, 'r') as f:
            config_data = toml.load(f)
            # Return the 'server' section of the config, or an empty dict if not found
            return config_data.get("server", {})
    except FileNotFoundError:
        # Handle case where the config file does not exist
        print(f"Config file not found at {config_path}. Shutting down.")
        exit(1)
    except toml.TomlDecodeError:
        # Handle case where the config file has invalid TOML syntax
        print(f"Invalid TOML in config file {config_path}. Shutting down.")
        exit(1)


def restart_program():
    """Restarts the script to perform a hard reset, primarily for clearing GPU memory."""
    global shutdown_requested
    
    # Do not restart if a graceful shutdown was initiated by the user
    if shutdown_requested:
        print("Shutdown requested - NOT restarting program")
        exit(0)
    
    try:
        import subprocess
        print("=" * 60)
        print("RESTARTING PROGRAM TO CLEAR GPU MEMORY")
        print("=" * 60)
        
        # Get the path to the current script and its arguments
        script_path = sys.argv[0]
        script_args = sys.argv[1:]
        
        # Launch a new instance of the script with the same arguments
        subprocess.Popen([sys.executable, script_path] + script_args)
        
        # Terminate the current process
        print("New process started. Exiting current process...")
        exit(0)
        
    except Exception as e:
        print(f"Failed to restart program: {e}")
        print("Continuing with current process...")


def cleanup_gpu_memory():
    """Attempts to free up GPU memory by unloading models and clearing caches."""
    try:
        if torch.cuda.is_available():
            print("Starting GPU memory cleanup...")
            
            # Attempt to use ComfyUI's model management for a graceful unload
            try:
                # Dynamically import ComfyUI's model manager
                import comfy.model_management
                
                # Unload all models from the GPU
                print("Unloading all models from ComfyUI model management...")
                comfy.model_management.unload_all_models()

                # Free as much VRAM as possible using ComfyUI's utilities
                try:
                    device = comfy.model_management.get_torch_device()
                    comfy.model_management.free_memory(1e30, device)
                except TypeError:
                    # Fallback for older ComfyUI versions with a different function signature
                    comfy.model_management.free_memory()

                # Perform a soft cache empty, which is less disruptive than a hard empty
                comfy.model_management.soft_empty_cache()

                # If available, run the full model cleanup
                if hasattr(comfy.model_management, 'cleanup_models'):
                    comfy.model_management.cleanup_models()

                print("ComfyUI models unloaded")
                
            except ImportError:
                # This handles cases where the script is run without ComfyUI in the path
                print("ComfyUI model_management not available, skipping model unload")
            except Exception as e:
                print(f"Error unloading ComfyUI models: {e}")
            
            # Directly call PyTorch's cache clearing functions as a fallback/supplement
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Force Python's garbage collector to run
            import gc
            gc.collect()
            
            # Clear cache again after garbage collection for good measure
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            print("GPU memory cleanup completed")
            
    except Exception as e:
        print(f"Error during GPU cleanup: {e}")


def get_access_token(WEB_SERVER: str, password: str) -> str:
    """Authenticates with the backend server to obtain a JWT for subsequent requests."""
    url = f"{WEB_SERVER}/token"
    data = {"password": password}
    response = requests.post(url, data=data)
    response.raise_for_status()  # Raise an exception for HTTP errors (e.g., 401, 500)
    return response.json()["access_token"]


def poll_job(url: str, apassword: str):
    """The main loop that polls the server for jobs and processes them."""
    global shutdown_requested
    WEB_SERVER = url
    password = apassword

    # Register the signal handler for graceful shutdown on Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Continuously try to get an access token until successful or shutdown is requested
    token = None
    while token is None and not shutdown_requested:
        try:
            token = get_access_token(WEB_SERVER, password)
            print("Successfully obtained access token")
        except Exception as e:
            print(f"Failed to get access token: {e}")
            print("Retrying in 10 seconds...")
            time.sleep(10)
            
    if shutdown_requested:
        print("Shutdown requested during token acquisition")
        return

    # Prepare headers for authenticated API requests
    headers = {"Authorization": f"Bearer {token}"}

    # Initialize the WorkflowDispatcher to manage and load different workflows
    dispatcher = WorkflowDispatcher()
    # Create placeholder objects for all available workflows without loading models yet
    workflow_objects = dispatcher.create_workflow_obj()

    last_workflow = None # Keep track of the previously used workflow to manage memory
    no_job_count = 1 # Counter for consecutive polls with no job

    while not shutdown_requested:
        try:
            # Poll the server for a new job
            response = requests.get(f"{WEB_SERVER}/job", headers=headers)
            
            # If the token has expired or is invalid, refresh it
            if response.status_code == 401:
                print("Unauthorized, refreshing token...")
                token = get_access_token(password)
                headers["Authorization"] = f"Bearer {token}"
                time.sleep(2)
                continue
            
            # If no job is available, enter a sleep cycle
            if response.status_code == 204:
                print("No job received...")
                no_job_count += 1

                # After 1 hour of inactivity, trigger a full program restart to clear memory
                if no_job_count >= 960: # no jobs for one hour
                    print("Total sleep mode reached! Polling again in 1 minute.")

                    if last_workflow is not None:
                        print("Resetting program to free GPU memory.")
                        print("(Press Ctrl+C to terminate the program)")
                        restart_program()

                    for _ in range(2):  # Wait for 2*30 seconds, checking for shutdown
                        if shutdown_requested:
                            return
                        time.sleep(30)

                # After 30 minutes of inactivity, clean up GPU memory
                elif no_job_count >= 900: # Approx. 30 minutes
                    print("Going into sleep mode! Polling again in 30 seconds.")
                    if last_workflow is not None:
                        cleanup_gpu_memory()
                    for _ in range(3):  # Wait for 3*10 seconds, checking for shutdown
                        if shutdown_requested:
                            return
                        time.sleep(10)

                else:
                    # Default short sleep between polls
                    if shutdown_requested:
                        return
                    time.sleep(2)
                continue

            # Reset the inactivity counter since a job was received
            no_job_count = 1

            # Extract the image data from the response body
            image_bytes = response.content

            # Extract job metadata from response headers
            job_id = response.headers.get("img_id")
            first_name = response.headers.get("first_name")
            last_name = response.headers.get("last_name")
            animal_name = response.headers.get("animal_name")
            animal_type = response.headers.get("animal_type")
            workflow = response.headers.get("workflow")

            # Validate that essential job data is present
            if not job_id or not image_bytes:
                print("No valid job data received, skipping...")
                time.sleep(2)
                continue

            # Ensure the requested workflow is known; otherwise, default to a fallback
            if workflow not in workflow_objects:
                print(f"Unknown workflow: {workflow}. Available: {list(workflow_objects.keys())}")
                workflow = "FLUX_Kontext"  # Fallback to a default workflow
            
            # Standardize the animal type for better prompting consistency
            if animal_type == "other":
                animal_type = "stuffed animal"

            # If the requested workflow is different from the last one, manage memory
            if workflow != last_workflow: 
                if last_workflow is not None:
                    print(f"Switching from {last_workflow} to {workflow}, cleaning GPU memory...")
                    try:
                        # Clear the internal state of the old workflow object
                        workflow_objects[last_workflow].__dict__.clear()
                        del workflow_objects[last_workflow]
                        
                        # Run the full GPU cleanup process
                        cleanup_gpu_memory()
                        
                        # Re-create the old workflow object so it can be used again later
                        workflow_objects[last_workflow] = dispatcher.create_single_workflow_obj(last_workflow)
                        print(f"GPU memory cleaned and {last_workflow} workflow recreated")
                    except Exception as e:
                        print(f"Error during workflow cleanup: {e}")
                        # Ensure the object is recreated even if cleanup fails
                        workflow_objects[last_workflow] = dispatcher.create_single_workflow_obj(last_workflow)
                
                # Load the models for the new workflow
                print(f"Loading workflow: {workflow}")
                workflow_objects[workflow].start_load_once()

            # Update the last workflow tracker
            last_workflow = workflow
            
            print(f"Job received: {job_id}; Using workflow: {workflow}")
            print(f"Patient: {first_name} {last_name}, Animal: {animal_name}, AnimalType: {animal_type}")

            # Main generation loop (currently set to run once per job)
            for i in range(1):
                start_time = time.time()
                # Call the generate method of the selected workflow
                img_buffer = workflow_objects[workflow].generate(workflow, image_bytes, animal_type, first_name, last_name, animal_name)
                elapsed_time = time.time() - start_time
                print(f"Time taken to generate image: {elapsed_time:.2f} seconds")

                # Prepare the generated image and metadata for sending back to the server
                files = {
                    "result": ("result.png", img_buffer.getvalue(), "image/png"),
                }
                data = {
                    "image_id": job_id,
                }

                # Post the result back to the server
                res = requests.post(f"{WEB_SERVER}/job", headers={"Authorization": f"Bearer {token}"}, files=files, data=data)
                print("Result sent:", res.status_code, res.text)

        # Handle Ctrl+C gracefully
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received - shutting down gracefully...")
            shutdown_requested = True
            break
        # Catch all other exceptions to prevent the poller from crashing
        except Exception as e:
            print("Error:", e)
            if shutdown_requested:
                break
            # Attempt to clean up GPU memory on error before continuing
            cleanup_gpu_memory()
            time.sleep(3)
    
    # Final message on graceful shutdown
    print("Program terminated gracefully")



@click.command()
@click.option('-test', '-t', is_flag=True, help='Run in test mode using local test server settings.')
def main(test):
    """Main entry point for the script, controlled by command-line flags."""
    # Preserve original command-line arguments for potential restarts
    # Remove the --test flag so it's not passed to other processes (like ComfyUI)
    if '-test' in sys.argv:
        sys.argv.remove('-test')
    if '-t' in sys.argv:
        sys.argv.remove('-t')
    
    # Test mode uses a hardcoded local server configuration for development
    if test: 
        print("Running in test mode...")
        WEB_SERVER = "http://localhost:8001"
        password = "Password"
        poll_job(WEB_SERVER, password)
    else:
        # Normal mode connects to the production backend server defined in config.toml
        # Load configuration from the file
        config = load_config()
        WEB_SERVER = config.get("WEB_SERVER") # URL of the backend server
        password = config.get("password") # Password for authentication
        
        # Start the main job polling loop
        poll_job(WEB_SERVER, password)


if __name__ == "__main__":
    main()