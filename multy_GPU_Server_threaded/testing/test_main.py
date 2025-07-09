import requests
import time
import os
import threading
import torch
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from dispatcher import WorkflowDispatcher

# Change this to your test server URL or production server
WEB_SERVER = "http://localhost:8001"  # For testing with local test server
# WEB_SERVER = "http://ssc-teddy.iwr.uni-heidelberg.de:8000"  # Production server 


def get_access_token(password: str) -> str:
    """Authenticate with the backend and get a JWT access token."""
    url = f"{WEB_SERVER}/token"
    data = {"password": password}
    response = requests.post(url, data=data)
    response.raise_for_status()
    return response.json()["access_token"]


def poll_job(dispatcher, gpu_id):
    """Poll for jobs and process them on a specific GPU."""
    
    # Set the GPU device for this thread
    if gpu_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        print(f"[GPU {gpu_id}] Set CUDA device to GPU {gpu_id}")
    elif gpu_id >= 0:
        print(f"[GPU {gpu_id}] Warning: CUDA not available, falling back to CPU")
        gpu_id = -1
    else:
        print(f"[CPU] Running on CPU")
    
    # TODO: make the password configurable in a config file
    password = "Password"
    token = get_access_token(password)
    headers = {"Authorization": f"Bearer {token}"}

    workflow_objects = dispatcher.create_workflow_obj(gpu_id)

    last_workflow = None

    while True:
        try:
            response = requests.get(f"{WEB_SERVER}/job", headers=headers)
            if response.status_code == 401:
                print(f"[GPU {gpu_id}] Unauthorized, refreshing token...")
                token = get_access_token(password)
                headers["Authorization"] = f"Bearer {token}"
                time.sleep(2)
                continue
            if response.status_code == 204:
                print(f"[GPU {gpu_id}] No job received...")
                time.sleep(2)
                continue

            # extract image data
            image_bytes = response.content

            # Header-Metadata
            job_id = response.headers.get("img_id")
            first_name = response.headers.get("first_name")
            last_name = response.headers.get("last_name")
            animal_name = response.headers.get("animal_name")
            animal_type = response.headers.get("animal_type")
            workflow = response.headers.get("workflow")

            if not job_id or not image_bytes or not workflow:
                print(f"[GPU {gpu_id}] No valid job data received, skipping...")
                time.sleep(2)
                continue

            # Select the appropriate workflow object
            if workflow not in workflow_objects:
                print(f"[GPU {gpu_id}] Unknown workflow: {workflow}. Available: {list(workflow_objects.keys())}")
                time.sleep(2)
                continue
            

            if animal_type == "other":
                animal_type = "stuffed animal"

            if workflow_objects[workflow] != last_workflow: 
                # Delete the last workflow object to free memory
                if last_workflow is not None:
                    print(f"[GPU {gpu_id}] Deleting previous workflow object to free memory")
                    del last_workflow
                    
                # Clear GPU memory
                clear_gpu_memory(gpu_id)
                print(f"[GPU {gpu_id}] Memory after cleanup: {get_gpu_memory_info(gpu_id)}")
                
                # Erstelle neues Workflow-Objekt für den aktuellen Workflow
                if workflow in workflow_objects:
                    del workflow_objects[workflow]  # Remove old instance if exists
                
                print(f"[GPU {gpu_id}] Creating new workflow object for '{workflow}'")
                try:
                    workflow_objects[workflow] = dispatcher.create_single_workflow_obj(gpu_id, workflow)
                    
                    # Lade die Modelle für den neuen Workflow
                    print(f"[GPU {gpu_id}] Loading models for workflow '{workflow}'")
                    workflow_objects[workflow].start_load_once()
                    
                    last_workflow = workflow_objects[workflow]
                    current_workflow_name = workflow
                    
                    print(f"[GPU {gpu_id}] Workflow switch completed")
                    print(f"[GPU {gpu_id}] Memory after loading: {get_gpu_memory_info(gpu_id)}")
                except Exception as workflow_error:
                    print(f"[GPU {gpu_id}] Error creating/loading workflow '{workflow}': {workflow_error}")
                    # Keep using the old workflow if possible
                    if last_workflow is not None:
                        print(f"[GPU {gpu_id}] Falling back to previous workflow")
                        workflow_objects[current_workflow_name] = last_workflow
                    else:
                        print(f"[GPU {gpu_id}] No fallback available, skipping job")
                        time.sleep(2)
                        continue
            else:
                # Workflow ist der gleiche, verwende das existierende Objekt
                last_workflow = workflow_objects[workflow]
            
            print(f"[GPU {gpu_id}] Job received: {job_id}; Using workflow: {workflow}")
            print(f"[GPU {gpu_id}] Patient: {first_name} {last_name}, Animal: {animal_name}, AnimalType: {animal_type}")

            for i in range(1):
                start_time = time.time()
                img_buffer = workflow_objects[workflow].generate(workflow, image_bytes, animal_type, first_name, last_name, animal_name)
                elapsed_time = time.time() - start_time
                print(f"[GPU {gpu_id}] Time taken to generate image: {elapsed_time:.2f} seconds")

                files = {
                    "result": ("result.png", img_buffer.getvalue(), "image/png"),
                }
                data = {
                    "image_id": job_id,
                }

                res = requests.post(f"{WEB_SERVER}/job", headers={"Authorization": f"Bearer {token}"}, files=files, data=data)
                print(f"[GPU {gpu_id}] Result sent:", res.status_code, res.text)

            
        except Exception as e:
            print(f"[GPU {gpu_id}] Error:", e)
            time.sleep(3)



def get_available_gpus():
    """Get the number of available CUDA GPUs."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        else:
            print("CUDA not available, falling back to CPU")
            return 0
    except ImportError:
        print("PyTorch not available, falling back to CPU")
        return 0


def print_gpu_info():
    """Print information about available GPUs."""
    try:
        import torch
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"\n=== Available GPUs ({num_gpus}) ===")
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            print()
        else:
            print("No CUDA GPUs available")
    except Exception as e:
        print(f"Error getting GPU info: {e}")


def get_gpu_selection():
    """Show available GPUs and let user select which ones to use."""
    num_gpus = get_available_gpus()
    
    if num_gpus <= 0:
        print("No GPUs available")
        return []
    
    print_gpu_info()
    print("Enter GPU numbers separated by spaces (e.g., '0 1 3') or press Enter for all GPUs:")
    
    while True:
        try:
            user_input = input("GPU selection: ").strip()
            
            # If empty, use all GPUs
            if not user_input:
                selected_gpus = list(range(num_gpus))
                print(f"Using all GPUs: {selected_gpus}")
                return selected_gpus
            
            # Parse space-separated numbers
            selected_gpus = [int(x) for x in user_input.split()]
            
            # Validate GPU IDs
            invalid_gpus = [gpu_id for gpu_id in selected_gpus if gpu_id < 0 or gpu_id >= num_gpus]
            if invalid_gpus:
                print(f"Invalid GPU IDs: {invalid_gpus}. Available: 0-{num_gpus-1}")
                continue
            
            # Remove duplicates and sort
            selected_gpus = sorted(list(set(selected_gpus)))
            print(f"Selected GPUs: {selected_gpus}")
            return selected_gpus
            
        except ValueError:
            print("Invalid input. Please enter numbers separated by spaces.")
            continue
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            exit(0)


def start_multi_gpu_workers(selected_gpus):
    """Start worker threads for selected GPUs."""
    if not selected_gpus:
        print("No GPUs selected, starting single CPU worker")
        dispatcher = WorkflowDispatcher()
        poll_job(dispatcher, -1)  # -1 indicates CPU
        return

    print(f"\nStarting {len(selected_gpus)} GPU workers...")
    
    threads = []
    for gpu_id in selected_gpus:
        # Create a separate dispatcher for each GPU to ensure isolation
        dispatcher = WorkflowDispatcher()
        
        thread = threading.Thread(
            target=poll_job, 
            args=(dispatcher, gpu_id),
            name=f"GPU-{gpu_id}-Worker"
        )
        thread.daemon = True  # Thread will terminate when main program exits
        thread.start()
        threads.append(thread)
        print(f"Started worker thread for GPU {gpu_id}")
        
        # Small delay to stagger thread startup
        time.sleep(0.5)
    
    print(f"All {len(selected_gpus)} workers started! Press Ctrl+C to stop...\n")
    
    # Print GPU info periodically
    def print_status():
        while True:
            try:
                time.sleep(30)  # Print status every 30 seconds
                print("\n=== GPU Status ===")
                for gpu_id in selected_gpus:
                    print(f"GPU {gpu_id}: {get_gpu_memory_info(gpu_id)}")
                print("==================\n")
            except Exception as e:
                print(f"Error printing status: {e}")
                break
    
    status_thread = threading.Thread(target=print_status, daemon=True)
    status_thread.start()
    
    # Wait for all threads to complete (they run indefinitely)
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("\nShutting down workers...")
        print("Clearing GPU memory...")
        for gpu_id in selected_gpus:
            clear_gpu_memory(gpu_id)


def clear_gpu_memory(gpu_id):
    """Clear GPU memory and run garbage collection."""
    import gc
    gc.collect()
    if gpu_id >= 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory_info(gpu_id):
    """Get GPU memory usage information."""
    if gpu_id >= 0 and torch.cuda.is_available():
        try:
            torch.cuda.set_device(gpu_id)
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
            cached = torch.cuda.memory_reserved(gpu_id) / 1024**3  # GB
            total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3  # GB
            return f"Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB, Total: {total:.2f}GB"
        except Exception as e:
            return f"Error getting memory info: {e}"
    return "CPU mode"


def main():
    selected_gpus = get_gpu_selection()
    start_multi_gpu_workers(selected_gpus)


if __name__ == "__main__":
    main()