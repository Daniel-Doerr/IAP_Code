import requests
import time
import os
import threading
import torch

from dispatcher import WorkflowDispatcher

# Change this to your server URL 
WEB_SERVER = "http://ssc-teddy.iwr.uni-heidelberg.de:8000"  # Production server 


def get_access_token(password: str) -> str:
    """Authenticate with the backend and get a JWT access token."""
    url = f"{WEB_SERVER}/token"
    data = {"password": password}
    response = requests.post(url, data=data)
    response.raise_for_status()
    return response.json()["access_token"]


def poll_job(dispatcher, gpu_id):
    """Poll for jobs and process them on a specific GPU."""
    
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
                    del last_workflow
                # Re-create the workflow object and update the dictionary
                workflow_objects[workflow] = dispatcher.create_single_workflow_obj(gpu_id, workflow)
                workflow_objects[workflow].start_load_once()


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
    
    # Create one dispatcher for all threads
    dispatcher = WorkflowDispatcher()
    
    threads = []
    for gpu_id in selected_gpus:
        thread = threading.Thread(target=poll_job, args=(dispatcher, gpu_id))
        thread.daemon = True  # Thread will terminate when main program exits
        thread.start()
        threads.append(thread)
        print(f"Started worker thread for GPU {gpu_id}")
    
    print(f"All {len(selected_gpus)} workers started! Press Ctrl+C to stop...\n")
    
    # Wait for all threads to complete (they run indefinitely)
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("\nShutting down workers...")


def main():
    selected_gpus = get_gpu_selection()
    start_multi_gpu_workers(selected_gpus)


if __name__ == "__main__":
    main()