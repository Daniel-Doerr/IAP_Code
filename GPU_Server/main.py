import requests
import time
import os
import click
import toml

from dispatcher import WorkflowDispatcher


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


def get_access_token(password: str) -> str:
    """Authenticate with the backend and get a JWT access token."""
    url = f"{WEB_SERVER}/token"
    data = {"password": password}
    response = requests.post(url, data=data)
    response.raise_for_status()
    return response.json()["access_token"]


def poll_job():
    global password

    # Try to get access token 
    token = None
    while token is None:
        try:
            token = get_access_token(password)
            print("Successfully obtained access token")
        except Exception as e:
            print(f"Failed to get access token: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)

    # Set up headers for authentication
    headers = {"Authorization": f"Bearer {token}"}

    # Create a WorkflowDispatcher instance to manage workflows
    dispatcher = WorkflowDispatcher()
    # Create workflow objects for each workflow type without loading the models yet
    workflow_objects = dispatcher.create_workflow_obj()

    last_workflow = None # Variable to keep track of the last used workflow object

    while True:
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

            # Check if all required data is present
            if not job_id or not image_bytes:
                print("No valid job data received, skipping...")
                time.sleep(2)
                continue

            # Check if the workflow is known
            if workflow not in workflow_objects:
                print(f"Unknown workflow: {workflow}. Available: {list(workflow_objects.keys())}")
                workflow = "ChromaV44"  # Default to a known workflow
                # time.sleep(2)
                # continue
            
            # Handle animal type "other" for prompting
            if animal_type == "other":
                animal_type = "stuffed animal"

            # Select the appropriate workflow object
            if workflow != last_workflow: 
                # Delete the last workflow object to free memory
                if last_workflow is not None:
                    del workflow_objects[last_workflow]
                    # Re-create the last_workflow object and update the dictionary
                    workflow_objects[last_workflow] = dispatcher.create_single_workflow_obj(last_workflow)
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
        except Exception as e:
            print("Error:", e)
            time.sleep(3)



@click.command()
@click.option('-test', '-t', is_flag=True, help='Run in test mode')
def main(test):
    """Main function to start the job polling."""
    # for testing start the test_server in the testing directory and run the main.py script with the -test or -t flag
    # Test mode for local testing, without the need for a backend server
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