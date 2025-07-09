import requests
import time
import os
import click

from dispatcher import WorkflowDispatcher

# WEB_SERVER = "http://ssc-teddy.iwr.uni-heidelberg.de:8000" 
WEB_SERVER = "http://localhost:8001"  # For local testing, change to your server address


def get_access_token(password: str) -> str:
    """Authenticate with the backend and get a JWT access token."""
    url = f"{WEB_SERVER}/token"
    data = {"password": password}
    response = requests.post(url, data=data)
    response.raise_for_status()
    return response.json()["access_token"]


def poll_job():
    # TODO: make the password configurable in a config file
    password = "Password"
    token = get_access_token(password)
    headers = {"Authorization": f"Bearer {token}"}

    dispatcher = WorkflowDispatcher()

    workflow_objects = dispatcher.create_workflow_obj()

    last_workflow = None

    while True:
        try:
            response = requests.get(f"{WEB_SERVER}/job", headers=headers)
            if response.status_code == 401:
                print("Unauthorized, refreshing token...")
                token = get_access_token(password)
                headers["Authorization"] = f"Bearer {token}"
                time.sleep(2)
                continue
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

            if not job_id or not image_bytes or not workflow:
                print("No valid job data received, skipping...")
                time.sleep(2)
                continue

            # Select the appropriate workflow object
            if workflow not in workflow_objects:
                print(f"Unknown workflow: {workflow}. Available: {list(workflow_objects.keys())}")
                time.sleep(2)
                continue
            

            if animal_type == "other":
                animal_type = "stuffed animal"

            if workflow_objects[workflow] != last_workflow: 
                # Delete the last workflow object to free memory
                if last_workflow is not None:
                    del last_workflow
                # Re-create the workflow object and update the dictionary
                workflow_objects[workflow] = dispatcher.create_single_workflow_obj(workflow)
                workflow_objects[workflow].start_load_once()


            last_workflow = workflow_objects[workflow]
            
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

                res = requests.post(f"{WEB_SERVER}/job", headers={"Authorization": f"Bearer {token}"}, files=files, data=data)
                print("Result sent:", res.status_code, res.text)

            
        except Exception as e:
            print("Error:", e)
            time.sleep(3)


def main():
    poll_job()


if __name__ == "__main__":
    main()