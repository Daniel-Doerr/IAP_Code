import requests
import time
import os
import click

from dispatcher import WorkflowDispatcher

WEB_SERVER = "http://ssc-teddy.uni-heidelberg.de:3000/" 


def get_access_token(password: str) -> str:
    """Authenticate with the backend and get a JWT access token."""
    url = f"{WEB_SERVER}/token"
    data = {"password": password}
    response = requests.post(url, data=data)
    response.raise_for_status()
    return response.json()["access_token"]


def poll_job(workflow):
    # TODO: make the password configurable in a config file
    password = "Password"
    token = get_access_token(password)
    headers = {"Authorization": f"Bearer {token}"}

    start_time = time.time()
    dispatcher = WorkflowDispatcher()
    workflow_obj = dispatcher.create_workflow(workflow)
    elapsed_time = time.time() - start_time
    print(f"Workflow loaded in {elapsed_time:.2f} seconds")

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
            bone_broken = response.headers.get("bone_broken")

            if not job_id or not image_bytes:
                print("No valid job data received, skipping...")
                time.sleep(2)
                continue

            if animal_type is None:
                animal_type = "bear"

            print(f"Job received: {job_id}")
            print(f"Patient: {first_name} {last_name}, Animal: {animal_name}, AnimalType: {animal_type}")

            for i in range(1):
                start_time = time.time()
                img_buffer = workflow_obj.generate(workflow, image_bytes, animal_type, first_name, last_name, animal_name)
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



@click.command()
@click.option(
    '--workflow',
    # Add your workflow name into the list below
    type=click.Choice(['IP_Adapter_SDXL', 'FLUX_Kontext'], case_sensitive=False),
    prompt='Please choose a workflow',
    help='The workflow to use for image generation'
)
def main(workflow):
    """Run the image generation workflow with user-selected workflow."""
    click.echo(f"Starting workflow: {workflow}")
    poll_job(workflow)


if __name__ == "__main__":
    main()