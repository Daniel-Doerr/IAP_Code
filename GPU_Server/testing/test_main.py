import requests
import time
import os
import click
import sys

# Add the parent directory to sys.path to handle relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from dispatcher import WorkflowDispatcher


def poll_job(workflow):

    start_time = time.time()
    dispatcher = WorkflowDispatcher()
    workflow_obj = dispatcher.create_workflow(workflow)
    elapsed_time = time.time() - start_time
    print(f"Workflow loaded in {elapsed_time:.2f} seconds")

    index = 0
    while True:
        ## Pfad auf Richtigkeit prüfen
        path = "/mnt/data/tbkh2025_dk/ComfyUI/input/IMG-20250422-WA0003.jpg"
        # extract image data
        with open(path, "rb") as f:
            image_bytes = f.read()

        job_id = index
        index += 1
        first_name = "hallo"
        last_name = "neuer Workflow"
        animal_name = "Bär"
        animal_type = "bear"

        print(f"Job received: {job_id}")
        print(f"Patient: {first_name} {last_name}, Animal: {animal_name}, AnimalType: {animal_type}")

        for i in range(1):
            
            start_time = time.time()
            img_buffer = workflow_obj.generate(workflow, image_bytes, animal_type, first_name, last_name, animal_name)
            elapsed_time = time.time() - start_time
            print(f"Time taken to generate image: {elapsed_time:.2f} seconds")

            # Save the output image as a PNG in a directory
            output_dir = "output_images"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{animal_name}_{job_id}.png")

            # img_buffer is expected to be bytes (PNG format)
            with open(output_path, "wb") as out_f:
                out_f.write(img_buffer.getvalue())



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