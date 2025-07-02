import requests
import time
import os

from multi_config import load_config_based_on_workflow
from multi_workflow import load_workflow

WEB_SERVER = "http://localhost:8001" 



def poll_job():
    # Choose the workflow to use 
    # TODO: make this in cooperation with the frontend 
    workflow = "AIprompt_FLUX_Kontext"
    
    workflow_obj = load_workflow(workflow)

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
            img_buffer = workflow_obj.generate(workflow, image_bytes, animal_type, first_name, last_name, animal_name)

            # Save the output image as a PNG in a directory
            output_dir = "output_images"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{animal_name}_{job_id}.png")

            # img_buffer is expected to be bytes (PNG format)
            with open(output_path, "wb") as out_f:
                out_f.write(img_buffer.getvalue())



if __name__ == "__main__":
    poll_job()