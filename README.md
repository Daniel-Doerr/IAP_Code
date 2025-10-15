

# Teddy Bear Hospital

Welcome to the **Teddy Bear Hospital X-Ray** project of Heidelberg University! The tool developed here allows children to "x-ray" their beloved stuffed animals on a playful and interactive way, receiving unique, personalized "x-ray images." This provides young visitors with a tangible souvenir of their visit to the Teddy Bear Hospital.
## Table of Contents

* [About the Project](#about-the-project)
* [For System Administrators](#for-system-administrators)
* [For Enthusiasts and Developers](#for-enthusiasts-and-developers)
* [License](#license)

-----

## About the Project

The Teddy Bear Hospital project transforms images of stuffed animals into "pseudo-x-ray images" using AI diffusion technology. This offers an interactive and modern alternative to traditional, pre-printed skeleton drawings and allows for higher accuracy and personalization. The resulting images can be viewed directly on-site and accessed online, allowing children to take their "x-rayed" stuffed animal home as a unique souvenir.

This project was completed by a group of six Computer Science Bsc. Students at Heidelberg University during summer semester 2025. As such, this project will most likely not be maintained.

### Advantages over the previous solution:

  * **Significantly more accurate:** The generated images adapt to the individual stuffed animal.
  * **"Take-away" (online):** Images can be accessed and shared via a QR code.
  * **Recognition of the stuffed animal:** Each "x-ray image" is unique to the scanned stuffed animal.


## For System Administrators

### Installation Steps

The system installation includes setting up a Python environment, installing ComfyUI and Ollama, and loading the required models. 
Running the workflow can either happen within ComfyUI or using the provided `.py` file and ran as a script without interface. Both approached still require ComfyUI to be installed.

1. Create and start python enviorment
1. Install [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
2. Install [ComfyUI Manager](https://github.com/Comfy-Org/ComfyUI-Manager)
3. Install [Ollama](https://ollama.com/download)
4. Start ComfyUI
5. Head over to the ComfyUI Menu and look for Custom Node Manager search for and install the following Custom Node Packs:
    * [Comfyui\_Controlnet_Aux](https://github.com/Fannovel16/comfyui_controlnet_aux)
    * [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use)
    * [Comfyui-Ollama](https://github.com/stavsap/comfyui-ollama)
    * [ComfyUI-Allor](https://github.com/Nourepide/ComfyUI-Allor)
    * [ComfyUI-LogicUtils](https://github.com/aria1th/ComfyUI-LogicUtils)
    * [WAS-Node-Suite-Comfyui](https://github.com/WASasquatch/was-node-suite-comfyui)
    * [ComfyUI-Text-On-Image](https://github.com/S4MUEL-404/ComfyUI-Text-On-Image)
6. Download [Chroma](https://huggingface.co/lodestones/Chroma) and place it into the `ComfyUI/models/diffusion_models`. Download and install all necesarry CLIP encoders (you can find them on the Chroma Model Card Page on Huggingface)
7. Download [Mistral-Small 3.1:24B](https://ollama.com/library/mistral-small3.1) using `ollama pull mistral-small3.1:24b`
8. Downloard [Depth-Anything](https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vitl14.pth)
8. Import the Workflow `assets/Chroma_X-Ray_LLM.json` into ComfyUI and start generating images.


For the web app components, further specific steps are necessary, which will be added later.

### Recommended Hardware Configuration

For smooth operation, an Nvidia GPU with sufficient VRAM is recommended to meet the requirements of Chroma and Mistral-Small 3.1:22B simultaneously.
Mistral-Small 3.1:24B is quoted to require 24GB of dedicated VRAM or 32GB of unified VRAM. Chroma on the other hand, is quoted to need about 17GB of dedicated VRAM.

In other words, this workflow requires modern, powerful hardware to run. Of course, Mistral-Small 3.1:24B can be switched out for any other multimodal LLM, such as Google's Gemma3, Qwen or distilled DeepSeek models. The only important requirements for the LLMs is acccepting images as input and being able to accurately describe the subject, as that is relevant for the workflow.

For speed, models should fit into VRAM simultanously, but can, of course, be loaded one after another if VRAM is tight.

### Supported Operating Systems

The system is compatible with all operating systems where ComfyUI and Ollama run. This was teseted on Linux, so specific implimentation challenges for other systems may not be adressed.

### System Start, Stop, and Restart

Before you start the GPU Worker you first have to setup the ‘config.toml’ file. First enter the same password you used for the backend but this time not hashed. Secondly enter the URL where the backend server is setup. If you have not jet setup the backend server, you can go to [testing](#testing) where this setup is not required. 
To start the GPU Worker go to the ‘GPU_server’ folder and run the main with:
```shell 
python main.py
```
This will bind the GPU Worker to your shell, so if you close your shell the Worker will terminate also you can use ctrl + c to terminate the script. 
To run the Worker independently you can use 
```shell
nohup python main.py > log.txt 2>&1 &
```
this will start the Worker in the background and will save the output into a log.txt file. If you want to end the process you first have to find the process with 
```shell
ps aux | grep main.py
```
which will return the process ID (PID). Secondly use the 
```shell
kill [PID]
```
command to end the process. 
After receiving no job for 1 hour the program will automatically restart to free up all the VRAM. To manually restart the program, you will have to end it and start again. 

### Testing 

The included `testing/test_server.py` script, is a lightweight FastAPI test backend used for local testing. It issues image jobs (image bytes + metadata headers), cycles test workflows and sample metadata and accepts multipart uploads of generated results which it stores in `testing/generated_results`. You can add your own test images in the `testing/test_images` folder. The number of jobs per Workflow as well as all the available Workflows can be adjusted in the `testing/config.toml` file. 
Start the testing backend locally with `python test_server.py` and start the Worker in test mode by adding a -t or -test to the start command: 
```shell
python main.py -t 
```
For information about your CPU and GPU you can use the `testing/test_mem.py` script. The script will display information about CPU, GPU, RAM and VRAM usage and additional information about your hardware this may be helpful for debugging. 




## For Enthusiasts and Developers

This section is aimed at anyone who wants to contribute to the code or delve deeper into the technical details. We hope to guide possible future development with the work we did. See the text below as for why specific choices were made.



# GPU Server Architecture

The GPU server is a Python application designed for processing AI workflows, particularly in the field of image generation. The architecture is modular and consists of several core components.



### How it works (GPU)

This part outlines the architecture and operational flow of the GPU server.

-   **`main.py`**: Responsible for polling for jobs, managing resources, and error handling.
-   **`dispatcher.py`**: Responsible for handling the workflow objects.
-   **`functions.py`**: Includes the necessary and additional functions for the workflows.
-   **`"The_workflow.py"`**: A specific workflow script (e.g., `ChromaV44.py`) responsible for generating an image.


### Job Processing Flow

The process begins when `main.py` is executed. It follows a structured sequence of initialization, authentication, and a continuous polling loop to receive and process jobs.

1.  **Initialization and Authentication**:
    *   The server starts and loads its configuration from `config.toml`, which specifies the backend URL and authentication credentials.
    *   It sends a request to the backend's `/token` endpoint to obtain a JWT (JSON Web Token). This token is required for all subsequent authenticated API calls.
    *   If authentication fails, the server retries every 10 seconds until it succeeds.

2.  **Workflow Dispatcher Setup**:
    *   Once authenticated, `main.py` creates an instance of the `WorkflowDispatcher`.
    *   The dispatcher's primary role is to prepare the environment for ComfyUI. It adds the necessary ComfyUI and model directories to the system path.
    *   It then discovers and instantiates all available workflow classes (like `FLUX_Kontext`, `ChromaV44`, etc.) defined in the `workflow_scripts/` directory. It's important to note that at this stage, only the Python objects for the workflows are created; the heavy AI models are **not** loaded into memory yet.

3.  **Polling for Jobs**:
    *   The server enters an infinite loop, continuously polling the backend's `/job` endpoint every 2 seconds to check for new tasks.
    *   **If no job is available** (HTTP 204), the server simply waits and polls again. It includes logic for resource management during idle periods:
        *   After 30 minutes of inactivity, it calls `cleanup_gpu_memory()` to free up VRAM.
        *   After 1 hour of inactivity, it triggers a full restart (`restart_program()`) to ensure a clean state and prevent memory leaks.
    *   **If a job is available** (HTTP 200), the backend sends the job data.

4.  **Receiving and Preparing the Job**:
    *   The job data is received: the input image arrives in the response body, while metadata (like `job_id`, `workflow` name, and patient/animal details) is passed in the response headers.
    *   The server checks which workflow is requested (e.g., `"ChromaV44"`).

5.  **Workflow and Model Loading**:
    *   The server compares the requested workflow with the previously executed one.
    *   **If the workflow has changed**, it first calls `cleanup_gpu_memory()` to unload all models from the previous workflow, freeing up the GPU.
    *   Then, it calls the `start_load_once()` method on the new workflow object. This method loads all the necessary models (like UNET, VAE, LoRAs) into GPU memory. This "load-once" approach ensures that models are only loaded when the workflow type is first activated, saving significant time on subsequent jobs of the same type.

6.  **Image Generation**:
    *   With the correct models loaded, `main.py` calls the `generate()` method of the active workflow object, passing the input image and all metadata.
    *   The `generate()` method within the workflow script (e.g., `ChromaV44.py`) executes the ComfyUI graph step-by-step, processing the input image and text prompts to create the final artwork.

7.  **Returning the Result**:
    *   The `generate()` method returns the final image as an in-memory byte buffer.
    *   `main.py` receives this buffer and sends it back to the backend via a POST request to the `/job` endpoint, along with the original `job_id` to associate the result with the correct task.
    *   The server then prints the status of the upload and immediately polls for the next job, restarting the cycle.






### AI Pipeline and Generation Process

The AI pipeline is based on **ComfyUI**. The detailed generation process includes the following steps:

1.  **Background Removal and Image Manipulation:** The background of the input image is removed. Slight image manipulation places the depth map onto a grainy background.
    * **Why?** We found that removing the background elimantes the need for a clean, perfect background, if unavailable. Using a depth map, which is mostly white/light grey forces the diffusion model to 1) abide by contour limits introduced by the color difference (important for character conistency) and 2) to not generate a result which uses the texture of the original image (like fur). This is especially important, since as of the completion of the project, Chroma is not yet compatible with Controlnets, who would be responsible for outline/character consistency.
2.  **LLM Prompt Generation:** A Large Language Model (LLM) receives the input image, describes it, and adapts the description with "X-Ray" prompts to generate an optimized prompt for Chroma.
    * **Why?** Using an LLM allows for a more tailored prompt, identifying each animal and possible anomalies in limb length, ear size etc. Quirks of each individual plush can be considered and adjustet.
3.  **Image Generation:** The input image along with the positive, LLM-generated prompt are passed to the Chroma model, which generates the "x-ray image."
4.  **Overlay Addition:** Finally, an overlay is added to the generated image.
    * **Why?** This does not influence the result, but adds to the "realness" of the x-ray, at least in our application. This step can easily ommitted or adjustet to overlay different text/images.
5.  **Optimization:** A Turbo Low Step LoRA is used for process optimization.
    * **Why?** Using a Low Step LoRA simply allowed for faster, higher quality results. Without this, generation times would be unfeasible for our application. Since impact on quality is minimal, we urge you to keep this/something similiar.

    
As for model choice: The model we finally settled on to handle the bulk of the generation is Chroma. The reasons for this are quite simple:

1. Prompt following is vastly superior compared to FLUX base/SD
2. Uncensored nature allows for bone/skull/medically accurate generation
3. License.

Unfortunately other capable models released around the same time, such as Qwen Image Edit, FLUX Kontext, SeaDream 4.0 or WAN2.1/2.2, while capable, seem to be heavily censored, especially those from China.



### Adding Fractures to the X-Ray Images


After the x-ray image has been generated, the web interface allows users to manually draw in fractures to make the images even more realistic.

This functionality is implemented with a lightweight Python script. The workflow is as follows:

1. A mask is drawn onto the generated x-ray image at the position where the fracture should appear.
2. The script automatically samples two points from the surrounding image background.
3. Based on these samples, the mask is recolored so that it blends seamlessly into the x-ray background.
4. The result is a visually consistent "fracture" overlay that appears naturally integrated into the final x-ray image.


### How to create your own workflow 
Creating a workflow in ComfyUI is quite intuitive. You start by connecting different nodes that represent various steps in the image generation process: for example, loading a model, setting prompts, and defining image outputs. Each node can be customized to adjust parameters like resolution, seed, or sampling method. Once your nodes are connected, you can easily preview and generate images directly within the interface.
For more details and documentation, visit the official ComfyUI website: https://comfyui.org


#### How to implement your own workflow into the code
You need the ComfyUI workflow which you want to implement and you need a tool to export the workflow into a python script (https://github.com/pydn/ComfyUI-to-Python-Extension). 
1.	Export the workflow into a python script 
2.	Open the exported file and go to the `main` function where you copy all the code that is executed in the `with torch` block (without copying this line). 
```python 
with torch.inference_mode():
  # copy all code after this line (from the main function)
``` 

Theoretically this would work if you continue with steps 6 to 11, but this will reload the models which easily doubles your generating time. So, we will now move code into the `load_once` function to not reloaded models constantly (recommended).

3.	By using `Ctrl + f` search for `NODE_CLASS_MAPPINGS` and move all attributes that get initialized with this string into the `load_once()` function. (not a real time saver but needed for the next steps)
4.	Now search for `.safetensors` and move all lines of code into the `load_once()` function
5.	Now search for `model_name` or `modelloader` and move all lines of code into the `load_once()` function

Now all the models should only be loaded once and therefore save time. Optionally you can move all the code that is not directly or indirectly dependent on the input image (e.g. a watermark image) into the `load_once()` function, but if you are not certain which code to move you can skip this step. 

6.	Locate your input image, for that search for the first `loadimage_X` (X = a number, e.g. 17) and change the code to 

```python
loadimage_X = loadimage.load_image(image=tmp_path) # where X is the same number as before
```
If you are using constant images like a watermark or inputs for an IP-Adapter, make sure you input the correct path to set images. If you use text on the image, you can use the
```python
format_text_for_field(your_text, line_length, number_of_lines)
```
function to format the text correctly, the inputs are the string, the length of a line and the number of lines (functions are located in the `functions.py` file).

7.	Remove the `saveimge_Y` (Y = a number, e.g. 34) attribute at the end of the code because we don't need to save the image in the workflow, we will return it as a buffer
8.	Change in `converte_image(generatedImage)` `generatedImage` to the first attribute above this line of code (e.g. `converte_image(textonimage_142)`)
9.	Change the name of the class and the file to the name of your workflow, e.g. `FLUX_Kontext`

Now go to the `dispatcher.py` file and follow this last steps.

10.	Import the new workflow with 
```python
workflow_scripts.[workflow name] import [Class name]
```

11.	And add the `“[workflow name]”: [Class name],` into the workflow_class attribute 

If you want, you can change the default workflow in `main.py`. If you want to use the local test program to test your new workflow you have to add your workflows name to the `TEST_WORKFLOWS` list in `test_server.py`. 

### Contribution guidelines
Please contribute workflows, tests, and fixes - thank you! To add a new workflow, please follow the steps in the subsection [above](#how-to-implement-your-own-workflow-into-the-code). Before opening a pull request, test your implementation and ensure any new dependencies are documented in a `README` file. In your pull request description, explain what changed, why, and provide steps to reproduce. If the change affects runtime requirements (e.g., ComfyUI version or new model files), list them clearly. Small, focused pull requests that add a single workflow or fix a single issue are preferred, as this makes reviewing them faster and safer.


## License

This project uses a dual licensing approach:

### **Code License: MIT License**
All custom code in this repository is released under the MIT License:

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

### **AI Model Licenses**
Different AI models have their own licensing terms:

#### Diffusion Models
- **SDXL Base/Refiner**: CreativeML Open RAIL-M License
- **FLUX Models**: Apache 2.0 License
- **Chroma** : Apache-2.0 License
- **Custom LoRAs**: Individual licensing varies

#### Usage Rights for AI Models
-  **Non-commercial use**: Freely allowed for educational/research purposes
-  **Educational institutions**: Can use for events like Teddy Bear Hospital
-  **Commercial use**: Check individual model licenses
-  **Prohibited uses**: Harmful, illegal, or inappropriate content generation

### **Project-Specific Terms**

This software is specifically designed for:
- Educational and medical outreach programs
- Child-friendly hospital experiences  
- Non-profit community events
- Research and development in AI applications

### Third-Party Components

- **ComfyUI**: GPL-3.0 License
- **PyTorch**: BSD License  
- **CLIP**: MIT License
- **Custom Nodes**: Various licenses (see individual repositories)
- **DeepSeek / Janus-Pro** (used by FLUX_Kontext): model card on Hugging Face — https://huggingface.co/deepseek-ai/Janus-Pro-1B and repository: https://github.com/deepseek-ai/Janus (check the model and code licenses; code repo lists MIT for code and a model license file)
- **Ollama** (used as an LLM host/inference provider): https://ollama.com/ and documentation https://docs.ollama.com/ (check provider terms and any model licenses for LLMs you pull via Ollama)

Please verify the license and usage terms on each model card or provider page before deploying or redistributing any model artifacts.

### Disclaimer

This project generates artistic interpretations of X-ray images for entertainment and educational purposes only. Generated images are not medical diagnostics and should never be used for actual medical assessment or treatment decisions.


## Acknowledgments

- Dr. Dominic Kempf, who hosted this beginner internship
- The student contributors (team members) who designed and implemented the workflows, tests, and integration code.
- The ComfyUI community and the authors of custom node packs used in this project (see the Installation section for links).
- Model and dataset providers — in particular Lodestones/Chroma, DeepSeek/Janus, and Stability AI for the Stable Diffusion family of models.
- The many open-source projects and authors whose work we build upon (PyTorch, Hugging Face, and contributors to various node packs).
- Medical students and volunteers who organize such events for children.

If you or your project contributed to this repository and you'd like your name or organization added here, please open a pull request with the suggested acknowledgement text.