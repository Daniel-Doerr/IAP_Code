# Teddy Bear Hospital and Beginner Internship Project

![Teddy Bear X-Ray](https://img.shields.io/badge/AI-Powered-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![GPU](https://img.shields.io/badge/GPU-Required-red) ![License](https://img.shields.io/badge/License-Open%20Source-brightgreen)

## Table of Contents
1. [Introduction](#introduction)
2. [Project Architecture](#project-architecture)
3. [Hardware Requirements](#hardware-requirements)
4. [Installation Guide](#installation-guide)
5. [Available Workflows](#available-workflows)
6. [How to Use](#how-to-use)
7. [Explanation of Image Generation](#explanation-of-image-generation)
8. [Adding Custom Workflows](#adding-custom-workflows)
9. [Testing](#testing)
10. [Contributions](#contributions)
11. [License](#license)

## Introduction 

Many children are afraid of going to the doctor or hospital. To help them feel more comfortable, some universities and hospitals organize a special event called a **Teddy Bear Hospital**. In this project, children bring their favorite stuffed animals — usually teddy bears — to a play hospital. There, medical students act as "teddy doctors" who examine, treat, and care for the toys. This playful experience helps children understand medical procedures in a fun and relaxed way, making future visits to real doctors less scary.

As part of my beginner internship for my computer science studies, I was part of a group that contributed to this project with a technical twist: We built a **fake X-ray machine** for the teddy bears. The idea was to make the experience even more exciting and realistic for the children.

Here is how it works:
- The teddy bear is placed into a white background.
- A picture of the teddy bear is taken.
- Then, using **Stable Diffusion**, an AI image generation model, the photo is transformed into a creative "X-ray image" of the bear.
- The result is shown on a screen and will be available for download through a personalized QR code.

It's just for fun and a great way to introduce children to both medicine and technology in a creative, friendly way. 

## Project Architecture

This project is split into two main components:

### 🖥️ **GPU Server Side** (This Repository)
- **Purpose**: Handles AI image generation using ComfyUI and Stable Diffusion
- **Location**: `/GPU_Server/` directory
- **Technology**: Python, PyTorch, ComfyUI, Stable Diffusion models
- **Responsibility**: Processes images and generates X-ray style outputs

### 🌐 **Frontend and Backend** 
- **Purpose**: Web interface, user management, and API endpoints
- **Location**: Parent directory (separate from this GPU server component)
- **Technology**: Web framework for user interaction and job management
- **Responsibility**: Handles user uploads, job queuing, and result delivery

### Communication Flow
```
Frontend → Backend API → GPU Server → AI Processing → Result → Backend → Frontend
```

## Hardware Requirements

⚠️ **GPU Required** - This is called "GPU-Server" for a reason!

### Minimum Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free space for models
- **OS**: Linux (recommended) or Windows with WSL2

### Recommended Setup
- **GPU**: NVIDIA RTX 4090, A100, or similar high-end GPU
- **VRAM**: 16GB+ for optimal performance
- **RAM**: 32GB+ system RAM
- **Storage**: SSD with 100GB+ free space

### Model-Specific VRAM Requirements
- **FLUX_Kontext**: ~12GB VRAM
- **IP_Adapter_SDXL**: ~8GB VRAM
- **Custom workflows**: Varies based on model complexity

> 💡 **Note**: Performance and generation time will vary significantly based on your GPU. The test server used an NVIDIA A100 80GB for reference benchmarks.

## Installation Guide

### Prerequisites
1. **NVIDIA GPU Drivers**: Latest drivers installed
2. **CUDA**: CUDA 11.8 or 12.1+ installed
3. **Python**: Python 3.8 or higher
4. **Git**: For cloning repositories

### Step-by-Step Installation

#### 1. Clone the Repository
```bash
git clone <repository-url>
cd IAP_Code/GPU_Server
```

#### 2. Install ComfyUI
```bash
# Clone ComfyUI in the parent directory
cd ..
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install ComfyUI dependencies
pip install -r requirements.txt
```

#### 3. Install Python Dependencies
```bash
cd ../GPU_Server
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install requests click pillow numpy opencv-python
```

#### 4. Install Custom Nodes (Required)
```bash
cd ../ComfyUI/custom_nodes

# Install essential custom nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git
git clone https://github.com/AIGODLIKE/AIGODLIKE-ComfyUI-Translation.git

# Install dependencies for custom nodes
pip install -r ComfyUI-Manager/requirements.txt
```

#### 5. Download Required Models

Create the necessary directories and download models:

```bash
cd ../models

# Download base models (examples - adjust URLs as needed)
mkdir -p checkpoints controlnet vae clip

# FLUX models
wget -O checkpoints/flux1-kontext-dev.safetensors [FLUX_MODEL_URL]

# SDXL models  
wget -O checkpoints/sd_xl_base_1.0.safetensors [SDXL_BASE_URL]
wget -O checkpoints/SDXL/sd_xl_refiner_1.0.safetensors [SDXL_REFINER_URL]

# ControlNet models
wget -O controlnet/FLUX.1/Shakker-Labs-ControlNet-Union-Pro/diffusion_pytorch_model.safetensors [CONTROLNET_FLUX_URL]
wget -O controlnet/SDXL/controlnet-union-sdxl-1.0/diffusion_pytorch_model_promax.safetensors [CONTROLNET_SDXL_URL]

# VAE models
wget -O vae/diffusion_pytorch_model.safetensors [VAE_URL]

# CLIP models
wget -O clip/clip_l.safetensors [CLIP_L_URL]
wget -O clip/t5/t5xxl_fp16.safetensors [T5_URL]
```

#### 6. Configure Model Paths
Create `extra_model_paths.yaml` in the ComfyUI directory:

```yaml
comfyui:
    checkpoints: models/checkpoints/
    controlnet: models/controlnet/
    vae: models/vae/
    clip: models/clip/
```

#### 7. Test Installation
```bash
cd ../GPU_Server
python test_main.py
```

### Troubleshooting Installation

**Common Issues:**

1. **CUDA out of memory**: Reduce batch size or use a smaller model
2. **Missing models**: Check model paths and download status
3. **Import errors**: Ensure all custom nodes are properly installed
4. **Permission errors**: Check file permissions and run with appropriate privileges

## Available Workflows

The system currently supports two main AI workflows:

### 🚀 FLUX_Kontext
- **Model Type**: FLUX-based diffusion model
- **VRAM Requirement**: ~12GB
- **Strengths**: High-quality image generation, good at understanding context
- **Use Case**: Best for detailed, anatomically accurate X-ray generation
- **Generation Time**: ~15-30 seconds on A100

### 🎨 IP_Adapter_SDXL  
- **Model Type**: Stable Diffusion XL with IP-Adapter
- **VRAM Requirement**: ~8GB
- **Strengths**: Style transfer, consistent character features
- **Use Case**: Good for stylized X-ray effects with consistent animal appearance
- **Generation Time**: ~20-40 seconds on A100

### Model Comparison

| Feature | FLUX_Kontext | IP_Adapter_SDXL |
|---------|--------------|------------------|
| Quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Speed | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| VRAM Usage | High | Medium |
| Anatomical Accuracy | Excellent | Good |
| Style Consistency | Good | Excellent |

## How to Use

### Running the GPU Server

1. **Start the server:**
```bash
cd GPU_Server
python main.py
```

2. **Select workflow:**
The system will prompt you to choose between available workflows:
- `FLUX_Kontext`
- `IP_Adapter_SDXL`

3. **Server operation:**
The server will:
- Connect to the backend API
- Poll for new image processing jobs
- Process images using the selected AI workflow
- Return generated X-ray images

### API Integration

The GPU server communicates with the backend via REST API:

**Authentication:**
```http
POST /token
Content-Type: application/x-www-form-urlencoded

password=your_password
```

**Job Polling:**
```http
GET /job
Authorization: Bearer <jwt_token>
```

**Result Submission:**
```http
POST /job
Authorization: Bearer <jwt_token>
Content-Type: multipart/form-data

image_id: job_id
result: generated_image.png
```

### Testing Locally

For development and testing:

```bash
cd GPU_Server/testing
python test_main.py
```

This will:
- Load a test image from the specified path
- Process it through the selected workflow
- Save the result to `output_images/`

## Explanation of Image Generation

### How Stable Diffusion Works

Stable Diffusion is a **latent diffusion model** that generates images through a process of controlled noise removal:

#### 1. **Text Encoding**
- Input text prompt is converted to numerical embeddings using CLIP
- These embeddings guide the generation process

#### 2. **Latent Space Processing**  
- Instead of working directly with pixels, the model operates in a compressed "latent space"
- A Variational Autoencoder (VAE) handles encoding/decoding between pixel and latent space
- This makes the process much more computationally efficient

#### 3. **Diffusion Process**
- Starts with random noise in latent space
- Iteratively removes noise over multiple steps (typically 20-50)
- Each step is guided by the text prompt and control inputs
- A U-Net neural network predicts what noise to remove at each step

#### 4. **Control Mechanisms**
Our system uses several control mechanisms for better results:

- **ControlNet**: Guides generation using depth maps from the original teddy bear image
- **IP-Adapter**: Ensures the generated X-ray maintains visual similarity to the input
- **Prompt Engineering**: Carefully crafted prompts emphasize medical/X-ray aesthetics

#### 5. **Post-Processing**
- Generated latent is decoded back to pixel space using VAE
- Text overlay adds patient information
- Final compositing with X-ray template/watermark

### X-Ray Generation Pipeline

```
Input Image → Background Removal → Depth Estimation → 
Text Analysis → Stable Diffusion → Post-Processing → Final X-Ray
```

1. **Preprocessing**: Remove background, resize, analyze content
2. **AI Description**: Use vision model (Janus) to describe the teddy bear
3. **Diffusion Generation**: Create X-ray using controlled diffusion process
4. **Postprocessing**: Add patient text, composite with medical template

### Why This Approach Works

- **Anatomical Plausibility**: AI models trained on diverse data can infer reasonable skeletal structures
- **Style Transfer**: Models can apply X-ray visual characteristics while preserving shape
- **Control**: Multiple guidance mechanisms ensure output matches input teddy bear
- **Safety**: System avoids scary/inappropriate content through careful prompt engineering

## Adding Custom Workflows

Want to add your own AI workflow? Follow these steps:

### 1. Create Workflow File
Create a new file in `workflow_scripts/YourWorkflowName.py`:

```python
import os
import sys
import io
from typing import Any, Union, Sequence, Mapping
import torch
from PIL import Image  
import random

# Add parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from functions import Functions

class YourWorkflowName:
    def __init__(self):
        self.functions = Functions()
        self.functions.add_comfyui_directory_to_sys_path()
        self.functions.add_extra_model_paths()
        try:
            from nodes import NODE_CLASS_MAPPINGS
            self.NODE_CLASS_MAPPINGS = NODE_CLASS_MAPPINGS
        except ImportError:
            print("Could not import NODE_CLASS_MAPPINGS from nodes.")
            self.NODE_CLASS_MAPPINGS = {}
        self.functions.import_custom_nodes()
        self.config = self.load_once()

    def load_once(self):
        """Load static resources once for efficiency."""
        # Load models, static prompts, etc.
        return {k: v for k, v in locals().items() if k != "self"}

    def generate(self, workflow_name: str, image_bytes: bytes, 
                animal_type: str, first_name: str, last_name: str, 
                animal_name: str) -> io.BytesIO:
        """Generate X-ray image from input."""
        # Your workflow implementation here
        result = self.functions.converte_image(generated_image)
        return result
```

### 2. Export ComfyUI Workflow
1. Create your workflow in ComfyUI interface
2. Export as Python script
3. Follow the integration steps in the template comments

### 3. Register Workflow
Add to `dispatcher.py`:

```python
from workflow_scripts.YourWorkflowName import YourWorkflowName

class WorkflowDispatcher:
    def __init__(self):
        self.workflow_class = {
            "FLUX_Kontext": FLUX_Kontext, 
            "IP_Adapter_SDXL": IP_Adapter_SDXL,
            "YourWorkflowName": YourWorkflowName  # Add here
        }
```

### 4. Update CLI Options
Add to `main.py` choices:

```python
@click.option(
    '--workflow',
    type=click.Choice(['IP_Adapter_SDXL', 'FLUX_Kontext', 'YourWorkflowName'], 
                      case_sensitive=False),
    prompt='Please choose a workflow',
    help='The workflow to use for image generation'
)
```

### Integration Guidelines

- **Static Resources**: Load models/prompts in `load_once()` for efficiency
- **Input Handling**: Use provided utility functions for image processing
- **Error Handling**: Include proper error handling and logging
- **Memory Management**: Use `torch.inference_mode()` for memory efficiency
- **Output Format**: Return `io.BytesIO` buffer with PNG image data

## Testing

### Unit Testing
```bash
cd GPU_Server
python -m pytest tests/ -v
```

### Integration Testing  
```bash
cd GPU_Server/testing
python test_main.py --workflow FLUX_Kontext
```

### Performance Testing
Monitor GPU memory usage and generation times:
```bash
nvidia-smi -l 1  # Monitor GPU usage
```

### Test Cases Covered
- ✅ Model loading and initialization
- ✅ Image preprocessing pipeline
- ✅ AI generation workflow
- ✅ Post-processing and text overlay
- ✅ Error handling and recovery
- ✅ Memory management

## Contributions

We welcome contributions to improve the Teddy Bear Hospital project! Here's how you can help:

### 🐛 Bug Reports
- Use GitHub Issues to report bugs
- Include system specifications and error logs
- Provide steps to reproduce the issue

### 🚀 Feature Requests
- Suggest new AI models or workflows
- Propose UI/UX improvements
- Request new customization options

### 💻 Code Contributions

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Follow coding standards**:
   - Use descriptive variable names
   - Add docstrings to functions
   - Include type hints where possible
   - Follow PEP 8 style guidelines

4. **Test your changes**:
   - Run existing tests
   - Add tests for new features
   - Test with different GPU configurations

5. **Submit pull request**:
   - Clear description of changes
   - Reference related issues
   - Include performance impact assessment

### 📖 Documentation
- Improve installation guides
- Add workflow tutorials
- Translate documentation
- Create video tutorials

### 🎨 Creative Contributions
- Design new X-ray templates
- Create example prompts
- Develop new AI workflows
- Improve visual aesthetics

### Development Guidelines

- **Code Quality**: All contributions should maintain high code quality
- **Performance**: Consider memory usage and generation speed
- **Compatibility**: Ensure compatibility across different GPU types
- **Safety**: Maintain child-friendly, appropriate content generation
- **Documentation**: Update documentation for any new features

## License

This project uses a dual licensing approach:

### 📄 **Code License: MIT License**
All custom code in this repository is released under the MIT License:

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

### 🤖 **AI Model Licenses**
Different AI models have their own licensing terms:

#### Stable Diffusion Models
- **SDXL Base/Refiner**: CreativeML Open RAIL-M License
- **FLUX Models**: Apache 2.0 License
- **Custom LoRAs**: Individual licensing varies

#### Usage Rights for AI Models
- ✅ **Non-commercial use**: Freely allowed for educational/research purposes
- ✅ **Educational institutions**: Can use for events like Teddy Bear Hospital
- ⚠️ **Commercial use**: Check individual model licenses
- ❌ **Prohibited uses**: Harmful, illegal, or inappropriate content generation

### 🏥 **Project-Specific Terms**

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

### Disclaimer

This project generates artistic interpretations of X-ray images for entertainment and educational purposes only. Generated images are not medical diagnostics and should never be used for actual medical assessment or treatment decisions.

---

## 🎉 Acknowledgments

Special thanks to:
- **Heidelberg University** for supporting this internship project
- **ComfyUI Community** for the excellent workflow management system
- **Stability AI** for open-sourcing Stable Diffusion models
- **Medical students and volunteers** who make Teddy Bear Hospitals possible
- **Children and families** who participate in these wonderful events

---

**Made with ❤️ for children's healthcare education** 