"""
Test server to simulate the web backend for testing main.py
This server provides jobs with images and receives generated results.
"""

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import Response
import uvicorn
import os
import time
import threading
from datetime import datetime, timedelta
import jwt
import glob
from PIL import Image
import io

app = FastAPI(title="Test Server for GPU Processing")
security = HTTPBearer()

# Configuration
SECRET_KEY = "test_secret_key_for_jwt"
PASSWORD = "Password"
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24

# Test data
TEST_WORKFLOWS = ["FLUX_Kontext", "IP_Adapter_SDXL"]
current_workflow_index = 0
job_counter = 0
jobs_per_workflow = 3  # Switch workflow after this many jobs

def find_test_images():
    """Find test images in the 'test_images' subdirectory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(current_dir, "test_images")
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    test_images = []

    # Create test_images directory if it doesn't exist
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        print(f"Created directory: {images_dir}")

    # Search for images in the test_images directory
    for ext in image_extensions:
        test_images.extend(glob.glob(os.path.join(images_dir, ext)))

    # If no images found, create a simple test image
    if not test_images:
        print("No test images found, creating a simple test image...")
        try:
            dummy_image = Image.new('RGB', (512, 512), color=(100, 150, 200))
            dummy_path = os.path.join(images_dir, "test_image.png")
            dummy_image.save(dummy_path)
            test_images = [dummy_path]
            print(f"Created test image: {dummy_path}")
        except Exception as e:
            print(f"Could not create test image: {e}")

    print(f"Found {len(test_images)} test images in {images_dir}")
    for img in test_images:
        print(f"  - {os.path.basename(img)}")

    return test_images

TEST_IMAGES = find_test_images()
current_image_index = 0

# Test animal data
TEST_ANIMALS = [
    {"first_name": "Max", "last_name": "Mustermann", "animal_name": "Teddy", "animal_type": "bear"},
    {"first_name": "Anna", "last_name": "Schmidt", "animal_name": "Fluffy", "animal_type": "bear"},
    {"first_name": "Tom", "last_name": "Johnson", "animal_name": "Buddy", "animal_type": "bear"},
    {"first_name": "Lisa", "last_name": "Weber", "animal_name": "Luna", "animal_type": "bear"},
]

def create_access_token(data: dict):
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

@app.post("/token")
async def login(password: str = Form(...)):
    """Authenticate and return JWT token"""
    if password != PASSWORD:
        raise HTTPException(status_code=401, detail="Incorrect password")
    
    access_token = create_access_token(data={"sub": "test_user"})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/job")
async def get_job(user=Depends(verify_token)):
    """Get a job with image and metadata"""
    global job_counter, current_workflow_index, current_image_index
    
    # Check if we have any test images
    if not TEST_IMAGES:
        print("No test images available - returning 204")
        return Response(status_code=204)  # No Content
    
    
    # Switch workflow after certain number of jobs
    if job_counter > 0 and job_counter % jobs_per_workflow == 0:
        current_workflow_index = (current_workflow_index + 1) % len(TEST_WORKFLOWS)
        print(f"Switching to workflow: {TEST_WORKFLOWS[current_workflow_index]}")
    
    # Get current test data
    workflow = TEST_WORKFLOWS[current_workflow_index]
    animal_data = TEST_ANIMALS[job_counter % len(TEST_ANIMALS)]
    image_path = TEST_IMAGES[current_image_index % len(TEST_IMAGES)]
    
    print(f"Reading image: {image_path}")
    
    # Read image
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        job_counter += 1  # Increment counter even when there's an error
        return Response(status_code=204)
    
    # Prepare job metadata
    job_id = f"job_{job_counter:04d}"
    
    # Create response with image data and headers
    headers = {
        "img_id": job_id,
        "first_name": animal_data["first_name"],
        "last_name": animal_data["last_name"],
        "animal_name": animal_data["animal_name"],
        "animal_type": animal_data["animal_type"],
        "workflow": workflow
    }
    
    print(f"Sending job {job_id}: {workflow} - {animal_data['animal_name']} ({animal_data['animal_type']})")
    
    job_counter += 1
    current_image_index += 1
    
    return Response(
        content=image_bytes,
        media_type="application/octet-stream",
        headers=headers
    )

@app.post("/job")
async def submit_result(
    image_id: str = Form(...),
    result: UploadFile = File(...),
    user=Depends(verify_token)
):
    """Receive generated image result"""
    try:
        # Create output directory
        output_dir = os.path.join(os.path.dirname(__file__), "generated_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # Read the uploaded image
        image_data = await result.read()
        
        # Save the result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{image_id}_{timestamp}.png"
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, "wb") as f:
            f.write(image_data)
        
        print(f"Received and saved result for {image_id}: {output_path}")
        
        # Validate it's a proper image
        try:
            with Image.open(output_path) as img:
                print(f"Result image size: {img.size}, mode: {img.mode}")
        except Exception as e:
            print(f"Warning: Could not validate image {output_path}: {e}")
        
        return {"status": "success", "message": f"Result saved as {filename}"}
        
    except Exception as e:
        print(f"Error saving result for {image_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving result: {e}")

@app.get("/")
async def root():
    """Root endpoint with server info"""
    return {
        "message": "Test Server for GPU Processing",
        "available_workflows": TEST_WORKFLOWS,
        "current_workflow": TEST_WORKFLOWS[current_workflow_index],
        "jobs_processed": job_counter,
        "test_images": len(TEST_IMAGES)
    }

@app.get("/status")
async def status():
    """Get current server status"""
    return {
        "jobs_processed": job_counter,
        "current_workflow": TEST_WORKFLOWS[current_workflow_index],
        "next_workflow_in": jobs_per_workflow - (job_counter % jobs_per_workflow),
        "available_workflows": TEST_WORKFLOWS,
        "test_images_available": len(TEST_IMAGES)
    }

def run_server():
    """Run the test server"""
    print("=" * 60)
    print("Starting Test Server for GPU Processing")
    print("=" * 60)
    print(f"Available workflows: {TEST_WORKFLOWS}")
    print(f"Test images found: {len(TEST_IMAGES)}")
    print(f"Jobs per workflow: {jobs_per_workflow}")
    print(f"Server URL: http://localhost:8001")
    print(f"Password: {PASSWORD}")
    print("=" * 60)
    
    if not TEST_IMAGES:
        print("WARNING: No test images found!")
        print("Please add image files to the 'test_images' subdirectory.")
        print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")

if __name__ == "__main__":
    run_server()
