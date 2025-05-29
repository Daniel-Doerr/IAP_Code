import os
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import time

DUMMY_IMAGE_PATH = "/mnt/data/tbkh2025_dk/ComfyUI/Input_animals/Cat_done/cat_original.png" 

class JobServerHandler(BaseHTTPRequestHandler):
    job_given = False
    result_received = False

    # give the GPU server a dummy job 
    def do_GET(self):
        if self.path == "/job" and not JobServerHandler.job_given:
            JobServerHandler.job_given = True
            # Check if the dummy image exists
            if not os.path.exists(DUMMY_IMAGE_PATH):
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"Missing image")
                return
            with open(DUMMY_IMAGE_PATH, "rb") as f:
                img_bytes = f.read()
            # Send the job as a response
            self.send_response(200)
            self.send_header("Content-type", "image/png")
            self.send_header("img_id", "testid0")
            self.send_header("first_name", "John")
            self.send_header("last_name", "Doe")
            self.send_header("animal_name", "Fido")
            self.send_header("animal_type", "Cat")
            self.end_headers()
            self.wfile.write(img_bytes)
            print("Job given to client.")
        else:
            self.send_response(204)
            self.end_headers()
            print("Job already given.")


    # handle the result from the GPU server
    def do_POST(self):
        if self.path == "/job":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            output_dir = "/mnt/data/tbkh2025_dk/ComfyUI/output_test"
            os.makedirs(output_dir, exist_ok=True)
            # create a unique filename
            base_name = "result_image"
            ext = ".png"
            output_path = os.path.join(output_dir, base_name + ext)
            idx = 1
            while os.path.exists(output_path):
                output_path = os.path.join(output_dir, f"{base_name}_{idx}{ext}")
                idx += 1
            # save the result image
            with open(output_path, "wb") as f:
                f.write(post_data)
            JobServerHandler.result_received = True
            # send a response back to the GPU server
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
            print("Result received from client.")


def run():
    # Start the HTTP server
    server = HTTPServer(("0.0.0.0", 8000), JobServerHandler)
    print("Serving on port 8000...")

    # Start a thread to reset job flags periodically
    def reset_job_flags():
        # give the GPU server every 10 seconds a job 
        while True:
            if JobServerHandler.result_received:
                time.sleep(10)
                JobServerHandler.job_given = False
                JobServerHandler.result_received = False
                print("Job flags reset, ready for new job.")
            time.sleep(2)
    
    # Start the reset thread
    print("Starting reset thread...")
    reset_thread = threading.Thread(target=reset_job_flags, daemon=True)
    reset_thread.start()

    try:
        while True:
            server.handle_request()
    except KeyboardInterrupt:
        print("Shutting down.")


if __name__ == "__main__":
    run()