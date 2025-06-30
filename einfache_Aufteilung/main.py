import requests
import time


from NEW_FLUX_Kontext import generateImage

WEB_SERVER = "http://localhost:8001" 


def get_access_token(password: str) -> str:
    """Authenticate with the backend and get a JWT access token."""
    url = f"{WEB_SERVER}/token"
    data = {"password": password}
    response = requests.post(url, data=data)
    response.raise_for_status()
    return response.json()["access_token"]


def poll_job():
    password = "Password"
    token = get_access_token(password)
    headers = {"Authorization": f"Bearer {token}"}

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

            for i in range(3):
                img_buffer = generateImage(image_bytes, animal_type, first_name, last_name, animal_name)

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



if __name__ == "__main__":
    poll_job()