
from ultralytics import YOLO
import cv2
import requests
import numpy as np
import os

from bs4 import BeautifulSoup
from urllib.parse import urljoin
from io import BytesIO

# Load fracture model
model = YOLO("models/fracture_best.pt")

headers = {
    "User-Agent": "Mozilla/5.0"
}


def load_image(url):
    response = requests.get(
        url,
        headers=headers,
        timeout=20
    )
    response.raise_for_status()

    if not response.headers.get(
        "Content-Type", ""
    ).startswith("image/"):
        raise ValueError("URL did not return an image.")

    image_array = np.frombuffer(
        response.content, np.uint8
    )

    image = cv2.imdecode(
        image_array, cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError("Invalid image format.")

    return image


def get_images_from_page(page_url):

    response = requests.get(
        page_url,
        headers=headers,
        timeout=20
    )
    response.raise_for_status()

    soup = BeautifulSoup(
        response.text, "html.parser"
    )

    image_urls = []

    for tag in soup.find_all("img"):

        src = (
            tag.get("src")
            or tag.get("data-src")
            or tag.get("data-original")
        )

        if src:
            image_url = urljoin(page_url, src)

            if image_url not in image_urls:
                image_urls.append(image_url)

    return image_urls


def detect_fracture(image):

    results = model.predict(
        source=image,
        conf=0.25,
        imgsz=640
    )

    for result in results:

        if len(result.boxes) == 0:
            print("No fracture detected by the model.")

        else:
            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                print(
                    "Class:",
                    model.names[class_id]
                )

                print(
                    "Confidence:",
                    f"{confidence:.2%}"
                )

        output = result.plot()

        cv2.imshow(
            "YOLO11n Fracture Detection",
            output
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Main program

user_input = input(
    "Enter image path or URL: "
).strip().strip('"')

try:

    if user_input.startswith(("http://", "https://")):

        try:
            # First try as a direct image URL
            image = load_image(user_input)

        except ValueError:
            # Otherwise treat it as a webpage
            image_urls = get_images_from_page(user_input)

            if not image_urls:
                raise ValueError(
                    "No images found on this webpage."
                )

            print("\nImages found:")

            for i, url in enumerate(image_urls):
                print(f"{i + 1}. {url}")

            choice = int(
                input("\nSelect image number: ")
            )

            if choice < 1 or choice > len(image_urls):
                raise ValueError("Invalid image selection.")

            image = load_image(image_urls[choice - 1])

    else:

        if not os.path.isfile(user_input):
            raise FileNotFoundError("Image path not found.")

        image = cv2.imread(user_input)

        if image is None:
            raise ValueError("Invalid image.")

    print("\nImage loaded successfully!")
    print("Running fracture detection...\n")

    detect_fracture(image)

except Exception as error:

    print("\nError:", error)
