# LaMa Inpainting API (Object Remover)

A FastAPI-based image inpainting service powered by Simple LaMa. This application removes unwanted objects or regions from images using a mask and reconstructs the missing content with deep learning-based inpainting.

## Features

* Image inpainting using LaMa
* FastAPI REST API
* Supports PNG and JPG images
* Base64 encoded output
* Automatic mask processing
* Health check endpoint

## Project Structure

```text
.
├── app.py
├── model.py
├── requirements.txt
└── README.md
```

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/lama-inpainting-api.git
cd lama-inpainting-api
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Required Packages

```text
fastapi
uvicorn
numpy
pillow
imageio
simple-lama-inpainting
python-multipart
```

## Run the API

Start the FastAPI server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

## API Endpoints

### Health Check

**GET /**

Response:

```json
{
  "status": "success",
  "message": "LaMa Inpainting API is running. Use POST /inpaint to process images."
}
```

### Image Inpainting

**POST /inpaint**

Parameters:

| Parameter | Type | Required | Description |
| --------- | ---- | -------- | ----------- |
| image     | File | Yes      | Input image |
| mask      | File | No       | Mask image  |

#### Example Request

```bash
curl -X POST \
  "http://localhost:8000/inpaint" \
  -F "image=@image.png" \
  -F "mask=@mask.png"
```

---

## ☁️ HuggingFace API Usage (Production)

### 📤 Example Request

import requests

url = "https://umscorleonis-object-remover-space.hf.space/inpaint"

files = {
    "image": open("image.png", "rb"),
    "mask": open("mask.png", "rb")
}

response = requests.post(url, files=files)
print(response.json())

---
#### Example Response

```json
{
  "status": "success",
  "message": "✅ Processing completed!",
  "result_base64": "iVBORw0KGgoAAAANSUhEUg..."
}
```

## How It Works

1. Upload an image.
2. Upload a mask indicating the region to remove.
3. The mask is converted into a binary mask.
4. Simple LaMa reconstructs the masked region.
5. The resulting image is returned as a Base64 encoded PNG.

## Technology Stack

* FastAPI
* Simple LaMa Inpainting
* NumPy
* Pillow
* ImageIO
* Python

## Author

Sukma Wati

## License

This project is provided for educational and research purposes.
