from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from model import process_image
import numpy as np
from PIL import Image
import io
import base64

app = FastAPI(title="LaMa Inpainting API")

@app.post("/inpaint")
async def inpaint(image: UploadFile = File(...), mask: UploadFile = File(None)):
    try:
        # Read input files
        image_data = await image.read()
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGBA")
        image_np = np.array(image_pil)

        if mask:
            mask_data = await mask.read()
            mask_pil = Image.open(io.BytesIO(mask_data)).convert("RGBA")
            mask_np = np.array(mask_pil)
        else:
            mask_np = None

        # Process
        result_np = process_image(image_np, mask_np)
        result_img = Image.fromarray(result_np)

        # Encode as Base64
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        result_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return JSONResponse({
            "status": "success",
            "message": "✅ Processing completed!",
            "result_base64": result_base64
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# Health check / default route
@app.get("/")
async def root():
    return JSONResponse({
        "status": "success",
        "message": "LaMa Inpainting API is running. Use POST /inpaint to process images."
    })
