import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from diffusers import StableDiffusionPipeline
from PIL import Image
import tempfile
from urllib.parse import unquote
import torch
import base64

app = FastAPI()

# Load the Stable Diffusion pipeline with fallback to CPU
def load_pipeline():
    model_id = "runwayml/stable-diffusion-v1-5"
    try:
        if torch.cuda.is_available():
            return StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
        else:
            raise Exception("CUDA not available")
    except Exception as e:
        print(f"Error loading model on GPU: {e}. Falling back to CPU.")
        return StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to("cpu")

pipe = load_pipeline()

# Function to generate an image from text
def generate_image(text_prompt: str):
    with torch.no_grad():
        images = pipe(text_prompt, num_inference_steps=50).images
    return images

@app.get("/generate-images/{text_prompt}", summary="Generate Four Images from Text", description="Generate four images based on the provided text prompt included in the URL.")
async def generate_images(text_prompt: str):
    try:
        text_prompt = unquote(text_prompt)
        images = generate_image(text_prompt)
        
        base64_images = []
        for i, image in enumerate(images[:4]):  # Ensure only four images are included
            # Convert each image to base64 string
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                image.save(temp_file, format="PNG")
                temp_file_path = temp_file.name
                with open(temp_file_path, "rb") as image_file:
                    base64_images.append(base64.b64encode(image_file.read()).decode("utf-8"))

        return JSONResponse(content={"images": base64_images})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", summary="Welcome", description="Welcome message with basic API information.")
def home():
    return {"message": "Welcome to the Text to Image Generation API."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
