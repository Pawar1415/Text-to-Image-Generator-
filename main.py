import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import tempfile
import time

# Set Streamlit page configuration
st.set_page_config(page_title="Text to Image Generator", layout="centered")
st.title("🖼️ Text to Image Generator (CPU Only)")

# Load Stable Diffusion model on CPU
@st.cache_resource
def load_pipeline():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        use_safetensors=True
    ).to("cpu")
    return pipe

pipe = load_pipeline()

# Prompt input field
prompt = st.text_input("Enter a text prompt:")

# Generate image only when prompt is entered and button is clicked
if prompt and st.button("Generate Image"):
    with st.spinner("Generating image... please wait (20–40 seconds)"):
        start_time = time.time()
        with torch.no_grad():
            result = pipe(prompt, num_inference_steps=25)
            image = result.images[0]
        end_time = time.time()

        st.success(f"Image generated in {end_time - start_time:.2f} seconds")
        st.image(image, caption="Generated Image", use_column_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            image.save(tmp_file.name)
            st.download_button(
                label="Download Image",
                data=open(tmp_file.name, "rb"),
                file_name="generated_image.png",
                mime="image/png"
            )
