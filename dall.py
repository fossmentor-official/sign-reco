from diffusers import StableDiffusionPipeline
import torch

# Load the pre-trained Stable Diffusion model from Hugging Face
pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipeline.to("cuda")  # Use GPU for faster inference if available

# Define your text prompt
prompt = "A futuristic cityscape with flying cars and neon lights at night"

# Generate the image
image = pipeline(prompt).images[0]

# Save the image to a file
image.save("generated_image.png")

print("Image saved as 'generated_image.png'")
