import torch
from compel import Compel
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from torch import Generator

device = "mps"
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
# dpm++
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config,
                                                             algorithm_type="dpmsolver++")

compel = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)

prompt = "a cat playing with a ball in the forest"
seed = 123

embeds = compel.build_conditioning_tensor(prompt)
fixed_seed_generator = Generator(device="cpu" if compel.device.type == "mps" else compel.device).manual_seed(seed)
image = pipeline(prompt_embeds=embeds,
                 num_inference_steps=7,
                 generator=fixed_seed_generator).images[0]

image.save('cat-ball-forest.png')
