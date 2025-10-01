import torch
from numpy.ma.core import negative

from compel import Compel, ReturnedEmbeddingsType, CompelForSD
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from torch import Generator

device = "mps"
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
# dpm++
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config,
                                                             algorithm_type="dpmsolver++")

prompts = ["a cat playing with a ball++ in the forest", "a dog wearing a hat++"]

compel = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
prompt_embeds = compel(prompts)
images = pipeline(prompt_embeds=prompt_embeds, num_inference_steps=25, width=512, height=512).images
print(images)

images[0].save('sd-img0.jpg')
images[1].save('sd-img1.jpg')

# new method using CompelForSD
compel = CompelForSD(pipeline)
conditioning = compel(prompts, negative_prompt="badly drawn")
generator = torch.Generator().manual_seed(42)
images = pipeline(prompt_embeds=conditioning.embeds, negative_prompt_embeds=conditioning.negative_embeds,
                 num_inference_steps=25, width=512, height=512, generator=generator).images
images[0].save('sd-img0_w_neg.jpg')
images[1].save('sd-img1_w_neg.jpg')
