import torch
from compel import Compel
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from torch import Generator

device = "mps"
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
# dpm++
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config,
                                                             algorithm_type="dpmsolver++")

prompts = ["a cat playing with a ball++ in the forest", "a cat playing with a ball in the forest"]

compel = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
prompt_embeds = torch.cat([compel.build_conditioning_tensor(prompt) for prompt in prompts])
images = pipeline(prompt_embeds=prompt_embeds, num_inference_steps=10, width=256, height=256).images
print(images)

images[0].save('/tmp/img0.jpg')
images[1].save('/tmp/img1.jpg')
