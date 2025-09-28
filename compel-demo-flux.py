from dataclasses import dataclass

import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from torch import Generator

from typing import List, Union
from diffusers import FluxPipeline
import torch

from compel import CompelForFlux

device = "mps"
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)

prompt = "Astronaut---- in a jungle++++, cold color palette, muted colors, detailed, 8k"
print(f"encoding prompt '{prompt}'...")
compel = CompelForFlux(pipe)
conditioning = compel(prompt)
print(conditioning.pooled_embeds.shape, conditioning.embeds.shape)


generator = torch.Generator().manual_seed(42)
images = pipe(prompt_embeds=conditioning.embeds, pooled_prompt_embeds=conditioning.pooled_embeds,
             num_inference_steps=4, width=512, height=512, generator=generator)

images[0][0].save('img0.jpg')
