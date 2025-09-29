from dataclasses import dataclass

import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from torch import Generator

from typing import List, Union
from diffusers import FluxPipeline
import torch

from compel import CompelForFlux

with torch.no_grad():
    device = "cpu"
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to(device)
    compel = CompelForFlux(pipe)

    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    prompt_weighted = "Astronaut---- in a jungle++++, cold color palette, muted colors, detailed, 8k"

    print(f"generating baseline images for '{prompt}' without compel...")
    generator = torch.Generator().manual_seed(42)
    images = pipe(prompt=prompt, num_inference_steps=4, width=512, height=512, generator=generator)
    print("generated, saving...")
    images[0][0].save('flux_baseline.jpg')


    print(f"encoding plain prompt '{prompt}' with compel...")
    conditioning = compel(prompt)
    print("generating with plain compel embeddings...")
    generator = torch.Generator().manual_seed(42)
    images = pipe(prompt_embeds=conditioning.embeds, pooled_prompt_embeds=conditioning.pooled_embeds,
                  num_inference_steps=4, width=512, height=512, generator=generator)
    print("generated, saving...")
    images[0][0].save('flux_compel_plain.jpg')


    print(f"encoding weighted prompt '{prompt_weighted}' with compel...")
    conditioning_weighted = compel(prompt_weighted)
    print("generating with weighted compel embeddings...")
    generator = torch.Generator().manual_seed(42)
    images = pipe(prompt_embeds=conditioning_weighted.embeds, pooled_prompt_embeds=conditioning_weighted.pooled_embeds,
                 num_inference_steps=4, width=512, height=512, generator=generator)
    print("generated, saving...")
    images[0][0].save('flux_compel_weighted.jpg')

