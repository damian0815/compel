from dataclasses import dataclass

import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from torch import Generator

from typing import List, Union
from diffusers import FluxPipeline
import torch

@dataclass(frozen=True)
class CompelEmbeddings:
    pooled_embeds: torch.Tensor|None
    embeds: torch.Tensor|None

class CompelForFlux:
    def __init__(self, pipe: FluxPipeline):
        self.compel_1 = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder, requires_pooled=True)
        self.compel_2 = Compel(tokenizer=pipe.tokenizer_2, text_encoder=pipe.text_encoder_2)

    def __call__(self, prompt: Union[str, List[str]]):
        _, pooled_embeds = self.compel_1(prompt)
        embeds = self.compel_2(prompt)
        return CompelEmbeddings(pooled_embeds=pooled_embeds, embeds=embeds)

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
