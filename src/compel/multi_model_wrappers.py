from dataclasses import dataclass
from typing import Union, List

import torch
from diffusers import FluxPipeline, StableDiffusionXLPipeline

from compel import Compel, ReturnedEmbeddingsType


@dataclass(frozen=True)
class MultiEmbeddings:
    pooled_embeds: torch.Tensor
    embeds: torch.Tensor


class CompelForFlux:
    def __init__(self, pipe: FluxPipeline):
        self.compel_1 = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder, returned_embeddings_type=ReturnedEmbeddingsType.POOLED)
        self.compel_2 = Compel(tokenizer=pipe.tokenizer_2, text_encoder=pipe.text_encoder_2)

    def __call__(self, prompt: Union[str, List[str]]):
        pooled_embeds = self.compel_1(prompt)
        embeds = self.compel_2(prompt)
        return MultiEmbeddings(pooled_embeds=pooled_embeds, embeds=embeds)


class CompelForSDXL:
    def __init__(self, pipe: StableDiffusionXLPipeline):
        self.compel_1 = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder, returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED)
        self.compel_2 = Compel(tokenizer=pipe.tokenizer_2, text_encoder=pipe.text_encoder_2, returned_embeddings_type=ReturnedEmbeddingsType.POOLED)

    def __call__(self, prompt: Union[str, List[str]]):
        embeds = self.compel_1(prompt)
        pooled_embeds = self.compel_2(prompt)
        return MultiEmbeddings(pooled_embeds=pooled_embeds, embeds=embeds)
