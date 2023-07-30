from typing import Any

from compel import Compel, ReturnedEmbeddingsType
from diffusers import StableDiffusionXLPipeline
import torch

#device='mps'
device='cuda'
pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to(device)
pipeline.enable_vae_slicing()
pipeline.enable_sequential_cpu_offload()

compel = Compel(tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2], text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])

def run_single():
    prompt = "a cat playing with a ball++ in the forest"
    prompt_embeds, pooled = compel(prompt)
    print("single:", prompt_embeds.shape, pooled.shape)

    # generate image
    images = pipeline(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled, num_inference_steps=8, width=384, height=384).images
    return images

def run_and():
    pp = '("a cat", "a forest").and()'
    np = '("a dog, ugly", "sketch, scribble").and()'


    pp_embeds, pp_pooled = compel(pp)
    print("positive:", pp, pp_embeds.shape, pp_pooled.shape)
    np_embeds, np_pooled = compel(np)
    print("negative:", np, np_embeds.shape, np_pooled.shape)
    images = pipeline(prompt_embeds=pp_embeds, pooled_prompt_embeds=pp_pooled,
                      negative_prompt_embeds=np_embeds, negative_pooled_prompt_embeds=np_pooled,
                      num_inference_steps=24, width=768, height=768).images
    return images

images = run_and()
for index,image in enumerate(images):
    image.save(f'img{index}.jpg')
