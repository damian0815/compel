import os
from typing import Any

from compel import Compel, ReturnedEmbeddingsType
from diffusers import StableDiffusionXLPipeline
import torch

#device='mps'
device='cuda'
pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to(device)
pipeline.enable_vae_slicing()
#pipeline.enable_sequential_cpu_offload()

compel = Compel(tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2], text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])

def run_single():
    prompt = "a cat playing with a ball++ in the forest"
    prompt_embeds, pooled = compel(prompt)
    print("single:", prompt_embeds.shape, pooled.shape)

    # generate image
    images = pipeline(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled, num_inference_steps=24, width=768, height=512).images
    return images

def run_and():
    pp = '("a cat in the forest", "a fantasy forest").and()'
    np = 'ugly, sketch, scribble'

    pp_embeds, pp_pooled = compel(pp)
    print("positive:", pp, pp_embeds.shape, pp_pooled.shape)
    np_embeds, np_pooled = compel(np)
    print("negative:", np, np_embeds.shape, np_pooled.shape)
    [pp_embeds, np_embeds] = compel.pad_conditioning_tensors_to_same_length([pp_embeds, np_embeds])
    images = pipeline(prompt_embeds=pp_embeds, pooled_prompt_embeds=pp_pooled,
                      negative_prompt_embeds=np_embeds, negative_pooled_prompt_embeds=np_pooled,
                      num_inference_steps=24, width=768, height=512).images
    return images

def run_and_multi():
    pp = '("a cat in the forest", "a fantasy forest").and()'
    np = 'ugly, sketch, scribble'

    embeds, pooled = compel([pp,np])
    print(f"embeds: '{pp}', '{np}', embeds shape {embeds.shape}, pooled shape {pooled.shape}")
    #np_embeds, np_pooled = compel(np)
    #print("negative:", np, np_embeds.shape, np_pooled.shape)
    #[pp_embeds, np_embeds] = compel.pad_conditioning_tensors_to_same_length([pp_embeds, np_embeds])
    images = pipeline(prompt_embeds=embeds[0:1], pooled_prompt_embeds=pooled[0:1],
                      negative_prompt_embeds=embeds[1:2], negative_pooled_prompt_embeds=pooled[1:2],
                      num_inference_steps=24, width=768, height=512).images
    return images


images = run_and_multi()
#images = run_single()
for index,image in enumerate(images):
    image.save(f'img{index}.jpg')
print(f"images saved to {os.getcwd()}")
