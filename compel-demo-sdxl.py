import torch

from compel import Compel, ReturnedEmbeddingsType, CompelForSDXL
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionXLPipeline

device='mps'
pipeline: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16
).to(device)

prompt = "a cat playing with a ball++ in the forest"
negative_prompt = "badly drawn"


# new method using CompelForSDXL
compel = CompelForSDXL(pipeline)
conditioning = compel(prompt, negative_prompt=negative_prompt)

generator = torch.Generator().manual_seed(42)
image = pipeline(prompt_embeds=conditioning.embeds, pooled_prompt_embeds=conditioning.pooled_embeds,
                 negative_prompt_embeds=conditioning.negative_embeds, negative_pooled_prompt_embeds=conditioning.negative_pooled_embeds,
             num_inference_steps=25, width=1024, height=1024, generator=generator).images[0]
image.save('sdxl_new_method.jpg')


# old method using manual settings wrangling
compel = Compel(tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2] ,
                text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True])
prompt_embeds, pooled_prompt_embeds = compel([prompt, negative_prompt])

generator = torch.Generator().manual_seed(42)
image = pipeline(prompt_embeds=prompt_embeds[0:1], pooled_prompt_embeds=pooled_prompt_embeds[0:1],
                 negative_prompt_embeds=prompt_embeds[1:2], negative_pooled_prompt_embeds=pooled_prompt_embeds[1:2],
             num_inference_steps=25, width=1024, height=1024, generator=generator).images[0]
image.save('sdxl_old_method.jpg')


