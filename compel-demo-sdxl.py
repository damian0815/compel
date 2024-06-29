import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from torch import Generator

device='cuda'
pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                             variant="fp16",
                                             use_safetensors=True,
                                             torch_dtype=torch.float16).to(device)

compel = Compel(tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2] ,
                text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True])


device = "mps"
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
# dpm++
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config,
                                                             algorithm_type="dpmsolver++")

prompts = ["a cat playing with a ball++ in the forest", "a cat playing with a ball in the forest"]

compel = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
prompt = "a cat playing with a ball++ in the forest"
conditioning, pooled = compel(prompt)
print(conditioning.shape, pooled.shape)

prompt_embeds = compel(prompts)
images = pipeline(prompt_embeds=prompt_embeds, num_inference_steps=10, width=256, height=256).images
print(images)

images[0].save('/tmp/img0.jpg')
images[1].save('/tmp/img1.jpg')
