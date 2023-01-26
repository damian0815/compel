import torch
from compel import Compel
from diffusers import StableDiffusionPipeline
from torch import Generator

pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").half().to('mps')

compel = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)

positive_prompts = ["a man picking apricots- from a tree",
                    "a man picking apricots from a tree",
                    "a man picking apricots+ from a tree",
                    "a man picking apricots++ from a tree"]
negative_prompts = ["ladder--"] * len(positive_prompts)

positive_embeddings = torch.cat([compel.build_conditioning_tensor(p).unsqueeze(0) for p in positive_prompts])
negative_embeddings = torch.cat([compel.build_conditioning_tensor(p).unsqueeze(0) for p in negative_prompts])

fixed_seed_generator = Generator(seed=123)
images = pipeline(prompt_embeds=positive_embeddings,
                  negative_prompt_embeds=negative_embeddings,
                  generator=fixed_seed_generator).images


