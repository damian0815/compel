# compel
A prompting enhancement library for transformers-type text embedding systems, by @damian0815. 

Adapted from the InvokeAI prompting code (also by @damian0815). For now, the syntax is fully documented [here](https://invoke-ai.github.io/InvokeAI/features/PROMPTS/).

### Demo

see [compel-demo.ipynb](compel-demo.ipynb)

<a target="_blank" href="https://colab.research.google.com/github/damian0815/compel/blob/main/compel-demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


### Quickstart

with Hugging Face diffusers >=0.12:

```python
from diffusers import StableDiffusionPipeline
from compel import Compel

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
compel = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)

# upweight "ball"
prompt = "a cat playing with a ball++ in the forest"
conditioning = compel.build_conditioning_tensor(prompt)

# generate image
image = pipeline(prompt_embeds=conditioning, num_inference_steps=20).images[0]
```


