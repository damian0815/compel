
# Overview

Compel is a single-purpose, lightweight library that converts prompt strings to embedding tensors with some handy weighting and blending features.

To instantiate, pass a `tokenizer` and `text_encoder` (Transformer), typically from a StableDiffusion pipeline:

```python
from diffusers import StableDiffusionPipeline
from compel import Compel

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
compel = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
```

To generate conditioning tensors, call `build_conditioning_tensor(prompt)` or use the `__call__` interface:

```
prompt = "a cat playing in the forest"
conditioning = compel(prompt)
# or: conditioning = compel.build_conditioning_tensor(prompt)
```

Typically with a Stable Diffusion workflow you'll also be using a negative prompt:

```python
negative_prompt = "ugly, distorted, deformed"
negative_conditioning = compel(prompt)
```

If you've disabled truncation or if your prompt contains [`.and()` syntax](syntax.md#conjunction), you'll need to ensure the conditioning and negative conditioning tensors are the same length:

```python
[conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])
```

Finally, pass both of the tensors to your `StableDiffusionPipeline`, like this:

```python
images = pipeline(prompt_embeds=conditioning, negative_prompt_embeds=negative_conditioning).images
images[0].save("image.jpg")
```

# Details

Compel first [parses the prompt](syntax.md) into a [`Conjunction`](prompt_parser.md#compel.prompt_parser.Conjunction), then uses the structure of the `Conjunction` to construct a conditioning tensor for the prompt.

### Weighting

[Weighting](syntax.md#weighting) is applied using a combination of masked scaling of the parts of the conditioning tensor that correspond to the weighted terms in the prompt and, for weights < 1, a blend (lerp) with a version of the conditioning tensor where the negatively weighted terms have been removed. 

#### Upweighting 
Depending on model and CFG you can weight up to around 1.5 or 1.6 before things start to get weird. 

#### Downweighting 
For downweighting you can go all the way down to 0, at which point the downweighting terms completely disappear. The weight space is non-linear, and there seems to be an inflection point around 0.5 where the SD process suddenly changes how much attention it pays to the downweighted terms. 


### Blend

[Blends](syntax.md#blend) are implemented as a mathematical lerp of the conditioning tensors. Sometimes they work the way you'd expect, more often they don't. Experiment and have fun.

### Conjunction 
[Conjunctions](syntax.md#conjunction) work by concatenating prompts together. You'll need to use `pad_conditioning_tensors_to_same_length` if you want to combine conjunctions and negative prompts:

```python
[conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])
```

# Reference

Syntax features: [Syntax](syntax.md)

The `Compel` class: [Compel](compel.md)

The prompt parser and structured prompt classes (`Conjunction`, `FlattenedPrompt`, `Blend` etc): [PromptParser](prompt_parser.md)
