# Compel
A text prompt weighting and blending library for transformers-type text embedding systems, by [@damian0815](https://github.com/damian0815).

With a flexible and intuitive syntax, you can re-weight different parts of a prompt string and thus re-weight the different parts of the embeddning tensor produced from the string.

Tested and developed against Hugging Face's `StableDiffusionPipeline` but it should work with any diffusers-based system that uses an `Tokenizer` and a `Text Encoder` of some kind.  

Adapted from the [InvokeAI](https://github.com/invoke-ai) prompting code (also by [@damian0815](https://github.com/damian0815)). For now, the syntax is fully documented [here](Reference.md).

Note that cross-attention control `.swap()` is currently ignored by Compel, but you can use it by calling `build_conditioning_tensor_for_prompt_object()` yourself, and implementing cross-attention control in your diffusion loop.

### Installation

`pip install compel`

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
# or: conditioning = compel([prompt])

# generate image
images = pipeline(prompt_embeds=conditioning, num_inference_steps=20).images
images[0].save("image.jpg")
```

For batched input, use the __call__ interface to compel:

```python
import torch

from diffusers import StableDiffusionPipeline
from compel import Compel

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
compel = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)

prompts = ["a cat playing with a ball++ in the forest", "a dog playing with a ball in the forest"]
prompt_embeds = compel(prompts)
images = pipeline(prompt_embeds=prompt_embeds).images

images[0].save("image0.jpg")
images[1].save("image1.jpg")
```

## Changelog

#### 1.1.3 - enable fetching the penultimate CLIP hidden layer (aka "clip skip")

To use, pass `use_penultimate_clip_layer=True` when initializing your `Compel` instance. Note that there's no need to pass this flag for SD2.0/SD2.1 because diffusers already throws away the last hidden layer when loading the SD2.0+ text encoder.

#### 1.1.2 - fix for #21 (crash when parsing long prompts with truncation enabled if there is weighted fragments beyond the truncation boundary)

#### 1.1.1 - fix for #22 (issues parsing `.` characters inside parentheses)

#### 1.1.0 - support for parsing `withLora`/`useLora` on `parse_prompt_string()`.

* `Compel.parse_prompt_string()` now returns a `Conjunction`
* any appearances of `withLora(name[, weight])` or `useLora(name[, weight])` anywhere in the prompt string will be parsed to `LoraWeight` instances, and returned on the outermost `Conjunction` returned by `parse_prompt_string()`.

#### 1.0.5 - fix incorrect parsing when passing invalid (auto1111) syntax that has a float

also fix test case for default swap parameters

#### 1.0.4 - fix embeddings for empty swap target (eg `cat.swap("")`) when truncation is disabled 

#### 1.0.3 - better defaults for .swap (https://github.com/damian0815/compel/issues/8)

#### 1.0.2 - fix padding for non-truncated batched embeddings (https://github.com/damian0815/compel/issues/9)

#### 1.0.1 - fix for InvokeAI's `--free_gpu_mem` option

### 1.0.0 - new downweighting algorithm 

Downweighting now works by applying an attention mask to remove the downweighted tokens, rather than literally removing them from the sequence. This behaviour is the default, but the old behaviour can be re-enabled by passing `downweight_mode=DownweightMode.REMOVE` on init of the `Compel` instance.

Formerly, downweighting a token worked by both multiplying the weighting of the token's embedding, and doing an inverse-weighted blend with a copy of the token sequence that had the downweighted tokens removed. The intuition is that as weight approaches zero, the tokens being downweighted should be actually removed from the sequence. However, removing the tokens resulted in the positioning of all downstream tokens becoming messed up. The blend ended up blending a lot more than just the tokens in question. 

As of v1.0.0, taking advice from @keturn and @bonlime (https://github.com/damian0815/compel/issues/7) the procedure is by default different. Downweighting still involves a blend but what is blended is a version of the token sequence with the downweighted tokens masked out, rather than removed. This correctly preserves positioning embeddings of the other tokens. 

Also a bugfix: fix black images on weight 0 (https://github.com/invoke-ai/InvokeAI/issues/2832)

### 0.1.10 - add support for prompts longer than the model's max token length. 

To enable, initialize `Compel` with `truncate_long_prompts=False` (default is True). Prompts that are longer than the model's `max_token_length` will be chunked and padded out to an integer multiple of `max_token_length`. 

Note that even if you don't use a negative prompt, you'll need to build a conditioning tensor for a negative prompt of at least `""`, and use `compel.pad_conditioning_tensors_to_same_length()`, otherwise the you'll get an error about mismatched conditioning tensor lengths:

```python
compel = Compel(..., truncate_long_prompts=False)
prompt = "a cat playing with a ball++ in the forest, amazing, exquisite, stunning, masterpiece, skilled, powerful, incredible, amazing, trending on gregstation, greg, greggy, greggs greggson, greggy mcgregface, ..." # very long prompt
conditioning = compel.build_conditioning_tensor(prompt)
negative_prompt = "" # it's necessary to create an empty prompt - it can also be very long, if you want
negative_conditioning = compel.build_conditioning_tensor(negative_prompt)
[conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])
```

#### 0.1.9 - broken

#### 0.1.8 - downgrade Python min version to 3.7

#### 0.1.7 - InvokeAI compatibility

