import torch
from diffusers import Lumina2Pipeline

from compel import CompelForLumina2

with torch.no_grad():
    device = "cuda"
    pipe = Lumina2Pipeline.from_pretrained(
        "Alpha-VLLM/Lumina-Next-SFT-diffusers", torch_dtype=torch.bfloat16
    ).to(device)
    compel = CompelForLumina2(pipe)

    prompt = "a cat playing with a ball in the forest"
    prompt_weighted = "a cat playing with a ball++ in the forest"

    print(f"generating baseline image for '{prompt}' without compel...")
    generator = torch.Generator().manual_seed(42)
    images = pipe(prompt=prompt, num_inference_steps=50, width=1024, height=1024, generator=generator)
    print("generated, saving...")
    images.images[0].save("lumina2_baseline.jpg")

    print(f"encoding plain prompt '{prompt}' with compel...")
    conditioning = compel(prompt)
    print("generating with plain compel embeddings...")
    generator = torch.Generator().manual_seed(42)
    images = pipe(
        prompt_embeds=conditioning.embeds,
        prompt_attention_mask=conditioning.attention_mask,
        num_inference_steps=50,
        width=1024,
        height=1024,
        generator=generator,
    )
    print("generated, saving...")
    images.images[0].save("lumina2_compel_plain.jpg")

    print(f"encoding weighted prompt '{prompt_weighted}' with compel...")
    conditioning_weighted = compel(prompt_weighted)
    print("generating with weighted compel embeddings...")
    generator = torch.Generator().manual_seed(42)
    images = pipe(
        prompt_embeds=conditioning_weighted.embeds,
        prompt_attention_mask=conditioning_weighted.attention_mask,
        num_inference_steps=50,
        width=1024,
        height=1024,
        generator=generator,
    )
    print("generated, saving...")
    images.images[0].save("lumina2_compel_weighted.jpg")
