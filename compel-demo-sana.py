import torch
from diffusers import SanaPipeline

from compel import CompelForSana

with torch.no_grad():
    device = "cuda"
    pipe = SanaPipeline.from_pretrained(
        "Efficient-Large-Model/Sana_600M_1024px_diffusers",
        torch_dtype=torch.bfloat16
    ).to(device)
    compel = CompelForSana(pipe)

    prompt = "a cat playing with a ball in the forest"
    prompt_weighted = "a cat playing with a ball++ in the forest"

    #print(f"generating baseline image for '{prompt}' without compel...")
    #generator = torch.Generator().manual_seed(42)
    #images = pipe(
    #    prompt=prompt,
    #    num_inference_steps=20,
    #    guidance_scale=4.5,
    #    width=1024,
    #    height=1024,
    #    generator=generator,
    #)
    #print("generated, saving...")
    #images.images[0].save("sana_baseline.jpg")

    print(f"encoding plain prompt '{prompt}' with compel...")
    conditioning = compel(prompt)
    print("generating with plain compel embeddings...")
    generator = torch.Generator().manual_seed(42)
    images = pipe(
        prompt_embeds=conditioning.embeds,
        prompt_attention_mask=conditioning.attention_mask,
        num_inference_steps=20,
        guidance_scale=4.5,
        width=1024,
        height=1024,
        generator=generator,
    )
    print("generated, saving...")
    images.images[0].save("sana_compel_plain.jpg")

    print(f"encoding weighted prompt '{prompt_weighted}' with compel...")
    conditioning_weighted = compel(prompt_weighted)
    print("generating with weighted compel embeddings...")
    generator = torch.Generator().manual_seed(42)
    images = pipe(
        prompt_embeds=conditioning_weighted.embeds,
        prompt_attention_mask=conditioning_weighted.attention_mask,
        num_inference_steps=20,
        guidance_scale=4.5,
        width=1024,
        height=1024,
        generator=generator,
    )
    print("generated, saving...")
    images.images[0].save("sana_compel_weighted.jpg")
