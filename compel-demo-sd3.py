import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers import StableDiffusion3Pipeline

from transformers.models.t5.modeling_t5 import T5EncoderModel

device='cpu'
pipeline: StableDiffusion3Pipeline = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers",
                                             torch_dtype=torch.float16).to(device, dtype=torch.float32)


text_encoder_3: T5EncoderModel = pipeline.text_encoder_3
tokenizer_3 = pipeline.tokenizer_3

prompt = "this is a test"
tokenized_prompt = tokenizer_3(prompt)
print(tokenized_prompt)
encoded_prompt = text_encoder_3(input_ids=torch.tensor([tokenized_prompt.input_ids]))
print(encoded_prompt)

max_sequence_length = 256 # ?
text_inputs = tokenizer_3(
    prompt,
    padding="max_length",
    max_length=max_sequence_length,
    truncation=True,
    add_special_tokens=True,
    return_tensors="pt",
)
text_input_ids = text_inputs.input_ids
untruncated_ids = tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids


#images = pipeline(    prompt=prompt,    negative_prompt="",    num_inference_steps=28,    height=1024,    width=1024,    guidance_scale=7.0)

#embeddings = pipeline.encode_prompt(prompt, prompt, prompt)

#compel = Compel(tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3] ,
#                text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3])

compel = Compel(tokenizer=pipeline.tokenizer_3,
                text_encoder=pipeline.text_encoder_3)
embeddings = compel("a big test")
print(embeddings)
