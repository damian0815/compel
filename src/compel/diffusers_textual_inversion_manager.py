from typing import List

from .embeddings_provider import BaseTextualInversionManager


class DiffusersTextualInversionManager(BaseTextualInversionManager):
    """
    A textual inversion manager for use with diffusers.
    """
    def __init__(self, pipe):
        self.pipe = pipe

    def expand_textual_inversion_token_ids_if_necessary(self, token_ids: List[int]) -> List[int]:
        if len(token_ids) == 0:
            return token_ids

        prompt = self.pipe.tokenizer.decode(token_ids)
        prompt = self.pipe.maybe_convert_prompt(prompt, self.pipe.tokenizer)
        return self.pipe.tokenizer.encode(prompt, add_special_tokens=False)
