from typing import Union

import torch
from transformers import CLIPTokenizer, CLIPTextModel

from .conditioning_scheduler import ConditioningScheduler, StaticConditioningScheduler
from .embeddings_provider import EmbeddingsProvider
from .prompt_parser import Blend, FlattenedPrompt, PromptParser

__all__ = ["Compel"]

class Compel:

    def __init__(self, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel):
        self.conditioning_provider = EmbeddingsProvider(tokenizer=tokenizer, text_encoder=text_encoder)

    def make_conditioning_scheduler(self, positive_prompt: str, negative_prompt: str='') -> ConditioningScheduler:
        positive_conditioning = self.build_conditioning_tensor(positive_prompt)
        negative_conditioning = self.build_conditioning_tensor(negative_prompt)
        return StaticConditioningScheduler(positive_conditioning=positive_conditioning,
                                           negative_conditioning=negative_conditioning)

    def build_conditioning_tensor(self, text: str) -> torch.Tensor:
        prompt_object = self.parse_prompt_string(text)
        conditioning = self.build_conditioning_tensor_for_prompt_object(prompt_object)
        return conditioning

    @classmethod
    def parse_prompt_string(cls, prompt_string: str) -> Union[FlattenedPrompt, Blend]:
        pp = PromptParser()
        conjunction = pp.parse_conjunction(prompt_string)
        # we don't support conjunctions for now
        parsed_prompt = conjunction.prompts[0]
        return parsed_prompt

    def describe_tokenization(self, text: str) -> list[str]:
        """
        For the given text, return a list of strings showing how it will be tokenized.

        :param text: The text that is to be tokenized.
        :return: A list of strings representing the output of the tokenizer. It's expected that the output list may be
        longer than the number of words in `text` because the tokenizer may split words to multiple tokens. Because of
        this, word boundaries are indicated in the output with `</w>` strings.
        """
        return self.conditioning_provider.tokenizer.tokenize(text)

    def build_conditioning_tensor_for_prompt_object(self, prompt: Union[Blend, FlattenedPrompt]) -> torch.Tensor:
        """

        """
        if type(prompt) is Blend:
            return self._get_conditioning_for_blend(prompt)
        elif type(prompt) is FlattenedPrompt:
            if prompt.wants_cross_attention_control:
                raise NotImplementedError("Not implemented yet - see InvokeAI for reference implementation")
                # conditioning, cac_args = self._get_conditioning_for_cross_attention_control(model, parsed_prompt, log_tokens)
            else:
                return self._get_conditioning_for_flattened_prompt(prompt)

        raise ValueError(f"unsupported prompt type: {type(prompt).__name__}")

    def _get_conditioning_for_flattened_prompt(self, prompt: FlattenedPrompt) -> torch.Tensor:
        if type(prompt) is not FlattenedPrompt:
            raise ValueError(f"embeddings can only be made from FlattenedPrompts, got {type(prompt).__name__} instead")
        fragments = [x.text for x in prompt.children]
        weights = [x.weight for x in prompt.children]
        conditioning = self.conditioning_provider.get_embeddings_for_weighted_prompt_fragments(text_batch=[fragments],
                                                                                               fragment_weights_batch=[
                                                                                                   weights])
        return conditioning

    def _get_conditioning_for_blend(self, blend: Blend):
        conditionings_to_blend = None
        for i, flattened_prompt in enumerate(blend.prompts):
            this_conditioning = self._get_conditioning_for_flattened_prompt(flattened_prompt)
            conditionings_to_blend = this_conditioning if conditionings_to_blend is None else torch.cat(
                (conditionings_to_blend, this_conditioning))
        conditioning = EmbeddingsProvider.apply_embedding_weights(conditionings_to_blend.unsqueeze(0),
                                                                  blend.weights,
                                                                  normalize=blend.normalize_weights)
        return conditioning


