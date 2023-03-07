from dataclasses import dataclass
from typing import Union, Optional, Callable, List, Tuple

import torch
from transformers import CLIPTokenizer, CLIPTextModel

from . import cross_attention_control
from .conditioning_scheduler import ConditioningScheduler, StaticConditioningScheduler
from .embeddings_provider import EmbeddingsProvider, BaseTextualInversionManager
from .prompt_parser import Blend, FlattenedPrompt, PromptParser, CrossAttentionControlSubstitute

__all__ = ["Compel"]

@dataclass
class ExtraConditioningInfo:
    pass

class Compel:

    def __init__(self,
                 tokenizer: CLIPTokenizer,
                 text_encoder: CLIPTextModel,
                 textual_inversion_manager: Optional[BaseTextualInversionManager] = None,
                 dtype_for_device_getter: Callable[[torch.device], torch.dtype] = lambda device: torch.float32):
        self.conditioning_provider = EmbeddingsProvider(tokenizer=tokenizer,
                                                        text_encoder=text_encoder,
                                                        textual_inversion_manager=textual_inversion_manager,
                                                        dtype_for_device_getter=dtype_for_device_getter
                                                        )

    @property
    def device(self):
        return self.conditioning_provider.text_encoder.device

    def make_conditioning_scheduler(self, positive_prompt: str, negative_prompt: str='') -> ConditioningScheduler:
        positive_conditioning = self.build_conditioning_tensor(positive_prompt)
        negative_conditioning = self.build_conditioning_tensor(negative_prompt)
        return StaticConditioningScheduler(positive_conditioning=positive_conditioning,
                                           negative_conditioning=negative_conditioning)

    def build_conditioning_tensor(self, text: str) -> torch.Tensor:
        prompt_object = self.parse_prompt_string(text)
        conditioning, _ = self.build_conditioning_tensor_for_prompt_object(prompt_object)
        return conditioning

    @torch.no_grad()
    def __call__(self, text: Union[str, List[str]]) -> torch.Tensor:
        if not isinstance(text, list):
            text = [text]

        cond_tensor = []
        for text_input in text:
            cond_tensor.append(self.build_conditioning_tensor(text_input))

        cond_tensor = torch.cat(cond_tensor)

        return cond_tensor

    @classmethod
    def parse_prompt_string(cls, prompt_string: str) -> Union[FlattenedPrompt, Blend]:
        pp = PromptParser()
        conjunction = pp.parse_conjunction(prompt_string)
        # we don't support conjunctions for now
        parsed_prompt = conjunction.prompts[0]
        return parsed_prompt

    def describe_tokenization(self, text: str) -> List[str]:
        """
        For the given text, return a list of strings showing how it will be tokenized.

        :param text: The text that is to be tokenized.
        :return: A list of strings representing the output of the tokenizer. It's expected that the output list may be
        longer than the number of words in `text` because the tokenizer may split words to multiple tokens. Because of
        this, word boundaries are indicated in the output with `</w>` strings.
        """
        return self.conditioning_provider.tokenizer.tokenize(text)

    def build_conditioning_tensor_for_prompt_object(self, prompt: Union[Blend, FlattenedPrompt],
                                                    ) -> Tuple[torch.Tensor, dict]:
        """

        """
        if type(prompt) is Blend:
            return self._get_conditioning_for_blend(prompt), {}
        elif type(prompt) is FlattenedPrompt:
            if prompt.wants_cross_attention_control:
                cac_args = self._get_conditioning_for_cross_attention_control(prompt)
                return cac_args.original_conditioning, { 'cross_attention_control': cac_args }
            else:
                return self._get_conditioning_for_flattened_prompt(prompt), {}

        raise ValueError(f"unsupported prompt type: {type(prompt).__name__}")

    def _get_conditioning_for_flattened_prompt(self, prompt: FlattenedPrompt, should_return_tokens: bool=False
                                               ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if type(prompt) is not FlattenedPrompt:
            raise ValueError(f"embeddings can only be made from FlattenedPrompts, got {type(prompt).__name__} instead")
        fragments = [x.text for x in prompt.children]
        weights = [x.weight for x in prompt.children]
        conditioning, tokens = self.conditioning_provider.get_embeddings_for_weighted_prompt_fragments(
            text_batch=[fragments], fragment_weights_batch=[weights], should_return_tokens=True, device=self.device)
        if should_return_tokens:
            return conditioning, tokens
        else:
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



    def _get_conditioning_for_cross_attention_control(self, prompt: FlattenedPrompt) -> cross_attention_control.Arguments:
        original_prompt = FlattenedPrompt()
        edited_prompt = FlattenedPrompt()
        # for name, a0, a1, b0, b1 in edit_opcodes: only name == 'equal' is currently parsed
        original_token_count = 0
        edited_token_count = 0
        edit_options = []
        edit_opcodes = []
        # beginning of sequence
        edit_opcodes.append(
            ('equal', original_token_count, original_token_count + 1, edited_token_count, edited_token_count + 1))
        edit_options.append(None)
        original_token_count += 1
        edited_token_count += 1
        for fragment in prompt.children:
            if type(fragment) is CrossAttentionControlSubstitute:
                original_prompt.append(fragment.original)
                edited_prompt.append(fragment.edited)

                to_replace_token_count = self._get_tokens_length([x.text for x in fragment.original])
                replacement_token_count = self._get_tokens_length([x.text for x in fragment.edited])
                edit_opcodes.append(('replace',
                                     original_token_count, original_token_count + to_replace_token_count,
                                     edited_token_count, edited_token_count + replacement_token_count
                                     ))
                original_token_count += to_replace_token_count
                edited_token_count += replacement_token_count
                edit_options.append(fragment.options)
            else:
                # regular fragment
                original_prompt.append(fragment)
                edited_prompt.append(fragment)

                count = self._get_tokens_length([fragment.text])
                edit_opcodes.append(('equal', original_token_count, original_token_count + count, edited_token_count,
                                     edited_token_count + count))
                edit_options.append(None)
                original_token_count += count
                edited_token_count += count
        # end of sequence
        edit_opcodes.append(
            ('equal', original_token_count, original_token_count + 1, edited_token_count, edited_token_count + 1))
        edit_options.append(None)
        original_token_count += 1
        edited_token_count += 1
        original_embeddings, original_tokens = self._get_conditioning_for_flattened_prompt(
            original_prompt, should_return_tokens=True
        )
        # naïvely building a single edited_embeddings like this disregards the effects of changing the absolute location of
        # subsequent tokens when there is >1 edit and earlier edits change the total token count.
        # eg "a cat.swap(smiling dog, s_start=0.5) eating a hotdog.swap(pizza)" - when the 'pizza' edit is active but the
        # 'cat' edit is not, the 'pizza' feature vector will nevertheless be affected by the introduction of the extra
        # token 'smiling' in the inactive 'cat' edit.
        # todo: build multiple edited_embeddings, one for each edit, and pass just the edited fragments through to the CrossAttentionControl functions
        edited_embeddings, edited_tokens = self._get_conditioning_for_flattened_prompt(
            edited_prompt, should_return_tokens=True
        )
        original_conditioning = original_embeddings
        edited_conditioning = edited_embeddings
        cac_args = cross_attention_control.Arguments(
            original_conditioning=original_conditioning,
            edited_conditioning=edited_conditioning,
            edit_opcodes=edit_opcodes,
            edit_options=edit_options
        )
        return cac_args

    def _get_tokens_length(self, texts: [str]) -> int:
        tokens = self.conditioning_provider.get_token_ids(texts, include_start_and_end_markers=False)
        return sum([len(x) for x in tokens])

    def get_tokens(self, text: str) -> List[int]:
        return self.conditioning_provider.get_token_ids([text], include_start_and_end_markers=False)[0]

