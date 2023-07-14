from dataclasses import dataclass
from typing import Union, Optional, Callable, List, Tuple

import torch
from torch import Tensor
from transformers import CLIPTokenizer, CLIPTextModel

from . import cross_attention_control
from .conditioning_scheduler import ConditioningScheduler, StaticConditioningScheduler
from .embeddings_provider import EmbeddingsProvider, BaseTextualInversionManager, DownweightMode, ReturnedEmbeddingsType, EmbeddingsProviderMulti
from .prompt_parser import Blend, FlattenedPrompt, PromptParser, CrossAttentionControlSubstitute, Conjunction

__all__ = ["Compel", "DownweightMode"]

@dataclass
class ExtraConditioningInfo:
    pass


class Compel:


    def __init__(self,
                 tokenizer: Union[CLIPTokenizer, List[CLIPTokenizer]],
                 text_encoder: Union[CLIPTextModel, List[CLIPTextModel]],
                 textual_inversion_manager: Optional[BaseTextualInversionManager] = None,
                 dtype_for_device_getter: Callable[[torch.device], torch.dtype] = lambda device: torch.float32,
                 truncate_long_prompts: bool = True,
                 padding_attention_mask_value: int = 1,
                 downweight_mode: DownweightMode = DownweightMode.MASK,
                 returned_embeddings_type: ReturnedEmbeddingsType = ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
                 requires_pooled: bool = False,
                 device: Optional[str] = None
                 ):
        """
        Initialize Compel. The tokenizer and text_encoder can be lifted directly from any DiffusionPipeline.

        `textual_inversion_manager`: Optional instance to handle expanding multi-vector textual inversion tokens.
        `dtype_for_device_getter`: A Callable that returns a torch dtype for a given device. You probably don't need to
            use this.
        `truncate_long_prompts`: if True, truncate input prompts to 77 tokens long including beginning/end markers
            (default behaviour).
            If False, do not truncate, and instead assemble as many 77 token long chunks, each capped by beginning/end
            markers, as is necessary to encode the whole prompt. You will likely need to supply both positive and
            negative prompts in this case - use `pad_conditioning_tensors_to_same_length` to prevent having tensor
            length mismatch errors when passing the embeds on to your DiffusionPipeline for inference.
        `padding_attention_mask_value`: Value to write into the attention mask for padding tokens. Stable Diffusion needs 1.
        `downweight_mode`: Specifies whether downweighting should be applied by MASKing out the downweighted tokens
            (default) or REMOVEing them (legacy behaviour; messes up position embeddings of tokens following).
        `returned_embeddings_type`: controls how the embedding vectors are taken from the result of running the text encoder over the parsed prompt's text
        `requires_pooled`: for SDXL, append the pooled embeddings when returning conditioning tensors
        `device`: The torch device on which the tensors should be created. If a device is not specified, the device will
            be the same as that of the `text_encoder` at the moment when `build_conditioning_tensor()` is called.
        """
        if isinstance(tokenizer, (tuple, list)) and not isinstance(text_encoder, (tuple, list)):
            raise ValueError("Cannot provide list of tokenizers, but not of text encoders.")
        elif not isinstance(tokenizer, (tuple, list)) and isinstance(text_encoder, (tuple, list)):
            raise ValueError("Cannot provide list of text encoders, but not of tokenizers.")
        elif isinstance(tokenizer, (tuple, list)) and isinstance(text_encoder, (tuple, list)):
            self.conditioning_provider = EmbeddingsProviderMulti(tokenizers=tokenizer,
                                                            text_encoders=text_encoder,
                                                            textual_inversion_manager=textual_inversion_manager,
                                                            dtype_for_device_getter=dtype_for_device_getter,
                                                            truncate=truncate_long_prompts,
                                                            padding_attention_mask_value = padding_attention_mask_value,
                                                            downweight_mode=downweight_mode,
                                                            returned_embeddings_type=returned_embeddings_type,
                                                            requires_pooled_mask = requires_pooled
            )
        else:
            self.conditioning_provider = EmbeddingsProvider(tokenizer=tokenizer,
                                                            text_encoder=text_encoder,
                                                            textual_inversion_manager=textual_inversion_manager,
                                                            dtype_for_device_getter=dtype_for_device_getter,
                                                            truncate=truncate_long_prompts,
                                                            padding_attention_mask_value = padding_attention_mask_value,
                                                            downweight_mode=downweight_mode,
                                                            returned_embeddings_type=returned_embeddings_type,
                                                            )
        self._device = device
        self.requires_pooled = requires_pooled

    @property
    def device(self):
        return self._device if self._device else self.conditioning_provider.text_encoder.device

    def make_conditioning_scheduler(self, positive_prompt: str, negative_prompt: str='')  -> ConditioningScheduler:
        """
        Return a ConditioningScheduler object that provides conditioning tensors for different diffusion steps (currently
        not fully implemented).
        """
        positive_conditioning = self.build_conditioning_tensor(positive_prompt)
        negative_conditioning = self.build_conditioning_tensor(negative_prompt)
        [positive_conditioning, negative_conditioning] = self.pad_conditioning_tensors_to_same_length(
            [positive_conditioning, negative_conditioning]
        )
        return StaticConditioningScheduler(positive_conditioning=positive_conditioning,
                                           negative_conditioning=negative_conditioning)

    def build_conditioning_tensor(self, text: str) -> torch.Tensor:
        """
        Build a conditioning tensor by parsing the text for Compel syntax, constructing a Conjunction, and then
        building a conditioning tensor from that Conjunction.
        """
        conjunction = self.parse_prompt_string(text)
        conditioning, _ = self.build_conditioning_tensor_for_conjunction(conjunction)

        if self.requires_pooled:
            pooled = self.conditioning_provider.get_pooled_embeddings([text])
            return conditioning, pooled
        else:
            return conditioning

    @torch.no_grad()
    def __call__(self, text: Union[str, List[str]]) -> torch.FloatTensor:
        """
        Take a string or a list of strings and build conditioning tensors to match.

        If multiple strings are passed, the resulting tensors will be padded until they have the same length.

        :return: A tensor consisting of conditioning tensors for each of the passed-in strings, concatenated along dim 0.
        """
        if not isinstance(text, list):
            text = [text]

        cond_tensor = []
        pooled = []
        for text_input in text:
            output = self.build_conditioning_tensor(text_input)

            if self.requires_pooled:
                cond_tensor.append(output[0])
                pooled.append(output[1])
            else:
                cond_tensor.append(output)

        cond_tensor = self.pad_conditioning_tensors_to_same_length(conditionings=cond_tensor)
        cond_tensor = torch.cat(cond_tensor)

        if self.requires_pooled:
            pooled = torch.cat(pooled)
            return cond_tensor, pooled
        else:
            return cond_tensor

    @classmethod
    def parse_prompt_string(cls, prompt_string: str) -> Conjunction:
        """
        Parse the given prompt string and return a structured Conjunction object that represents the prompt it contains.
        """
        pp = PromptParser()
        conjunction = pp.parse_conjunction(prompt_string)
        return conjunction

    def describe_tokenization(self, text: str) -> List[str]:
        """
        For the given text, return a list of strings showing how it will be tokenized.

        :param text: The text that is to be tokenized.
        :return: A list of strings representing the output of the tokenizer. It's expected that the output list may be
        longer than the number of words in `text` because the tokenizer may split words to multiple tokens. Because of
        this, word boundaries are indicated in the output with `</w>` strings.
        """
        return self.conditioning_provider.tokenizer.tokenize(text)


    def build_conditioning_tensor_for_conjunction(self, conjunction: Conjunction) -> Tuple[torch.Tensor, dict]:
        """
        Build a conditioning tensor for the given Conjunction object.
        :return: A tuple of (conditioning tensor, options dict). The contents of the options dict depends on the prompt,
        at the moment it is only used for returning cross-attention control conditioning data (`.swap()`).
        """
        if len(conjunction.prompts) > 1 and conjunction.type != 'AND':
            raise ValueError("Only AND conjunctions are supported by build_conditioning_tensor()")
        # concatenate each prompt in the conjunction (typically there will only be 1)
        to_concat = []
        options = {}
        empty_conditioning = None
        for i, p in enumerate(conjunction.prompts):
            this_conditioning, this_options = self.build_conditioning_tensor_for_prompt_object(p)
            options.update(this_options)  # this is not a smart way to do this but 🤷‍
            weight = conjunction.weights[i]
            if weight != 1:
                # apply weight if we need to
                empty_conditioning = self.build_conditioning_tensor('') if empty_conditioning is None else empty_conditioning
                [padded_empty_conditioning, _] = self.pad_conditioning_tensors_to_same_length([empty_conditioning, this_conditioning])
                this_conditioning = padded_empty_conditioning + (this_conditioning - padded_empty_conditioning) * weight
            to_concat.append(this_conditioning)
        return torch.concat(to_concat, dim=1), options


    def build_conditioning_tensor_for_prompt_object(self, prompt: Union[Blend, FlattenedPrompt],
                                                    ) -> Tuple[torch.Tensor, dict]:
        """
        Build a conditioning tensor for the given prompt object (either a Blend or a FlattenedPrompt).
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

    @classmethod
    def _pad_conditioning_tensors_to_same_length(cls, conditionings: List[torch.Tensor], emptystring_conditioning: torch.Tensor
                                                 ) -> List[torch.Tensor]:
        c0_shape = conditionings[0].shape
        if not all([len(c.shape) == len(c0_shape) for c in conditionings]):
            raise ValueError("Conditioning tensors must all have either 2 dimensions (unbatched) or 3 dimensions (batched)")

        if len(c0_shape) == 2:
            # need to be unsqueezed
            conditionings = [c.unsqueeze(0) for c in conditionings]
            c0_shape = conditionings[0].shape
        if len(c0_shape) != 3:
            raise ValueError(f"All conditioning tensors must have the same number of dimensions (2 or 3)")

        if not all([c.shape[0] == c0_shape[0] and c.shape[2] == c0_shape[2] for c in conditionings]):
            raise ValueError(f"All conditioning tensors must have the same batch size ({c0_shape[0]}) and number of embeddings per token ({c0_shape[1]}")

        empty_z = torch.cat([emptystring_conditioning] * c0_shape[0])
        max_token_count = max([c.shape[1] for c in conditionings])
        # if necessary, pad shorter tensors out with an emptystring tensor
        for i, c in enumerate(conditionings):
            while c.shape[1] < max_token_count:
                c = torch.cat([c, empty_z], dim=1)
                conditionings[i] = c
        return conditionings


    def pad_conditioning_tensors_to_same_length(self, conditionings: List[torch.Tensor],
                                                ) -> List[torch.Tensor]:
        """
        If `truncate_long_prompts` was set to False on initialization, or if your prompt includes a `.and()` operator,
        conditioning tensors do not have a fixed length. This is a problem when using a negative and a positive prompt
        to condition the diffusion process. This function pads any of the passed-in tensors, as necessary, to ensure
        they all have the same length, returning the padded tensors in the same order they are passed.

        Example:
            ``` python
            embeds = compel('("a cat playing in the forest", "an impressionist oil painting").and()')
            negative_embeds = compel("ugly, deformed, distorted")
            [embeds, negative_embeds] = compel.pad_conditioning_tensors_to_same_length([embeds, negative_embeds])
            ```
        """
        emptystring_conditioning = self.build_conditioning_tensor("")
        if type(emptystring_conditioning) is tuple:
            # discard pooled
            emptystring_conditioning = emptystring_conditioning[0]
        return type(self)._pad_conditioning_tensors_to_same_length(conditionings, emptystring_conditioning=emptystring_conditioning)



    def _get_conditioning_for_flattened_prompt(self,
                                               prompt: FlattenedPrompt,
                                               should_return_tokens: bool=False
                                               ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if type(prompt) is not FlattenedPrompt:
            raise ValueError(f"embeddings can only be made from FlattenedPrompts, got {type(prompt).__name__} instead")
        fragments = [x.text for x in prompt.children]
        weights = [x.weight for x in prompt.children]
        return self.conditioning_provider.get_embeddings_for_weighted_prompt_fragments(
            text_batch=[fragments], fragment_weights_batch=[weights],
            should_return_tokens=should_return_tokens, device=self.device)


    def _get_conditioning_for_blend(self, blend: Blend):
        conditionings_to_blend = []
        for i, flattened_prompt in enumerate(blend.prompts):
            this_conditioning = self._get_conditioning_for_flattened_prompt(flattened_prompt)
            conditionings_to_blend.append(this_conditioning)
        conditionings_to_blend = self.pad_conditioning_tensors_to_same_length(conditionings_to_blend)
        conditionings_to_blend_tensor = torch.cat(conditionings_to_blend).unsqueeze(0)
        conditioning = EmbeddingsProvider.apply_embedding_weights(conditionings_to_blend_tensor,
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
        [original_conditioning, edited_conditioning] = self.pad_conditioning_tensors_to_same_length(
            [original_embeddings, edited_embeddings])
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

