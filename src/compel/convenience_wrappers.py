from dataclasses import dataclass
from typing import Union, List, Optional

import torch
from diffusers import FluxPipeline, StableDiffusionXLPipeline, StableDiffusionPipeline

from compel import Compel, ReturnedEmbeddingsType, DiffusersTextualInversionManager, BaseTextualInversionManager


@dataclass(frozen=True)
class LabelledConditioning:
    embeds: torch.Tensor
    pooled_embeds: Union[torch.Tensor, None] = None
    negative_embeds: Union[torch.Tensor, None] = None
    negative_pooled_embeds: Union[torch.Tensor, None] = None


class CompelForSD:
    def __init__(self, pipe: StableDiffusionPipeline, textual_inversion_manager: Optional[BaseTextualInversionManager]=None):
        self.compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder,
                             textual_inversion_manager=textual_inversion_manager,
                             truncate_long_prompts=False)

    def disable_no_weights_bypass(self):
        self.compel.disable_no_weights_bypass()

    def __call__(self, prompt: Union[str, List[str]], negative_prompt: Union[None, str, List[str]] = None):
        input, negative_start_index = _make_compel_input_with_optional_negative(prompt, negative_prompt)
        embeds = self.compel(input)
        return LabelledConditioning(embeds=embeds[0:negative_start_index],
                                    pooled_embeds=None,
                                    negative_embeds=None if negative_start_index is None else embeds[negative_start_index:],
                                    negative_pooled_embeds=None,
                                    )


class CompelForFlux:
    def __init__(self, pipe: FluxPipeline, textual_inversion_manager: Optional[BaseTextualInversionManager]=None):
        self.compel_1 = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder,
                               returned_embeddings_type=ReturnedEmbeddingsType.POOLED,
                               textual_inversion_manager=textual_inversion_manager,
                               truncate_long_prompts=True)
        self.compel_2 = Compel(tokenizer=pipe.tokenizer_2, text_encoder=pipe.text_encoder_2,
                               truncate_long_prompts=True)

    def disable_no_weights_bypass(self):
        self.compel_1.disable_no_weights_bypass()
        self.compel_2.disable_no_weights_bypass()

    def __call__(self,
                 main_prompt: Union[str, List[str]], style_prompt: Union[None, str, List[str]] = None,
                 negative_prompt: Union[None, str, List[str]] = None, negative_style_prompt: Union[None, str, List[str]] = None):
        if style_prompt is None:
            style_prompt = main_prompt
        main_input, negative_main_start_index = _make_compel_input_with_optional_negative(main_prompt, negative_prompt)
        style_input, negative_style_start_index = _make_compel_input_with_optional_negative(style_prompt, negative_style_prompt)
        pooled_embeds = self.compel_1(style_input)
        embeds = self.compel_2(main_input)
        return LabelledConditioning(embeds=embeds[0:negative_main_start_index],
                                    pooled_embeds=pooled_embeds[0:negative_style_start_index],
                                    negative_embeds=None if negative_main_start_index is None else embeds[negative_main_start_index:],
                                    negative_pooled_embeds=None if negative_style_start_index is None else pooled_embeds[negative_style_start_index:],
                                    )


class CompelForSDXL:
    def __init__(self, pipe: StableDiffusionXLPipeline, textual_inversion_manager: Optional[BaseTextualInversionManager]=None):
        self.compel_1 = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder,
                               returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                               textual_inversion_manager=textual_inversion_manager,
                               truncate_long_prompts=False
                               )
        self.compel_2 = Compel(tokenizer=pipe.tokenizer_2, text_encoder=pipe.text_encoder_2,
                               returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                               textual_inversion_manager=textual_inversion_manager,
                               truncate_long_prompts=False,
                               requires_pooled=True,
                               )

    def disable_no_weights_bypass(self):
        self.compel_1.disable_no_weights_bypass()
        self.compel_2.disable_no_weights_bypass()

    def __call__(self,
                 main_prompt: Union[str, List[str]], style_prompt: Union[None, str, List[str]] = None,
                 negative_prompt: Union[None, str, List[str]] = None, negative_style_prompt: Union[None, str, List[str]] = None):
        if type(main_prompt) is str:
            main_prompt = [main_prompt]
        if type(negative_prompt) is str:
            negative_prompt = [negative_prompt]
        if type(style_prompt) is str:
            style_prompt = [style_prompt]
        if type(negative_style_prompt) is str:
            negative_style_prompt = [negative_style_prompt]

        if style_prompt is None:
            style_prompt = main_prompt
        if negative_style_prompt is None:
            negative_style_prompt = negative_prompt

        if len(style_prompt) != len(main_prompt):
            if len(style_prompt) == 1:
                style_prompt = style_prompt * len(main_prompt)
            else:
                raise ValueError("when using style prompts, you must pass either 1 or the same number of style prompts as main prompts")

        main_input, negative_main_start_index = _make_compel_input_with_optional_negative(main_prompt, negative_prompt)
        style_input, negative_style_start_index = _make_compel_input_with_optional_negative(style_prompt, negative_style_prompt)

        embeds_left = self.compel_1(main_input)
        embeds_right, pooled_embeds = self.compel_2(style_input)

        # cat together along the embedding dimension

        # 1. pad to the same length on dim 1 (tokens) to compensate for potentially different tokenization length in cases where the
        # style prompt is longer/shorter than the base prompt
        if embeds_left.shape[1] > embeds_right.shape[1]:
            padding, _ = self.compel_2([""])
            num_repeats = (embeds_left.shape[1]-embeds_right.shape[1]) // padding.shape[1]
            padding = padding.repeat(1, num_repeats, 1)
            embeds_right = torch.cat([embeds_right, padding], dim=1)
        elif embeds_right.shape[1] > embeds_left.shape[1]:
            padding = self.compel_1([""] * embeds_left.shape[0])
            num_repeats = (embeds_right.shape[1]-embeds_left.shape[1]) // padding.shape[1]
            padding = padding.repeat(1, num_repeats, 1)
            embeds_left = torch.cat([embeds_left, padding], dim=1)

        # 2. now cat along the embedding dimension
        embeds = torch.cat([embeds_left, embeds_right], dim=-1)

        if negative_main_start_index is not None and embeds.shape[0] - negative_main_start_index == 1:
            # need to repeat negatives
            num_repeats = negative_main_start_index
            embeds = torch.cat([embeds[0:negative_main_start_index], embeds[negative_main_start_index:].repeat(num_repeats, 1, 1)])
        if negative_style_start_index is not None and pooled_embeds.shape[0] - negative_style_start_index == 1:
            # need to repeat negatives
            num_repeats = negative_style_start_index
            pooled_embeds = torch.cat([pooled_embeds[0:negative_style_start_index], pooled_embeds[negative_style_start_index:].repeat(num_repeats, 1)])

        return LabelledConditioning(embeds=embeds[0:negative_main_start_index],
                                    pooled_embeds=pooled_embeds[0:negative_style_start_index],
                                    negative_embeds=None if negative_main_start_index is None else embeds[negative_main_start_index:],
                                    negative_pooled_embeds=None if negative_style_start_index is None else pooled_embeds[negative_style_start_index:],
                                    )


def _make_compel_input_with_optional_negative(positive_prompt: list[str], negative_prompt: Union[None, list[str]]):
    if negative_prompt is None:
        return positive_prompt, None

    if len(negative_prompt) != 1 and len(negative_prompt) != len(positive_prompt):
        raise ValueError("when using negative prompts, you must pass either 1 or the same number of negative prompts as positive prompts")

    main_input = positive_prompt + negative_prompt
    negative_main_start_index = len(positive_prompt)
    return main_input, negative_main_start_index
