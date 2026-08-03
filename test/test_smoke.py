import unittest
from unittest.mock import MagicMock

import torch

from compel import Compel, CompelForSD, CompelForSDXL, CompelForLumina2, CompelForSana

from prompting_test_utils import DummyTokenizer, DummyTransformer


class DummyT5Tokenizer(DummyTokenizer):
    def __call__(self, fragments, **kwargs):
        tokenized = [
            [self.tokens.index(w) for w in fragment.split(" ")] + [self.eos_token_id]
            if len(fragment) > 0
            else [self.eos_token_id]
            for fragment in fragments
        ]
        default_truncation = False
        if kwargs.get("truncation", default_truncation):
            max_length = kwargs.get("max_length", self.model_max_length)
            tokenized = [
                x[0 : max_length - 1] + [self.eos_token_id] if len(x) > max_length else x
                for x in tokenized
            ]
        padding_strategy = kwargs.get("padding", "do_not_pad")
        if padding_strategy not in ["do_not_pad", "max_length"]:
            raise Exception(
                "for unit tests only 'do_not_pad' and 'max_length' is supported "
                f"as a padding strategy (got '{padding_strategy}')"
            )

        if padding_strategy == "max_length":
            tokenized = [
                tokens[:-1]
                + (self.model_max_length - len(tokens)) * [self.pad_token_id]
                + tokens[-1:]
                for tokens in tokenized
            ]

        return {"input_ids": tokenized}


class SmokeTestCase(unittest.TestCase):
    def test_public_api_imports(self):
        from compel import Compel as ImportedCompel
        from compel import CompelForSD as ImportedCompelForSD

        self.assertIs(ImportedCompel, Compel)
        self.assertIs(ImportedCompelForSD, CompelForSD)

    def test_clip_backed_path(self):
        compel = Compel(tokenizer=DummyTokenizer(), text_encoder=DummyTransformer())

        conditioning = compel("a b c")

        self.assertEqual(conditioning.shape, (1, 77, 768))

    def test_compel_for_sd_wrapper_with_negative_prompt(self):
        pipeline = MagicMock()
        pipeline.tokenizer = DummyTokenizer(model_max_length=5)
        pipeline.text_encoder = DummyTransformer(text_model_max_length=5)

        conditioning = CompelForSD(pipeline)("a b c", negative_prompt="a")

        self.assertEqual(conditioning.embeds.shape, (1, 5, 768))
        self.assertEqual(conditioning.negative_embeds.shape, (1, 5, 768))

    def test_sdxl_style_path_returns_pooled_embeddings(self):
        pipeline = MagicMock()
        pipeline.tokenizer = DummyTokenizer(model_max_length=5)
        pipeline.tokenizer_2 = DummyTokenizer(model_max_length=5)
        pipeline.text_encoder = DummyTransformer(text_model_max_length=5, embedding_length=1280)
        pipeline.text_encoder_2 = DummyTransformer(text_model_max_length=5, embedding_length=768)

        conditioning = CompelForSDXL(pipeline)("a b c", style_prompt="b c")

        self.assertEqual(conditioning.embeds.shape, (1, 5, 2048))
        self.assertEqual(conditioning.pooled_embeds.shape, (1, 768))

    def test_compel_for_lumina2_wrapper_with_negative_prompt(self):
        pipeline = MagicMock()
        pipeline.tokenizer = DummyTokenizer(model_max_length=5)
        pipeline.text_encoder = DummyTransformer(text_model_max_length=5)

        conditioning = CompelForLumina2(pipeline, system_prompt="", max_sequence_length=5)("a b c", negative_prompt="a")

        self.assertEqual(conditioning.embeds.shape, (1, 5, 768))
        self.assertEqual(conditioning.negative_embeds.shape, (1, 5, 768))
        self.assertEqual(conditioning.attention_mask.shape, (1, 5))
        self.assertEqual(conditioning.negative_attention_mask.shape, (1, 5))
        self.assertIsNone(conditioning.pooled_embeds)
        self.assertIsNone(conditioning.negative_pooled_embeds)

        # masks are 1 at non-pad positions and 0 at pad positions
        self.assertEqual(conditioning.attention_mask[0, 0].item(), 1)
        self.assertEqual(conditioning.negative_attention_mask[0, 0].item(), 1)
        # all positions in these prompts are real tokens, so no padding
        self.assertEqual(conditioning.attention_mask.sum().item(), 5)
        self.assertEqual(conditioning.negative_attention_mask.sum().item(), 5)

        # a prompt short enough to leave padding gets masked out at the pad position
        conditioning_with_padding = CompelForLumina2(pipeline, system_prompt="", max_sequence_length=5)(
            "a b c", negative_prompt="")
        self.assertEqual(conditioning_with_padding.negative_attention_mask[0, 0].item(), 1)
        self.assertEqual(conditioning_with_padding.negative_attention_mask[0, -1].item(), 0)

    def test_compel_for_lumina2_wrapper_without_negative_prompt(self):
        pipeline = MagicMock()
        pipeline.tokenizer = DummyTokenizer(model_max_length=5)
        pipeline.text_encoder = DummyTransformer(text_model_max_length=5)

        conditioning = CompelForLumina2(pipeline, system_prompt="", max_sequence_length=5)("a b c")

        self.assertEqual(conditioning.embeds.shape, (1, 5, 768))
        self.assertEqual(conditioning.attention_mask.shape, (1, 5))
        self.assertIsNone(conditioning.negative_embeds)
        self.assertIsNone(conditioning.negative_attention_mask)

    def test_compel_for_sana_wrapper_with_negative_prompt(self):
        pipeline = MagicMock()
        pipeline.tokenizer = DummyTokenizer(model_max_length=5)
        pipeline.text_encoder = DummyTransformer(text_model_max_length=5)

        conditioning = CompelForSana(pipeline, max_sequence_length=5)("a b c", negative_prompt="a")

        self.assertEqual(conditioning.embeds.shape, (1, 5, 768))
        self.assertEqual(conditioning.negative_embeds.shape, (1, 5, 768))
        self.assertEqual(conditioning.attention_mask.shape, (1, 5))
        self.assertEqual(conditioning.negative_attention_mask.shape, (1, 5))
        self.assertIsNone(conditioning.pooled_embeds)
        self.assertIsNone(conditioning.negative_pooled_embeds)

        self.assertEqual(conditioning.attention_mask[0, 0].item(), 1)
        self.assertEqual(conditioning.negative_attention_mask[0, 0].item(), 1)

        # the wrapper lowercases and strips prompts
        lowercased = CompelForSana(pipeline, max_sequence_length=5)("a b c")
        uppercased = CompelForSana(pipeline, max_sequence_length=5)("A B C")
        self.assertTrue(torch.allclose(lowercased.embeds, uppercased.embeds))
        self.assertTrue(torch.equal(lowercased.attention_mask, uppercased.attention_mask))

    def test_t5_backed_path_if_supported(self):
        tokenizer = DummyT5Tokenizer(model_max_length=6)
        tokenizer.bos_token_id = None
        text_encoder = DummyTransformer(text_model_max_length=6)
        compel = Compel(tokenizer=tokenizer, text_encoder=text_encoder)

        conditioning = compel("a b c")

        self.assertEqual(conditioning.shape, (1, 6, 768))
        self.assertEqual(conditioning.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
