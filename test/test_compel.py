import unittest

import torch

from src.compel.conditioning_scheduler import StaticConditioningScheduler, ConditioningScheduler
from prompting_test_utils import DummyTokenizer, DummyTransformer, KNOWN_WORDS, KNOWN_WORDS_TOKEN_IDS

from src.compel.compel import Compel


def make_dummy_compel():
    tokenizer = DummyTokenizer()
    text_encoder = DummyTransformer()
    return Compel(tokenizer=tokenizer, text_encoder=text_encoder)


def make_test_conditioning(text_encoder: DummyTransformer, tokenizer: DummyTokenizer, token_ids: list[int]) -> torch.Tensor:
    pre_padding = [tokenizer.bos_token_id]
    token_ids = token_ids[0:tokenizer.model_max_length-2]
    post_padding = [tokenizer.eos_token_id] * (tokenizer.model_max_length - len(token_ids) - 1)
    token_ids = pre_padding + token_ids + post_padding
    assert len(token_ids) == tokenizer.model_max_length
    conditioning =  text_encoder(input_ids=torch.tensor(token_ids, dtype=torch.int).unsqueeze(0)).last_hidden_state
    return conditioning


class TestPromptToEmbeddings(unittest.TestCase):

    def test_basic_prompt(self):
        tokenizer = DummyTokenizer()
        text_encoder = DummyTransformer()
        incite = Compel(tokenizer=tokenizer, text_encoder=text_encoder)

        # test "a b c" makes it to the Conditioning intact for t=0, t=0.5, t=1
        prompt = " ".join(KNOWN_WORDS[:3])
        conditioning_scheduler = incite.make_conditioning_scheduler(prompt)
        conditioning_scheduler_2 = incite.make_conditioning_scheduler(prompt)
        expected_positive_conditioning = make_test_conditioning(text_encoder, tokenizer, KNOWN_WORDS_TOKEN_IDS[:3])
        expected_negative_conditioning = make_test_conditioning(text_encoder, tokenizer, [])
        self.assert_constant_scheduling_matches_expected(conditioning_scheduler,
                                                         expected_positive_conditioning,
                                                         expected_negative_conditioning)


    def test_basic_negative_prompt(self):
        tokenizer = DummyTokenizer()
        text_encoder = DummyTransformer()
        incite = Compel(tokenizer=tokenizer, text_encoder=text_encoder)

        # positive "a b c" negative "c b a" makes it to the Conditioning intact for t=0, t=0.5, t=1
        positive_prompt = " ".join(KNOWN_WORDS[:3])
        negative_prompt = " ".join(reversed(KNOWN_WORDS[:3]))
        conditioning_scheduler = incite.make_conditioning_scheduler(positive_prompt, negative_prompt)
        expected_positive_conditioning = make_test_conditioning(text_encoder, tokenizer, KNOWN_WORDS_TOKEN_IDS[:3])
        expected_negative_conditioning = make_test_conditioning(text_encoder, tokenizer, list(reversed(KNOWN_WORDS_TOKEN_IDS[:3]))
        )
        self.assert_constant_scheduling_matches_expected(conditioning_scheduler,
                                                         expected_positive_conditioning,
                                                         expected_negative_conditioning)

    def test_too_long_prompt(self):
        tokenizer = DummyTokenizer()
        text_encoder = DummyTransformer()
        incite = Compel(tokenizer=tokenizer, text_encoder=text_encoder)

        # positive "a b c" negative "c b a" makes it to the Conditioning intact for t=0, t=0.5, t=1
        positive_prompt = " ".join(KNOWN_WORDS[:3] * 40)
        conditioning_scheduler = incite.make_conditioning_scheduler(positive_prompt)
        expected_positive_conditioning = make_test_conditioning(text_encoder, tokenizer, KNOWN_WORDS_TOKEN_IDS[:3] * 40)
        expected_negative_conditioning = make_test_conditioning(text_encoder, tokenizer, [])
        self.assert_constant_scheduling_matches_expected(conditioning_scheduler,
                                                         expected_positive_conditioning,
                                                         expected_negative_conditioning)


    def assert_constant_scheduling_matches_expected(self,
                                                    conditioning_scheduler: ConditioningScheduler,
                                                    expected_positive_conditioning: torch.Tensor,
                                                    expected_negative_conditioning: torch.Tensor):
        self.assertIs(StaticConditioningScheduler, type(conditioning_scheduler))

        conditioning_at_start = conditioning_scheduler.get_conditioning_for_step_pct(0)
        self.assertTrue(torch.allclose(expected_positive_conditioning,
                                       conditioning_at_start.positive_conditioning,
                                       atol=1e-6))
        self.assertTrue(torch.allclose(expected_negative_conditioning,
                                       conditioning_at_start.negative_conditioning,
                                       atol=1e-6))

        conditioning_at_mid = conditioning_scheduler.get_conditioning_for_step_pct(0.5)
        self.assertTrue(torch.allclose(expected_positive_conditioning,
                                       conditioning_at_mid.positive_conditioning,
                                       atol=1e-6))
        self.assertTrue(torch.allclose(expected_negative_conditioning,
                                       conditioning_at_mid.negative_conditioning,
                                       atol=1e-6))

        conditioning_at_end = conditioning_scheduler.get_conditioning_for_step_pct(1.0)
        self.assertTrue(torch.allclose(expected_positive_conditioning,
                                       conditioning_at_end.positive_conditioning,
                                       atol=1e-6))
        self.assertTrue(torch.allclose(expected_negative_conditioning,
                                       conditioning_at_end.negative_conditioning,
                                       atol=1e-6))




if __name__ == '__main__':
    unittest.main()
