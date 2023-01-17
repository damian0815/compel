import unittest

import torch

from incite.conditioning_scheduler import ConditioningSchedulerFactory, StaticConditioningScheduler
from incite.prompt_to_embeddings_converter import PromptToEmbeddingsConverter
from incite.textual_inversion_manager import TextualInversionManager
from test.prompting_test_utils import DummyTokenizer, DummyTransformer, KNOWN_WORDS, KNOWN_WORDS_TOKEN_IDS


def make_dummy_conditioning_scheduler_factor():
    tokenizer = DummyTokenizer()
    text_encoder = DummyTransformer()
    textual_inversion_manager = TextualInversionManager(tokenizer, text_encoder)
    return ConditioningSchedulerFactory(
        prompt_to_embeddings_converter=PromptToEmbeddingsConverter(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            textual_inversion_manager=textual_inversion_manager
        )
    )


class TestPromptToEmbeddings(unittest.TestCase):

    def test_basic_prompt_to_conditioning(self):
        csf = make_dummy_conditioning_scheduler_factor()

        # test "a b c" makes it to the Conditioning intact for t=0, t=0.5, t=1
        prompt_string = " ".join(KNOWN_WORDS[:3])
        cfg_scale = 7.5
        conditioning_scheduler = csf.make_conditioning_scheduler(prompt_string, cfg_scale=cfg_scale)
        expected_positive_conditioning = csf.text_encoder(torch.Tensor([]))
        expected_negative_conditioning = csf.text_encoder(torch.Tensor(KNOWN_WORDS_TOKEN_IDS[:3]))
        self.assert_constant_scheduling_matches_expected(conditioning_scheduler,
                                                         expected_positive_conditioning,
                                                         expected_negative_conditioning,
                                                         cfg_scale)


    def test_basic_with_negative_prompt_to_conditioning(self):
        csf = make_dummy_conditioning_scheduler_factor()
        # test "a b c" makes it to the Conditioning intact for t=0, t=0.5, t=1
        # "a b c [ c b a ]" makes it to the Conditioning intact for t=0, t=0.5, t=1
        prompt_string = " ".join(KNOWN_WORDS[:3]) + " [ " + " ".join(reversed(KNOWN_WORDS[:3])) + " ] "
        cfg_scale = 7.5
        conditioning_scheduler = csf.make_conditioning_scheduler(prompt_string, cfg_scale)
        expected_positive_conditioning = csf.text_encoder(torch.Tensor(reversed(KNOWN_WORDS_TOKEN_IDS[:3])))
        expected_negative_conditioning = csf.text_encoder(torch.Tensor(KNOWN_WORDS_TOKEN_IDS[:3]))
        self.assert_constant_scheduling_matches_expected(conditioning_scheduler,
                                                         expected_positive_conditioning,
                                                         expected_negative_conditioning,
                                                         cfg_scale)


    def assert_constant_scheduling_matches_expected(self,
                                                    conditioning_scheduler: StaticConditioningScheduler,
                                                    expected_positive_conditioning: torch.Tensor,
                                                    expected_negative_conditioning: torch.Tensor,
                                                    cfg_scale: float):
        conditioning_at_start = conditioning_scheduler.get_conditioning_for_step_pct(0)
        self.assertEqual(cfg_scale, conditioning_at_start.cfg_scale)
        self.assertTrue(torch.equal(expected_positive_conditioning, conditioning_at_start.positive_conditioning))
        self.assertTrue(torch.equal(expected_negative_conditioning, conditioning_at_start.negative_conditioning))

        conditioning_at_mid = conditioning_scheduler.get_conditioning_for_step_pct(0.5)
        self.assertEqual(cfg_scale, conditioning_at_mid.cfg_scale)
        self.assertTrue(torch.equal(expected_positive_conditioning, conditioning_at_mid.positive_conditioning))
        self.assertTrue(torch.equal(expected_negative_conditioning, conditioning_at_mid.negative_conditioning))

        conditioning_at_end = conditioning_scheduler.get_conditioning_for_step_pct(1.0)
        self.assertEqual(cfg_scale, conditioning_at_end.cfg_scale)
        self.assertTrue(torch.equal(expected_positive_conditioning, conditioning_at_end.positive_conditioning))
        self.assertTrue(torch.equal(expected_negative_conditioning, conditioning_at_end.negative_conditioning))


    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
