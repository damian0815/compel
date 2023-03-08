import unittest
from typing import List, Optional

import torch

from src.compel import EmbeddingsProvider
from src.compel.conditioning_scheduler import StaticConditioningScheduler, ConditioningScheduler
from prompting_test_utils import DummyTokenizer, DummyTransformer, KNOWN_WORDS, KNOWN_WORDS_TOKEN_IDS, NullTransformer

from src.compel.compel import Compel


def make_dummy_compel():
    tokenizer = DummyTokenizer()
    text_encoder = DummyTransformer()
    return Compel(tokenizer=tokenizer, text_encoder=text_encoder)


def make_test_conditioning(text_encoder: DummyTransformer,
                           tokenizer: DummyTokenizer,
                           token_ids: List[int],
                           truncate: bool=True,
                           pad_to_length: Optional[int]=None,
                           ) -> torch.Tensor:
    remaining_tokens = token_ids.copy()
    conditioning = None
    chunk_length = tokenizer.model_max_length-2
    while True:
        pre_padding = [tokenizer.bos_token_id]
        chunk_token_ids = remaining_tokens[0:chunk_length]
        remaining_tokens = remaining_tokens[chunk_length:]
        post_padding = [tokenizer.eos_token_id] + \
                       [tokenizer.pad_token_id] * (tokenizer.model_max_length - len(chunk_token_ids) - 2)
        chunk_token_ids = pre_padding + chunk_token_ids + post_padding
        assert len(chunk_token_ids) == tokenizer.model_max_length
        this_conditioning = text_encoder(input_ids=torch.tensor(chunk_token_ids,
                                                            dtype=torch.int,
                                                            device=text_encoder.device
                                                            ).unsqueeze(0)).last_hidden_state
        conditioning = (
            this_conditioning
            if conditioning is None
            else torch.cat([conditioning, this_conditioning], dim=1)
        )
        if truncate or len(remaining_tokens) == 0:
            break

    if pad_to_length is not None:
        empty_token_ids = [tokenizer.bos_token_id] + [tokenizer.eos_token_id] + [tokenizer.pad_token_id] * (tokenizer.model_max_length-2)
        empty_conditioning = text_encoder(input_ids=torch.tensor(empty_token_ids,
                                            dtype=torch.int,
                                            device=text_encoder.device
                                            ).unsqueeze(0)).last_hidden_state
        while pad_to_length > conditioning.shape[1]:
            conditioning = torch.cat([conditioning, empty_conditioning], dim=1)

    return conditioning


class EmbeddingsProviderTestCase(unittest.TestCase):

    def test_tokenizing(self):
        tokenizer = DummyTokenizer(model_max_length=5)
        embeddings_provider = EmbeddingsProvider(tokenizer=tokenizer, text_encoder=NullTransformer())

        prompts = ['a b']
        token_ids_tensor, weights_tensor = embeddings_provider.get_token_ids_and_expand_weights(prompts, weights=[0.8], device='cpu')
        self.assertTrue(torch.equal(token_ids_tensor, torch.tensor([3, 0, 1, 5, 4], dtype=torch.int64)))
        self.assertTrue(torch.equal(weights_tensor, torch.tensor([1.0] + [0.8] * 2 + [1.0] * 2)))

        prompts = ['a b c']
        token_ids_tensor, weights_tensor = embeddings_provider.get_token_ids_and_expand_weights(prompts, weights=[0.8], device='cpu')
        self.assertTrue(torch.equal(token_ids_tensor, torch.tensor([3, 0, 1, 2, 5], dtype=torch.int64)))
        self.assertTrue(torch.equal(weights_tensor, torch.tensor(([1.0] + [0.8] * 3 + [1.0]))))

    def test_long_tokenizing(self):
        tokenizer = DummyTokenizer(model_max_length=5)
        embeddings_provider = EmbeddingsProvider(tokenizer=tokenizer, text_encoder=NullTransformer(), truncate=False)

        prompts = ['a b c c b a a c b']
        token_ids_tensor, weights_tensor = embeddings_provider.get_token_ids_and_expand_weights(prompts, weights=[0.8], device='cpu')
        self.assertTrue(torch.equal(token_ids_tensor, torch.tensor([3, 0, 1, 2, 5, 3, 2, 1, 0, 5, 3, 0, 2, 1, 5], dtype=torch.int64)))
        self.assertTrue(torch.equal(weights_tensor, torch.tensor(([1.0] + [0.8] * 3 + [1.0]) * 3)))

        prompts = ['a b c c b a a c b a']
        token_ids_tensor, weights_tensor = embeddings_provider.get_token_ids_and_expand_weights(prompts, weights=[0.8], device='cpu')
        self.assertTrue(torch.equal(token_ids_tensor, torch.tensor([3, 0, 1, 2, 5, 3, 2, 1, 0, 5, 3, 0, 2, 1, 5, 3, 0, 5, 4, 4], dtype=torch.int64)))
        self.assertTrue(torch.equal(weights_tensor, torch.tensor(([1.0] + [0.8] * 3 + [1.0]) * 3 + [1.0] + [0.8] + [1.0] * 3)))

    def test_embeddings(self):
        tokenizer = DummyTokenizer(model_max_length=5)
        text_encoder = DummyTransformer()
        embeddings_provider = EmbeddingsProvider(tokenizer=tokenizer, text_encoder=text_encoder)

        prompts = ['a b']
        token_ids_tensor, weights_tensor = embeddings_provider.get_token_ids_and_expand_weights(prompts, weights=[0.8], device='cpu')
        embeddings_provider.build_weighted_embedding_tensor(token_ids_tensor, weights_tensor)

class CompelTestCase(unittest.TestCase):


    def test_basic_prompt(self):
        tokenizer = DummyTokenizer()
        text_encoder = DummyTransformer()
        compel = Compel(tokenizer=tokenizer, text_encoder=text_encoder)

        # test "a b c" makes it to the Conditioning intact for t=0, t=0.5, t=1
        prompt = " ".join(KNOWN_WORDS[:3])
        conditioning_scheduler = compel.make_conditioning_scheduler(prompt)
        conditioning_scheduler_2 = compel.make_conditioning_scheduler(prompt)
        expected_positive_conditioning = make_test_conditioning(text_encoder, tokenizer, KNOWN_WORDS_TOKEN_IDS[:3])
        expected_negative_conditioning = make_test_conditioning(text_encoder, tokenizer, [])
        self.assert_constant_scheduling_matches_expected(conditioning_scheduler,
                                                         expected_positive_conditioning,
                                                         expected_negative_conditioning)


    def test_basic_negative_prompt(self):
        tokenizer = DummyTokenizer()
        text_encoder = DummyTransformer()
        compel = Compel(tokenizer=tokenizer, text_encoder=text_encoder)

        # positive "a b c" negative "c b a" makes it to the Conditioning intact for t=0, t=0.5, t=1
        positive_prompt = " ".join(KNOWN_WORDS[:3])
        negative_prompt = " ".join(reversed(KNOWN_WORDS[:3]))
        conditioning_scheduler = compel.make_conditioning_scheduler(positive_prompt, negative_prompt)
        expected_positive_conditioning = make_test_conditioning(text_encoder, tokenizer, KNOWN_WORDS_TOKEN_IDS[:3])
        expected_negative_conditioning = make_test_conditioning(text_encoder, tokenizer, list(reversed(KNOWN_WORDS_TOKEN_IDS[:3]))
        )
        self.assert_constant_scheduling_matches_expected(conditioning_scheduler,
                                                         expected_positive_conditioning,
                                                         expected_negative_conditioning)

    def test_too_long_prompt_truncate(self):
        tokenizer = DummyTokenizer()
        text_encoder = DummyTransformer()
        compel = Compel(tokenizer=tokenizer, text_encoder=text_encoder)

        positive_prompt = " ".join(KNOWN_WORDS[:3] * 40)
        conditioning_scheduler = compel.make_conditioning_scheduler(positive_prompt)
        expected_positive_conditioning = make_test_conditioning(text_encoder, tokenizer, KNOWN_WORDS_TOKEN_IDS[:3] * 40)
        expected_negative_conditioning = make_test_conditioning(text_encoder, tokenizer, [])
        self.assert_constant_scheduling_matches_expected(conditioning_scheduler,
                                                         expected_positive_conditioning,
                                                         expected_negative_conditioning)


    def test_too_long_prompt_notruncate(self):
        tokenizer = DummyTokenizer(model_max_length=10)
        text_encoder = DummyTransformer()
        compel = Compel(tokenizer=tokenizer, text_encoder=text_encoder, truncate_long_prompts=False)

        positive_prompt = " ".join(KNOWN_WORDS[:3] * 4)
        conditioning_scheduler = compel.make_conditioning_scheduler(positive_prompt)
        expected_positive_conditioning = make_test_conditioning(text_encoder,
                                                                tokenizer,
                                                                KNOWN_WORDS_TOKEN_IDS[:3] * 4,
                                                                truncate=False)
        expected_negative_conditioning = make_test_conditioning(text_encoder,
                                                                tokenizer, [],
                                                                pad_to_length=expected_positive_conditioning.shape[1],
                                                                truncate=False)

        self.assert_constant_scheduling_matches_expected(conditioning_scheduler,
                                                         expected_positive_conditioning,
                                                         expected_negative_conditioning)


    def test_device(self):
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = DummyTokenizer()
        text_encoder = DummyTransformer(device=device)
        compel = Compel(tokenizer=tokenizer, text_encoder=text_encoder)

        # test "a b c" makes it to the Conditioning intact for t=0, t=0.5, t=1
        prompt = " ".join(KNOWN_WORDS[:3])
        conditioning_scheduler = compel.make_conditioning_scheduler(prompt)
        expected_positive_conditioning = make_test_conditioning(text_encoder, tokenizer, KNOWN_WORDS_TOKEN_IDS[:3])
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
