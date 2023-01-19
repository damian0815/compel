import unittest

import torch

from incite.embeddings_provider import EmbeddingsProvider
from prompting_test_utils import DummyTokenizer, DummyTransformer, KNOWN_WORDS, KNOWN_WORDS_TOKEN_IDS

def make_dummy_embeddings_provider(max_length=10) -> EmbeddingsProvider:
    tokenizer = DummyTokenizer(max_length)
    text_encoder = DummyTransformer()
    return EmbeddingsProvider(tokenizer=tokenizer, text_encoder=text_encoder)


class MyTestCase(unittest.TestCase):

    def test_get_token_ids(self):
        ep = make_dummy_embeddings_provider()

        prompts = [" ".join(KNOWN_WORDS), " ".join(reversed(KNOWN_WORDS))]
        expected_token_ids = [KNOWN_WORDS_TOKEN_IDS, list(reversed(KNOWN_WORDS_TOKEN_IDS))]
        token_ids = ep.get_token_ids(prompts, include_start_and_end_markers=False)
        self.assertEqual(expected_token_ids, token_ids)

        expected_token_ids_with_bos_eos = [[ep.tokenizer.bos_token_id] + x + [ep.tokenizer.eos_token_id] for x in expected_token_ids]
        print(expected_token_ids_with_bos_eos)
        token_ids = ep.get_token_ids(prompts, include_start_and_end_markers=True)
        self.assertEqual(expected_token_ids_with_bos_eos, token_ids)


    def test_build_weighted_embeddings_tensor(self):
        max_length = 10
        ep = make_dummy_embeddings_provider(max_length=max_length)

        # check that if fails if too short
        token_ids = torch.tensor(KNOWN_WORDS_TOKEN_IDS)
        per_token_weights = torch.tensor([1] * len(KNOWN_WORDS_TOKEN_IDS))
        with self.assertRaises(ValueError):
            ep.build_weighted_embedding_tensor(token_ids, per_token_weights)

        # all weighted
        empty_z = ep.build_weighted_embedding_tensor(torch.tensor([ep.tokenizer.eos_token_id] * max_length), torch.tensor([1] * max_length))
        token_ids = torch.tensor([ep.tokenizer.bos_token_id] + KNOWN_WORDS_TOKEN_IDS + [ep.tokenizer.eos_token_id] * (max_length-1-len(KNOWN_WORDS_TOKEN_IDS)))
        weighted_embeddings_1 = ep.build_weighted_embedding_tensor(token_ids, torch.tensor([1] * len(token_ids)))
        weighted_embeddings_2 = ep.build_weighted_embedding_tensor(token_ids, torch.tensor([2] * len(token_ids)))
        self.assertTrue(torch.allclose(empty_z + (weighted_embeddings_1-empty_z) * 2,
                                       weighted_embeddings_2,
                                       atol=1e-7))

        # different weights at different places
        rand_weights = 1 + torch.rand([len(token_ids)])
        weighted_embeddings_rand = ep.build_weighted_embedding_tensor(token_ids, rand_weights)
        weights_expanded = rand_weights.unsqueeze(1).expand(-1,768).unsqueeze(0)
        self.assertTrue(torch.allclose(empty_z + (weighted_embeddings_1-empty_z) * weights_expanded,
                                       weighted_embeddings_rand,
                                       atol=1e-7))


    def test_weighted_prompt_fragments(self):
        max_length = 10
        ep = make_dummy_embeddings_provider(max_length=max_length)

        text_batch = [[' '.join(KNOWN_WORDS)]]
        fragment_weights_batch = [[1]]
        embeddings = ep.get_embeddings_for_weighted_prompt_fragments(text_batch, fragment_weights_batch)

        expected_token_ids = torch.tensor([ep.tokenizer.bos_token_id] + KNOWN_WORDS_TOKEN_IDS + [ep.tokenizer.eos_token_id] * (max_length-1-len(KNOWN_WORDS_TOKEN_IDS)))
        expected_embeddings = ep.build_weighted_embedding_tensor(expected_token_ids, torch.tensor([1] * len(expected_token_ids)))
        self.assertTrue(torch.allclose(expected_embeddings, embeddings, atol=1e-8))

        # weighted fragments
        text_batch = [[KNOWN_WORDS[0], ' '.join(KNOWN_WORDS[1:3])]]
        fragment_weights_batch = [[1, 2]]
        embeddings = ep.get_embeddings_for_weighted_prompt_fragments(text_batch, fragment_weights_batch)
        expected_weights = [1] + [1] + [2, 2] + [1] * 6
        expected_embeddings = ep.build_weighted_embedding_tensor(expected_token_ids, torch.tensor(expected_weights))
        self.assertTrue(torch.allclose(expected_embeddings, embeddings, atol=1e-8))


    def test_too_long_weighted_prompt_fragments(self):
        max_length = 10
        ep = make_dummy_embeddings_provider(max_length=max_length)


        # too many weighted fragments
        text_batch = [[KNOWN_WORDS[0], ' '.join(reversed(KNOWN_WORDS*3)), ' '.join(KNOWN_WORDS[1:3], )]]
        fragment_weights_batch = [[1, 2, 3]]
        embeddings = ep.get_embeddings_for_weighted_prompt_fragments(text_batch, fragment_weights_batch)

        expected_token_ids = torch.tensor([ep.tokenizer.bos_token_id] +
                                          [KNOWN_WORDS_TOKEN_IDS[0]] +
                                          reversed(KNOWN_WORDS_TOKEN_IDS*3) +
                                          [KNOWN_WORDS_TOKEN_IDS[1]] + [ep.tokenizer.eos_token_id])
        expected_weights = [1] + [1] + [2, 2, 2, 2, 2, 2] + [3] + [1]
        expected_embeddings = ep.build_weighted_embedding_tensor(expected_token_ids, torch.tensor(expected_weights))
        self.assertTrue(torch.allclose(expected_embeddings, embeddings, atol=1e-8))


if __name__ == '__main__':
    unittest.main()
