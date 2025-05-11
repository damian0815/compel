import math
import unittest

import torch

from src.compel.embeddings_provider import EmbeddingsProvider, DownweightMode, SplitLongTextMode
from prompting_test_utils import DummyTokenizer, DummyTransformer, KNOWN_WORDS, KNOWN_WORDS_TOKEN_IDS, NullTransformer


def make_dummy_embeddings_provider(max_length=10, embedding_length=768, **kwargs) -> EmbeddingsProvider:
    tokenizer = DummyTokenizer(max_length)
    text_encoder = DummyTransformer(embedding_length=embedding_length)
    return EmbeddingsProvider(tokenizer=tokenizer, text_encoder=text_encoder, **kwargs)

BOS = len(KNOWN_WORDS)
PAD = BOS + 1
EOS = BOS + 2

class EmbeddingsProviderTestCase(unittest.TestCase):

    def test_tokenizing(self):
        tokenizer = DummyTokenizer(model_max_length=5)
        embeddings_provider = EmbeddingsProvider(tokenizer=tokenizer, text_encoder=NullTransformer(),
                                                 padding_attention_mask_value=0)

        prompts = ['a b']
        token_ids_tensor, weights_tensor, mask = embeddings_provider.get_token_ids_and_expand_weights(prompts,
                                                                                                      weights=[0.8],
                                                                                                      device='cpu')
        self.assertTrue(torch.equal(token_ids_tensor, torch.tensor([BOS, 0, 1, EOS, PAD], dtype=torch.int64)))
        self.assertTrue(torch.equal(weights_tensor, torch.tensor([1.0] + [0.8] * 2 + [1.0] * 2)))
        self.assertTrue(torch.equal(mask, torch.tensor([1, 1, 1, 1, 0])))

        prompts = ['a b c']
        token_ids_tensor, weights_tensor, mask = embeddings_provider.get_token_ids_and_expand_weights(prompts,
                                                                                                      weights=[0.8],
                                                                                                      device='cpu')
        self.assertTrue(torch.equal(token_ids_tensor, torch.tensor([BOS, 0, 1, 2, EOS], dtype=torch.int64)))
        self.assertTrue(torch.equal(weights_tensor, torch.tensor(([1.0] + [0.8] * 3 + [1.0]))))
        self.assertTrue(torch.equal(mask, torch.tensor([1, 1, 1, 1, 1])))

        prompts = ['']
        token_ids_tensor, weights_tensor, mask = embeddings_provider.get_token_ids_and_expand_weights(prompts,
                                                                                                      weights=[0.8],
                                                                                                      device='cpu')
        self.assertTrue(torch.equal(token_ids_tensor, torch.tensor([BOS, EOS, PAD, PAD, PAD], dtype=torch.int64)))
        self.assertTrue(torch.equal(weights_tensor, torch.tensor(([1.0] * 5))))
        self.assertTrue(torch.equal(mask, torch.tensor([1, 1, 0, 0, 0])))

        prompts = []
        token_ids_tensor, weights_tensor, mask = embeddings_provider.get_token_ids_and_expand_weights(prompts,
                                                                                                      weights=[],
                                                                                                      device='cpu')
        self.assertTrue(torch.equal(token_ids_tensor, torch.tensor([BOS, EOS, PAD, PAD, PAD], dtype=torch.int64)))
        self.assertTrue(torch.equal(weights_tensor, torch.tensor(([1.0] * 5))))
        self.assertTrue(torch.equal(mask, torch.tensor([1, 1, 0, 0, 0])))

        # truncation
        prompts = ['a b c a b c a b c']
        token_ids_tensor, weights_tensor, mask = embeddings_provider.get_token_ids_and_expand_weights(prompts,
                                                                                                      weights=[0.8],
                                                                                                      device='cpu')
        self.assertTrue(torch.equal(token_ids_tensor, torch.tensor([BOS, 0, 1, 2, EOS], dtype=torch.int64)))
        self.assertTrue(torch.equal(weights_tensor, torch.tensor(([1.0] + [0.8] * 3 + [1.0]))))
        self.assertTrue(torch.equal(mask, torch.tensor([1, 1, 1, 1, 1])))

    def test_long_tokenizing(self):
        tokenizer = DummyTokenizer(model_max_length=5)
        embeddings_provider = EmbeddingsProvider(tokenizer=tokenizer, text_encoder=NullTransformer(),
                                                 truncate=False, padding_attention_mask_value=0)

        prompts = ['a b c c b a a c b']
        token_ids_tensor, weights_tensor, mask = embeddings_provider.get_token_ids_and_expand_weights(prompts,
                                                                                                      weights=[0.8],
                                                                                                      device='cpu')
        self.assertTrue(torch.equal(token_ids_tensor,
                                    torch.tensor([BOS, 0, 1, 2, EOS] + [BOS, 2, 1, 0, EOS] + [BOS, 0, 2, 1, EOS],
                                                 dtype=torch.int64)))
        self.assertTrue(torch.equal(weights_tensor, torch.tensor(([1.0] + [0.8] * 3 + [1.0]) * 3)))
        self.assertTrue(torch.equal(mask, torch.tensor(([1, 1, 1, 1, 1] * 3))))

        prompts = ['a b c c b a a c b a']
        token_ids_tensor, weights_tensor, mask = embeddings_provider.get_token_ids_and_expand_weights(prompts,
                                                                                                      weights=[0.8],
                                                                                                      device='cpu')
        self.assertTrue(torch.equal(token_ids_tensor,
                                    torch.tensor([BOS, 0, 1, 2, EOS, BOS, 2, 1, 0, EOS, BOS, 0, 2, 1, EOS, BOS, 0, EOS, PAD, PAD],
                                                 dtype=torch.int64)))
        self.assertTrue(torch.equal(weights_tensor, torch.tensor(
            ([1.0] + [0.8] * 3 + [1.0]) * 3 + ([1.0] + [0.8] + [1.0]) + ([1.0] * 2))))
        self.assertTrue(torch.equal(mask, torch.tensor([1, 1, 1, 1, 1] * 3 + [1, 1, 1] + [0, 0])))

    def test_split(self):
        tokenizer = DummyTokenizer(model_max_length=7)
        text_encoder = DummyTransformer()

        tokens = ["gone</w>", ".</w>", "a", "b", "into</w>",
                  "a", "b", "c", "a", "b", #9
                  "c", "a", "b", "home</w>", #13
                  "a", "b", ";</w>", #16
                  "gone</w>", "home</w>"]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertTrue(tokens == tokenizer.convert_ids_to_tokens(token_ids))

        # brutal
        embeddings_provider = EmbeddingsProvider(tokenizer=tokenizer, text_encoder=text_encoder,
                                                 truncate=False, split_long_text_mode=SplitLongTextMode.BRUTAL)
        chunked = embeddings_provider._chunk_and_pad_token_ids(token_ids=token_ids, token_weights=[1] * len(token_ids), device='cpu')
        chunked_tokens = tokenizer.convert_ids_to_tokens(chunked[0].tolist())
        #print('BRUTAL:', chunked_tokens)
        self.assertEqual(chunked_tokens[8], tokens[5])
        self.assertEqual(chunked_tokens[15], tokens[10])
        self.assertEqual(chunked_tokens[22], tokens[15])

        # words
        embeddings_provider = EmbeddingsProvider(tokenizer=tokenizer, text_encoder=text_encoder,
                                                 truncate=False, split_long_text_mode=SplitLongTextMode.WORDS)
        chunked = embeddings_provider._chunk_and_pad_token_ids(token_ids=token_ids, token_weights=[1] * len(token_ids), device='cpu')
        chunked_tokens = tokenizer.convert_ids_to_tokens(chunked[0].tolist())
        #print('WORDS:', chunked_tokens)
        self.assertEqual(chunked_tokens[8], tokens[5])
        self.assertEqual(chunked_tokens[15], tokens[10])
        self.assertEqual(chunked_tokens[22], tokens[14])
        self.assertEqual(chunked_tokens[26], tokens[18])

    def test_split_phrases(self):
        # phrases
        tokenizer = DummyTokenizer(model_max_length=7)
        text_encoder = DummyTransformer()

        tokens = [
                  "a", "b", ";</w>", #3
                  "gone</w>", "home</w>", "a", ".</w>" #7
            ]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        embeddings_provider = EmbeddingsProvider(tokenizer=tokenizer, text_encoder=text_encoder,
                                                 truncate=False, split_long_text_mode=SplitLongTextMode.PHRASES)
        chunked = embeddings_provider._chunk_and_pad_token_ids(token_ids=token_ids, token_weights=[1] * len(token_ids), device='cpu')
        chunked_tokens = tokenizer.convert_ids_to_tokens(chunked[0].tolist())
        #print('PHRASES:', chunked_tokens)
        self.assertTrue(len(chunked_tokens) == 14)
        self.assertEqual(chunked_tokens,[
            '<|bos|>', 'a', 'b', ';</w>', '<|eos|>', '<|pad|>', '<|pad|>',
            '<|bos|>', 'gone</w>', 'home</w>', 'a', '.</w>', '<|eos|>', '<|pad|>'
        ] )


        tokens = [
            "a", "b", "gone</w>",
            "a", "b", ".</w>",
            "gone</w>", "home</w>", "a", ".</w>" #7
            ]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        chunked = embeddings_provider._chunk_and_pad_token_ids(token_ids=token_ids, token_weights=[1] * len(token_ids), device='cpu')
        chunked_tokens = tokenizer.convert_ids_to_tokens(chunked[0].tolist())
        #print('PHRASES 2:', chunked_tokens)
        self.assertTrue(len(chunked_tokens) == 21)
        self.assertEqual(chunked_tokens,[
            '<|bos|>', 'a', 'b', 'gone</w>', '<|eos|>', '<|pad|>', '<|pad|>',
            '<|bos|>', 'a', 'b', '.</w>', '<|eos|>', '<|pad|>', '<|pad|>',
            '<|bos|>', 'gone</w>', 'home</w>', 'a', '.</w>', '<|eos|>', '<|pad|>'
        ])


    def test_tokenize_to_mask(self):
        tokenizer = DummyTokenizer(model_max_length=7)
        text_encoder = DummyTransformer()
        embeddings_provider = EmbeddingsProvider(tokenizer=tokenizer, text_encoder=text_encoder,
                                                 padding_attention_mask_value=0)

        fragments = ['a b']
        _, _, mask = embeddings_provider.get_token_ids_and_expand_weights(fragments, weights=[1] * len(fragments),
                                                                          device='cpu')
        self.assertSequenceEqual(mask.tolist(), [1, 1, 1, 1, 0, 0, 0])

        fragments = ['a b c a b c a']
        _, _, mask = embeddings_provider.get_token_ids_and_expand_weights(fragments, weights=[1] * len(fragments),
                                                                          device='cpu')
        self.assertSequenceEqual(mask.tolist(), [1, 1, 1, 1, 1, 1, 1])

        fragments = ['a', 'b c']
        _, _, mask = embeddings_provider.get_token_ids_and_expand_weights(fragments, weights=[1, 2], device='cpu')
        self.assertSequenceEqual(mask.tolist(), [1, 1, 1, 1, 1, 0, 0])

        # eos/bos only
        fragments = []
        _, _, mask = embeddings_provider.get_token_ids_and_expand_weights(fragments, weights=[1] * len(fragments),
                                                                          device='cpu')
        self.assertSequenceEqual(mask.tolist(), [1, 1, 0, 0, 0, 0, 0])

        # too long
        fragments = ['a b c a b c a b c a b c']
        _, _, mask = embeddings_provider.get_token_ids_and_expand_weights(fragments, weights=[1] * len(fragments),
                                                                          device='cpu')
        self.assertSequenceEqual(mask.tolist(), [1, 1, 1, 1, 1, 1, 1])

    def test_get_token_ids(self):
        ep = make_dummy_embeddings_provider()

        prompts = [" ".join(KNOWN_WORDS), " ".join(reversed(KNOWN_WORDS))]
        expected_token_ids = [KNOWN_WORDS_TOKEN_IDS, list(reversed(KNOWN_WORDS_TOKEN_IDS))]
        token_ids = ep.get_token_ids(prompts, include_start_and_end_markers=False, truncation_override=False)
        self.assertEqual(expected_token_ids, token_ids)

        expected_token_ids_with_bos_eos = [[ep.tokenizer.bos_token_id] + x + [ep.tokenizer.eos_token_id] for x in expected_token_ids]
        #print(expected_token_ids_with_bos_eos)
        token_ids = ep.get_token_ids(prompts, include_start_and_end_markers=True, truncation_override=False)
        self.assertEqual(expected_token_ids_with_bos_eos, token_ids)


    def test_build_weighted_embeddings_tensor(self):
        max_length = 10
        ep = make_dummy_embeddings_provider(max_length=max_length)

        # check that if fails if too short
        words_token_ids = KNOWN_WORDS_TOKEN_IDS[:3]
        token_ids = torch.tensor(words_token_ids)
        per_token_weights = torch.tensor([1] * len(words_token_ids))
        with self.assertRaises(ValueError):
            ep.build_weighted_embedding_tensor(token_ids, per_token_weights)

        # all weighted
        empty_z = ep.build_weighted_embedding_tensor(torch.tensor([ep.tokenizer.bos_token_id] +
                                                                  [ep.tokenizer.eos_token_id] +
                                                                  [ep.tokenizer.pad_token_id] * (max_length - 2)),
                                                     torch.tensor([1] * max_length))
        token_ids = torch.tensor([ep.tokenizer.bos_token_id] + words_token_ids +
                                 [ep.tokenizer.eos_token_id] +
                                 [ep.tokenizer.pad_token_id] * (max_length - 2 - len(words_token_ids)))
        unweighted_embeddings = ep.build_weighted_embedding_tensor(token_ids, torch.tensor([1] * len(token_ids)))

        # confirm that the weighting works as expected (delta from empty)
        weight = 2.0
        weighted_embeddings_2 = ep.build_weighted_embedding_tensor(token_ids, torch.tensor([weight] * len(token_ids)))
        self.assertTrue(torch.allclose(empty_z + (unweighted_embeddings-empty_z) * weight,
                                       weighted_embeddings_2,
                                       atol=1e-5))

        # different weights at different places
        rand_weights = 1 + torch.rand([len(token_ids)])
        weighted_embeddings_rand = ep.build_weighted_embedding_tensor(token_ids, rand_weights)
        weights_expanded = rand_weights.unsqueeze(1).expand(-1,768).unsqueeze(0)
        self.assertTrue(torch.allclose(empty_z + (unweighted_embeddings-empty_z) * weights_expanded,
                                       weighted_embeddings_rand,
                                       atol=1e-7))

        # mask
        mask = torch.Tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        weighted_embeddings_masked = ep.build_weighted_embedding_tensor(token_ids, torch.tensor([1] * len(token_ids)),
                                                                        attention_mask=mask)
        # note: test framework's masking is not expected to match the actual Text Encoder's - it just does a brainless multiply
        self.assertTrue(torch.allclose(unweighted_embeddings * mask.unsqueeze(1).expand(-1,768).unsqueeze(0),
                                       weighted_embeddings_masked,
                                       atol=1e-7))

        mask = torch.Tensor([0, 1, 1, 1, 0, 0, 1, 0, 1, 1])
        weighted_embeddings_masked = ep.build_weighted_embedding_tensor(token_ids, torch.tensor([1] * len(token_ids)),
                                                                        attention_mask=mask)
        # note: test framework's masking is not expected to match the actual Text Encoder's - it just does a brainless multiply
        self.assertTrue(torch.allclose(unweighted_embeddings * mask.unsqueeze(1).expand(-1,768).unsqueeze(0),
                                       weighted_embeddings_masked,
                                       atol=1e-7))



    def test_upweighting_prompt_fragments(self):
        max_length = 10
        ep = make_dummy_embeddings_provider(max_length=max_length, padding_attention_mask_value=0)

        words = KNOWN_WORDS[:3]
        words_token_ids = KNOWN_WORDS_TOKEN_IDS[:3]
        text_batch = [[' '.join(words)]]
        fragment_weights_batch = [[1]]
        embeddings = ep.get_embeddings_for_weighted_prompt_fragments(text_batch, fragment_weights_batch)

        pad_length = (max_length-2-len(words_token_ids))
        expected_token_ids = torch.tensor([ep.tokenizer.bos_token_id] + words_token_ids + [ep.tokenizer.eos_token_id] +
                                          [ep.tokenizer.pad_token_id] * pad_length)
        expected_mask = torch.tensor([1] + [1] * len(words_token_ids) + [1] + [0] * pad_length)
        expected_embeddings = ep.build_weighted_embedding_tensor(expected_token_ids, torch.tensor([1] * len(expected_token_ids)), expected_mask)
        self.assertTrue(torch.allclose(expected_embeddings, embeddings, atol=1e-8))

        # weighted fragments
        text_batch = [[words[0], ' '.join(words[1:3])]]
        fragment_weights_batch = [[1, 2]]
        embeddings = ep.get_embeddings_for_weighted_prompt_fragments(text_batch, fragment_weights_batch)
        expected_weights = [1] + [1] + [2, 2] + [1] * 6
        expected_embeddings = ep.build_weighted_embedding_tensor(expected_token_ids, torch.tensor(expected_weights), expected_mask)
        self.assertTrue(torch.allclose(expected_embeddings, embeddings, atol=1e-8))


    def test_downweighting_prompt_fragments_mask(self):
        max_length = 10
        ep = make_dummy_embeddings_provider(max_length=max_length, padding_attention_mask_value=0, downweight_mode=DownweightMode.MASK)

        # downweighting
        words = KNOWN_WORDS[:3]
        words_token_ids = KNOWN_WORDS_TOKEN_IDS[:3]
        text_batch = [[words[0], words[1]]]
        downweighted_fragment_weight = 0.5
        fragment_weights_batch = [[1, downweighted_fragment_weight]]
        embeddings = ep.get_embeddings_for_weighted_prompt_fragments(text_batch, fragment_weights_batch)
        expected_token_ids = torch.tensor([ep.tokenizer.bos_token_id] + words_token_ids[0:2] +
                             [ep.tokenizer.eos_token_id] +
                             [ep.tokenizer.pad_token_id] * (max_length-4))
        expected_weights = [1] + [1, downweighted_fragment_weight] + [1] * 7
        # when downweighting, additionally blend against a version of the prompt with the downweighted term masked out
        unweighted_mask = torch.tensor([1, 1, 1, 1] + [0] * 6)
        downweighted_fragment_dropper_mask = torch.tensor([1, 1, 0, 1] + [0] * 6)

        expected_embeddings_main_part = ep.build_weighted_embedding_tensor(expected_token_ids,
                                                                           torch.tensor(expected_weights),
                                                                           attention_mask=unweighted_mask)
        expected_embeddings_downweighted_dropped = ep.build_weighted_embedding_tensor(expected_token_ids,
                                                                                      torch.tensor(expected_weights),
                                                                                      attention_mask=downweighted_fragment_dropper_mask)
        # use tan, like in EmbeddingsProvider.get_embeddings_for_weighted_prompt_fragments()
        downweighted_lerp_weight = math.tan((1.0 - downweighted_fragment_weight) * math.pi / 2)
        blend_weights = [1.0, downweighted_lerp_weight]

        expected_embeddings = EmbeddingsProvider.apply_embedding_weights(torch.cat([expected_embeddings_main_part,
                                                                                    expected_embeddings_downweighted_dropped]).unsqueeze(0),
                                                                         per_embedding_weights=blend_weights,
                                                                         normalize=True)
        self.assertTrue(torch.allclose(embeddings, expected_embeddings, atol=1e-8))

    def test_downweighting_prompt_fragments_remove(self):
        max_length = 10
        ep = make_dummy_embeddings_provider(max_length=max_length, downweight_mode=DownweightMode.REMOVE)
        # downweighting
        words = KNOWN_WORDS[:3]
        words_token_ids = KNOWN_WORDS_TOKEN_IDS[:3]
        text_batch = [[words[0], words[1]]]
        downweighted_fragment_weight = 0.5
        fragment_weights_batch = [[1, downweighted_fragment_weight]]
        embeddings = ep.get_embeddings_for_weighted_prompt_fragments(text_batch, fragment_weights_batch)
        expected_token_ids = torch.tensor([ep.tokenizer.bos_token_id] + words_token_ids[0:2] +
                             [ep.tokenizer.eos_token_id] +
                             [ep.tokenizer.pad_token_id] * (max_length-4))
        expected_weights = [1] + [1, downweighted_fragment_weight] + [1] * 7
        # when downweighting, additionally blend against a version of the prompt without the downweighted term
        expected_token_ids_cut = torch.tensor([ep.tokenizer.bos_token_id] + words_token_ids[0:1] +
                             [ep.tokenizer.eos_token_id] +
                             [ep.tokenizer.pad_token_id] * (max_length-3))
        expected_weights_cut = [1] + [1] + [1] * 8
        expected_embeddings_main_part = ep.build_weighted_embedding_tensor(expected_token_ids, torch.tensor(expected_weights))
        expected_embeddings_cut = ep.build_weighted_embedding_tensor(expected_token_ids_cut, torch.tensor(expected_weights_cut))

        downweighted_lerp_weight = math.tan((1.0 - downweighted_fragment_weight) * math.pi / 2)
        blend_weights = [1.0, downweighted_lerp_weight]

        expected_embeddings = EmbeddingsProvider.apply_embedding_weights(torch.cat([expected_embeddings_main_part,
                                                                                    expected_embeddings_cut]).unsqueeze(0),
                                                                         per_embedding_weights=blend_weights,
                                                                         normalize=True)
        self.assertTrue(torch.allclose(expected_embeddings, embeddings, atol=1e-8))


    def test_too_long_weighted_prompt_fragments_truncate(self):
        max_length = 10
        ep = make_dummy_embeddings_provider(max_length=max_length, truncate=True, padding_attention_mask_value=0)

        words = KNOWN_WORDS[:3]
        words_token_ids = KNOWN_WORDS_TOKEN_IDS[:3]

        # too many weighted fragments
        text_batch = [[words[0], ' '.join(reversed(words*3)), ' '.join(words[1:3], )]]
        fragment_weights_batch = [[1, 2, 3]]
        embeddings = ep.get_embeddings_for_weighted_prompt_fragments(text_batch, fragment_weights_batch)

        expected_token_ids = torch.tensor([ep.tokenizer.bos_token_id] +
                                          ([words_token_ids[0]] +
                                          list(reversed(words_token_ids*3)) +
                                          [words_token_ids[1]])[0:8] +
                                          [ep.tokenizer.eos_token_id])
        expected_weights = [1] + [1] + [2, 2, 2, 2, 2, 2, 2] + [1]
        expected_embeddings = ep.build_weighted_embedding_tensor(expected_token_ids, torch.tensor(expected_weights))
        self.assertTrue(torch.allclose(expected_embeddings, embeddings, atol=1e-8))


    def test_too_long_weighted_prompt_fragments_notruncate(self):
        max_length = 10
        ep = make_dummy_embeddings_provider(max_length=max_length, truncate=False, padding_attention_mask_value=0)

        words = KNOWN_WORDS[:3]
        words_token_ids = KNOWN_WORDS_TOKEN_IDS[:3]

        # too many weighted fragments
        text_batch = [[words[0], ' '.join(reversed(words*3)), ' '.join(words[1:3], )]]
        fragment_weights_batch = [[1, 2, 3]]
        embeddings = ep.get_embeddings_for_weighted_prompt_fragments(text_batch, fragment_weights_batch)

        expected_token_ids_part1 = ([ep.tokenizer.bos_token_id] +
                                          [words_token_ids[0]] +
                                          list(reversed(words_token_ids*3))[0:7] +
                                          [ep.tokenizer.eos_token_id])
        expected_token_ids_part2 = ([ep.tokenizer.bos_token_id] +
                                          list(reversed(words_token_ids*3))[7:9] +
                                          words_token_ids[1:3] +
                                          [ep.tokenizer.eos_token_id] +
                                          ([ep.tokenizer.pad_token_id] * 4))
        expected_token_ids = torch.tensor(expected_token_ids_part1 + expected_token_ids_part2)
        expected_weights = ([1] + [1] + [2, 2, 2, 2, 2, 2, 2] + [1] +
                            [1] + [2, 2] + [3, 3] + [1] + ([1] * 4))
        expected_mask = torch.Tensor([1] * 16 + [0] * 4)
        expected_embeddings = ep.build_weighted_embedding_tensor(expected_token_ids, torch.tensor(expected_weights), attention_mask=expected_mask)
        self.assertTrue(torch.allclose(expected_embeddings, embeddings, atol=1e-8))






if __name__ == '__main__':
    unittest.main()
