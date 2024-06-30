import functools

import math
from abc import ABC
from enum import Enum
from typing import Callable, Union, Optional, Any

import torch
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from typing import List, Tuple

__all__ = ["EmbeddingsProvider", "DownweightMode", "ReturnedEmbeddingsType"]


class DownweightMode(Enum):
    REMOVE = 0  # Remove downweighted tokens from the token sequence (shifts all subsequent tokens)
    MASK = 1   # Default: Leave tokens in-place but mask them out using attention masking

class BaseTextualInversionManager(ABC):
    def expand_textual_inversion_token_ids_if_necessary(self, token_ids: List[int]) -> List[int]:
        raise NotImplementedError()

class ReturnedEmbeddingsType(Enum):
    LAST_HIDDEN_STATES_NORMALIZED = 0             # SD1/2 regular
    PENULTIMATE_HIDDEN_STATES_NORMALIZED = 1      # SD1.5 with "clip skip"
    PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED = 2  # SDXL


class EmbeddingsProvider:

    def __init__(self,
                 tokenizer: CLIPTokenizer,
                 text_encoder: Union[CLIPTextModel, CLIPTextModelWithProjection], # convert a list of int token ids to a tensor of embeddings
                 textual_inversion_manager: BaseTextualInversionManager = None,
                 dtype_for_device_getter: Callable[[torch.device], torch.dtype] = lambda device: torch.float32,
                 truncate: bool = True,
                 padding_attention_mask_value: int = 1,
                 downweight_mode: DownweightMode = DownweightMode.MASK,
                 returned_embeddings_type: ReturnedEmbeddingsType = ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
                 device: Optional[str] = None
                 ):
        """
        `tokenizer`: converts strings to lists of int token ids
        `text_encoder`: convert lists of token ids to embedding tensors
        `textual_inversion_manager`: manage token insertion for textual inversions with vector length >1
        `dtype_for_device_getter`: callback that returns an appropriate dtype for the requested device. if unset, defaults to torch.float32.
        `truncate`: if True, truncate inputs to the maximum length specified by the tokenizer. if False, returns
                    tensors that may be longer than the maximum length (but will always be an integer multiple of maximum length)
        `padding_attention_mask_value`: Value to write into the attention mask for padding tokens. Stable Diffusion needs 1.
        `downweight_mode`: if MASK, downweight by blending with a version of the prompt with the downweighted terms masked out.
                    if REMOVE, the blend is against a version of the prompt with the downweighted tokens removed
        `returned_embeddings_type`: controls how the embedding vectors are taken from the result of running the text
            encoder over the parsed prompt's text. For SD<=2.1, use LAST_HIDDEN_STATES_NORMALIZED, or
            PENULTIMATE_HIDDEN_STATES_NORMALIZED if you want to do "clip skip". For SDXL use PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED.
        """
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.textual_inversion_manager = textual_inversion_manager
        self.truncate_to_model_max_length = truncate
        self.padding_attention_mask_value = padding_attention_mask_value
        self.downweight_mode = downweight_mode
        self.returned_embeddings_type = returned_embeddings_type
        self.device = device if device else self.text_encoder.device
        self._emptystring_conditioning = None

        # by default always use float32
        self.get_dtype_for_device = dtype_for_device_getter

    @property
    def emptystring_conditioning(self):
        if self._emptystring_conditioning is None:
            emptystring_token_ids = torch.tensor(self.get_token_ids([""], padding="max_length"),
                                                 dtype=torch.long,
                                                 device=self.device)
            self._emptystring_conditioning = self._encode_token_ids_to_embeddings(emptystring_token_ids)
        return self._emptystring_conditioning

    @property
    def max_token_count(self) -> int:
        return self.tokenizer.model_max_length

    @property
    def needs_bos(self) -> bool:
        return self.tokenizer.bos_token_id is not None

    @property
    def needs_eos(self) -> bool:
        return self.tokenizer.eos_token_id is not None

    @classmethod
    def apply_embedding_weights(cls, embeddings: torch.Tensor, per_embedding_weights: List[float],
                                normalize: bool) -> torch.Tensor:
        if len(embeddings.shape) != 3:
            raise ValueError("embeddings has the wrong shape - must be [B, tokens, embedding dim]")
        per_embedding_weights = torch.tensor(per_embedding_weights, dtype=embeddings.dtype, device=embeddings.device)
        if len(per_embedding_weights.shape) != 1:
            raise ValueError("per_embedding_weights must be a 1d vector")
        if embeddings.shape[0] != len(per_embedding_weights):
            raise ValueError(f"wrong number of weights - should have {embeddings.shape[0]} weights for embeddings tensor of shape {embeddings.shape}")
        if normalize:
            per_embedding_weights = per_embedding_weights / torch.sum(per_embedding_weights)

        reshaped_weights = per_embedding_weights.reshape(per_embedding_weights.shape + (1, 1,))
        blended_embeddings = torch.sum(embeddings * reshaped_weights, dim=0)
        # blended_embeddings now has shape (77, 768)
        return blended_embeddings


    def get_embeddings_for_weighted_prompt_fragments(self,
                                                     text_batch: List[List[str]],
                                                     fragment_weights_batch: List[List[float]],
                                                     should_return_tokens: bool = False,
                                                     device='cpu'
                                 ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """

        :param text_batch: A list of fragments of text to which different weights are to be applied.
        :param fragment_weights_batch: A list of weights, one for each entry in `fragments`.
        :param should_return_tokens: If True, return a tuple of (embeddings, tokens), otherwise just return embeddings.
        :param device: Where to put the constructed tensor(s)
        :return: A tensor of shape `[B, 77, token_dim]` containing weighted embeddings where token_dim depends on the underlying text encoder
        """
        if len(text_batch) != len(fragment_weights_batch):
            raise ValueError(
                f"lengths of text and fragment_weights lists are not the same "+
                f"({len(text_batch)} != {len(fragment_weights_batch)})")

        token_counts_batch = [[len(self.tokenizer.tokenize(t)) for t in b] for b in text_batch]
        # one weight entry per token
        weights_flattened_batch = []
        # one fragment id per token. fragment ids are non-unique, counting up from zero for each entry in the batch.
        fragment_ids_batch = []
        for weights, token_counts in zip(fragment_weights_batch, token_counts_batch):
            weights_expanded = [[w] * c for w, c in zip(weights, token_counts)]
            fragment_ids_expanded = [[fid] * c for fid, c in zip(range(len(token_counts)), token_counts)]
            weights_flattened = functools.reduce(lambda a, b: a+b, weights_expanded)
            fragment_ids_flattened = functools.reduce(lambda a, b: a+b, fragment_ids_expanded)
            weights_flattened_batch.append(weights_flattened)
            fragment_ids_batch.append(fragment_ids_flattened)

        token_ids_batch = self.get_token_ids([' '.join(text) for text in text_batch],
                                             include_start_and_end_markers=False,
                                             padding="do_not_pad",
                                             truncation_override=False)

        chunked_data_batch = self._chunk_tokens_and_cap(token_ids_batch, weights_flattened_batch, fragment_ids_batch)

        # encode and weight
        embeddings_batch = [] # apply_weights(chunked_data_batch)
        for batch_index, chunked_data in enumerate(chunked_data_batch):
            token_ids_list, weights_list, fragment_ids_list = zip(*chunked_data)

            token_ids = torch.tensor(token_ids_list, dtype=torch.long, device=device)
            weights = torch.tensor(weights_list, dtype=torch.float, device=device)
            fragment_ids = torch.tensor(fragment_ids_list, dtype=torch.long, device=device)

            # First, weight tokens in individual fragments by scaling the feature vectors as requested (effectively
            # applying a multiplier to the CFG scale on a per-token basis).
            # For tokens weighted<1, intuitively we want SD to become not merely *less* interested in the concept
            # captured by the fragment but actually *dis*interested in it (a 0.01 interest in "red" is still an active
            # interest, however small, in redness; what the user probably intends when they attach the number 0.01 to
            # "red" is to tell SD that it should almost completely *ignore* redness).
            # To do this, the embedding is lerped away from base_embedding in the direction of an embedding for a prompt
            # string from which the low-weighted fragment has been simply removed. The closer the weight is to zero, the
            # closer the resulting embedding is to an embedding for a prompt that simply lacks this fragment.

            # upweight
            base_embeddings = self._encode_token_ids_to_embeddings(token_ids) * weights.unsqueeze(dim=-1).squeeze(0)

            # downweight
            downweighted_fragment_ids_chunked = [fragment_ids[i].index_select(
                dim=0, index=torch.nonzero(weights[i]<1).squeeze()
            ).unique().tolist() for i in range(fragment_ids.shape[0])]
            chunk_embeddings = []
            for chunk_index, downweighted_fragment_ids in enumerate(downweighted_fragment_ids_chunked):
                if len(downweighted_fragment_ids) == 0:
                    chunk_embeddings.append(base_embeddings[chunk_index])
                else:

                    lerp_contributions = []
                    lerp_weights = []

                    lerp_contributions.append(base_embeddings[chunk_index].unsqueeze(0))
                    lerp_weights.append(1.0)

                    for fid in downweighted_fragment_ids:
                        if self.downweight_mode != DownweightMode.MASK:
                            raise ValueError("As of compel>=2.1, only DownweightMode.MASK is supported")
                        attention_mask = torch.where(fragment_ids[chunk_index]==fid, 0, 1)
                        embeddings_without_this = self._encode_token_ids_to_embeddings(
                            token_ids[chunk_index].unsqueeze(0), attention_mask=attention_mask.unsqueeze(0)
                        )
                        # weight of the embedding *without* this fragment gets *stronger* as its weight approaches 0
                        # if fragment_weight = 0, basically we want embedding_without_this to completely overwhelm base_embedding
                        # therefore:
                        # fragment_weight = 1: we are at base_embeddings => lerp weight 0
                        # fragment_weight = 0.5: we are halfway between base_embeddings and here => lerp weight 1
                        # fragment_weight = 0: we're now entirely overriding base_embeddings ==> lerp weight inf
                        # so let's use tan(), because:
                        # tan is 0.0 at 0,
                        #        1.0 at PI/4, and
                        #        inf at PI/2
                        # -> tan((1-weight)*PI/2) should give us ideal lerp weights
                        epsilon = 1e-5
                        fragment_weight = max(epsilon, fragment_weights_batch[batch_index][fid])  # inf is bad
                        embedding_lerp_weight = math.tan((1.0 - fragment_weight) * math.pi / 2)

                        lerp_contributions.append(embeddings_without_this)
                        lerp_weights.append(embedding_lerp_weight)

                    # apply the lerp weights
                    lerped_embeddings = self.apply_embedding_weights(torch.concat(lerp_contributions), lerp_weights, normalize=True).squeeze(0)
                    chunk_embeddings.append(lerped_embeddings)

            # flatten chunks
            embeddings_batch.append(torch.concat(chunk_embeddings).unsqueeze(0))

        embeddings_batch = self._pad_conditioning_tensors_to_same_length(
            embeddings_batch, self.emptystring_conditioning
        )
        if should_return_tokens:
            return torch.concat(embeddings_batch), torch.tensor(token_ids_batch)
        else:
            return torch.concat(embeddings_batch)


    def _chunk_tokens_and_cap(self,
                              token_ids_batch: list[list[int]],
                              weights_flattened_batch: list[list[float]],
                              fragment_ids_batch: list[list[int]]
                              ) -> list[list[list]]:
        """
        Breaks token_ids_batch into chunks of length < self.max_model_length, where each chunk starts with bos, ends
         with eos, and is padded to self.max_model_length if necessary. Also splits weights_flattened_batch and
         fragment_ids_batch along the same indices.

        Returns a list of (chunk_token_ids:list, chunk_weights:list, chunk_fragment_ids:list) tuples.
        """
        if len(token_ids_batch) != len(weights_flattened_batch) or len(token_ids_batch) != len(fragment_ids_batch):
            raise ValueError(
                f'all inputs must have the same length (got {len(token_ids_batch)}, {len(weights_flattened_batch)}, ' +
                f'{len(fragment_ids_batch)})')

        chunked_data_batch = []
        for data in [list(zip(*x)) for x in zip(token_ids_batch, weights_flattened_batch, fragment_ids_batch)]:
            # data is a list of tuples (token_id, weight, fragment_id)
            chunked_data = []
            required_cap_count = (1 if self.needs_bos else 0) + (1 if self.needs_eos else 0)

            remaining_data = data
            while len(remaining_data) > 0:
                # todo: pick a better split point that self.max_token_count
                max_content_token_count = self.max_token_count - required_cap_count
                if len(remaining_data) <= max_content_token_count:
                    chunked_data.append(remaining_data)
                    break

                split_index = self.max_token_count - required_cap_count
                left = remaining_data[:split_index]
                chunked_data.append(left)
                if self.truncate_to_model_max_length:
                    # discard remainder
                    remaining_data = []
                else:
                    right = remaining_data[split_index:]
                    remaining_data = right

            # empty string tokenization will trigger this condition
            if len(chunked_data)==0:
                chunked_data.append([])

            # token id, weight, fragment id
            chunked_data_capped = [self.add_bos_eos_if_required(d,
                                                                bos_value=(self.tokenizer.bos_token_id, 1, -1),
                                                                eos_value=(self.tokenizer.eos_token_id, 1, -1)
                                                                )
                                   for d in chunked_data]

            if self.tokenizer.pad_token_id is not None:
                # pad last split to required length
                required_pad_token_count = self.max_token_count - len(chunked_data_capped[-1])
                chunked_data_capped[-1].extend([(self.tokenizer.pad_token_id, 1, -1)] * required_pad_token_count)

            # convert from list of lists of (token_id, weight, fragment_id) tuples
            # to list of (chunk_token_ids, chunk_weights, chunk_fragment_id) tuples
            chunked_data_flattened = [list(zip(*d)) for d in chunked_data_capped]
            chunked_data_batch.append(chunked_data_flattened)

        return chunked_data_batch

    def get_embeddings_for_weighted_prompt_fragments_old(self,
                                                 text_batch: List[List[str]],
                                                 fragment_weights_batch: List[List[float]],
                                                 should_return_tokens: bool = False,
                                                 device='cpu'
                             ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        for batch_index, text in enumerate(text_batch):
            token_ids = self.get_token_ids(' '.join(text), include_start_and_end_markers=False,
                                           padding="do_not_pad",
                                           truncation_override=False)

        batch_z = None
        batch_tokens = None
        for fragments, weights in zip(text_batch, fragment_weights_batch):

            # First, weight tokens in individual fragments by scaling the feature vectors as requested (effectively
            # applying a multiplier to the CFG scale on a per-token basis).
            # For tokens weighted<1, intuitively we want SD to become not merely *less* interested in the concept
            # captured by the fragment but actually *dis*interested in it (a 0.01 interest in "red" is still an active
            # interest, however small, in redness; what the user probably intends when they attach the number 0.01 to
            # "red" is to tell SD that it should almost completely *ignore* redness).
            # To do this, the embedding is lerped away from base_embedding in the direction of an embedding for a prompt
            # string from which the low-weighted fragment has been simply removed. The closer the weight is to zero, the
            # closer the resulting embedding is to an embedding for a prompt that simply lacks this fragment.

            # handle weights >=1
            tokens, per_token_weights, mask = self.get_token_ids_and_expand_weights(fragments, weights, device=device)
            base_embedding = self.build_weighted_embedding_tensor(tokens, per_token_weights, mask, device=device)

            # this is our starting point
            embeddings = base_embedding.unsqueeze(0)
            per_embedding_weights = [1.0]

            # now handle weights <1
            # Do this by building extra embeddings tensors that lack the words being <1 weighted. These will be lerped
            # with the embeddings tensors that have the words, such that if the weight of a word is 0.5, the resulting
            # embedding will be exactly half-way between the unweighted prompt and the prompt with the <1 weighted words
            # removed.
            # e.g. for "mountain:1 man:0.5", intuitively the "man" should be "half-gone". therefore, append an embedding
            # for "mountain" (i.e. without "man") to the already-produced embedding for "mountain man", and weight it
            # such that the resulting lerped embedding is exactly half-way between "mountain man" and "mountain".
            fragment_token_index_ranges = self._get_token_ranges_for_fragments(tokens.tolist(), fragments)

            for index in range(len(fragment_token_index_ranges)):
                fragment_weight = weights[index]
                if fragment_weight < 1:
                    if self.downweight_mode == DownweightMode.MASK:
                        fragment_start_token_id, fragment_end_token_id = fragment_token_index_ranges[index]
                        # mask out this fragment
                        mask_without_fragment = mask.clone()
                        mask_without_fragment[fragment_start_token_id:fragment_end_token_id+1] = 0
                        if not self.truncate_to_model_max_length:
                            # but don't mask chunk-delimiting eos/bos markers
                            mask_without_fragment[0::self.tokenizer.model_max_length] = 1
                            mask_without_fragment[self.tokenizer.model_max_length-1::self.tokenizer.model_max_length] = 1
                        embedding_without_this = self.build_weighted_embedding_tensor(tokens,
                                                                                      per_token_weights,
                                                                                      mask_without_fragment,
                                                                                      device=device)
                    else:
                        fragments_without_this = fragments[0:index] + fragments[index+1:]
                        weights_without_this = weights[0:index] + weights[index+1:]
                        tokens_without_fragment, per_token_weights_without_fragment, mask_without_fragment = \
                            self.get_token_ids_and_expand_weights(fragments_without_this, weights_without_this, device=device)
                        embedding_without_this = self.build_weighted_embedding_tensor(tokens_without_fragment,
                                                                                      per_token_weights_without_fragment,
                                                                                      device=device)

                    embeddings = torch.cat((embeddings, embedding_without_this.unsqueeze(0)), dim=1)
                    # weight of the embedding *without* this fragment gets *stronger* as its weight approaches 0
                    # if fragment_weight = 0, basically we want embedding_without_this to completely overwhelm base_embedding
                    # therefore:
                    # fragment_weight = 1: we are at base_z => lerp weight 0
                    # fragment_weight = 0.5: we are halfway between base_z and here => lerp weight 1
                    # fragment_weight = 0: we're now entirely overriding base_z ==> lerp weight inf
                    # so let's use tan(), because:
                    # tan is 0.0 at 0,
                    #        1.0 at PI/4, and
                    #        inf at PI/2
                    # -> tan((1-weight)*PI/2) should give us ideal lerp weights
                    epsilon = 1e-5
                    fragment_weight = max(epsilon, fragment_weight) # inf is bad
                    embedding_lerp_weight = math.tan((1.0 - fragment_weight) * math.pi / 2)

                    per_embedding_weights.append(embedding_lerp_weight)

            lerped_embeddings = self.apply_embedding_weights(embeddings, per_embedding_weights, normalize=True).squeeze(0)

            #print(f"assembled tokens for '{fragments}' into tensor of shape {lerped_embeddings.shape}")

            # append to batch
            batch_z = lerped_embeddings.unsqueeze(0) if batch_z is None else torch.cat([batch_z, lerped_embeddings.unsqueeze(0)], dim=1)
            batch_tokens = tokens.unsqueeze(0) if batch_tokens is None else torch.cat([batch_tokens, tokens.unsqueeze(0)], dim=1)

        # should have shape (B, 77, 768)
        #print(f"assembled all tokens into tensor of shape {batch_z.shape}")

        if should_return_tokens:
            return batch_z, batch_tokens
        else:
            return batch_z

    def get_token_ids(self, texts: List[str], include_start_and_end_markers: bool = True, padding: str = 'do_not_pad',
                      truncation_override: Optional[bool] = None) -> List[List[int]]:
        """
        Convert a list of strings like `["a cat", "a dog", "monkey riding a bicycle"]` into a list of lists of token
        ids like `[[bos, 0, 1, eos], [bos, 0, 2, eos], [bos, 3, 4, 0, 5, eos]]`. bos/eos markers are skipped if
        `include_start_and_end_markers` is `False`. Each list will be restricted to the maximum permitted length
        (typically 75 tokens + eos/bos markers).

        :param texts: The strings to convert.
        :param include_start_and_end_markers: If True (default), returned token id lists will start with the beginning
            of sequence marker and end with the end-of-sequence marker (`eos`).
        :padding: Padding argument passed through to the Tokenizer.
        :truncation_override: Optional, overrides the `truncate` argument passed to `__init__`.
        :return: A list of lists of token ids corresponding to the input strings.
        """
        # for args documentation of self.tokenizer() see ENCODE_KWARGS_DOCSTRING in tokenization_utils_base.py
        # (part of `transformers` lib)
        truncation = self.truncate_to_model_max_length if truncation_override is None else truncation_override
        token_ids_list = self.tokenizer(
            texts,
            truncation=truncation,
            padding=padding,
            return_tensors=None,  # just give me lists of ints
        )['input_ids']

        result = []
        for token_ids in token_ids_list:

            # trim eos/bos
            token_ids = self.remove_bos_eos(token_ids)
            # pad for textual inversions with vector length >1
            if self.textual_inversion_manager is not None:
                token_ids = self.textual_inversion_manager.expand_textual_inversion_token_ids_if_necessary(token_ids)

            # add back eos/bos if requested
            if include_start_and_end_markers:
                token_ids = self.add_bos_eos_if_required(token_ids)

            result.append(token_ids)

        return result

    def get_pooled_embeddings(self, texts: List[str], attention_mask: Optional[torch.Tensor]=None, device: Optional[str]=None) -> Optional[torch.Tensor]:
        
        device = device or self.device

        token_ids = self.get_token_ids(texts, padding="max_length", truncation_override=True)
        token_ids = torch.tensor(token_ids, dtype=torch.long).to(device)

        text_encoder_output = self.text_encoder(token_ids, attention_mask, return_dict=True)
        pooled = text_encoder_output.text_embeds

        return pooled


    def get_token_ids_and_expand_weights(self, fragments: List[str], weights: List[float], device: str
                                         ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        '''
        Given a list of text fragments and corresponding weights: tokenize each fragment, append the token sequences
        together and return a padded token sequence starting with the bos marker, ending with the eos marker, and padded
        or truncated as appropriate to `self.max_length`. Also return a list of weights expanded from the passed-in
        weights to match each token.

        :param fragments: Text fragments to tokenize and concatenate. May be empty.
        :param weights: Per-fragment weights (i.e. quasi-CFG scaling). Values from 0 to inf are permitted. In practise with SD1.5
                        values >1.6 tend to produce garbage output. Must have same length as `fragment`.
        :return: A tuple of tensors `(token_ids, weights, mask)`. `token_ids` is ints, `weights` is floats, `mask` is
                        ints, all have shape `[self.max_length]`.
        '''
        if len(fragments) != len(weights):
            raise ValueError(f"lengths of text and fragment_weights lists are not the same ({len(fragments)} != {len(weights)})")

        # empty is meaningful
        if len(fragments) == 0:
            fragments = ['']
            weights = [1.0]
        per_fragment_token_ids = self.get_token_ids(fragments, include_start_and_end_markers=False)
        all_token_ids: List[int] = []
        all_token_weights: List[float] = []
        # print("all fragments:", fragments, weights)
        for this_fragment_token_ids, weight in zip(per_fragment_token_ids, weights):
            # append
            all_token_ids += this_fragment_token_ids
            # fill out weights tensor with one float per token
            all_token_weights += [float(weight)] * len(this_fragment_token_ids)

        return self._chunk_and_pad_token_ids(all_token_ids, all_token_weights, device=device)

    def _chunk_and_pad_token_ids(self, token_ids: List[int], token_weights: List[float], device: str
                                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        remaining_token_ids = token_ids
        remaining_token_weights = token_weights
        chunk_length_without_eos_bos_markers = self.max_token_count - 2

        all_token_ids = []
        all_token_weights = []
        all_masks = []
        while True:
            # each chunk must leave room for bos/eos
            chunk_token_ids = remaining_token_ids[0:chunk_length_without_eos_bos_markers]
            chunk_token_weights = remaining_token_weights[0:chunk_length_without_eos_bos_markers]
            # update remaining
            remaining_token_ids = remaining_token_ids[chunk_length_without_eos_bos_markers:]
            remaining_token_weights = remaining_token_weights[chunk_length_without_eos_bos_markers:]

            # pad out to a self.max_length-entry array: [eos_token, <prompt tokens>, eos_token[, pad_token, ...]]
            # (typically self.max_length == 77)
            chunk_token_ids = [self.tokenizer.bos_token_id] + chunk_token_ids + [self.tokenizer.eos_token_id]
            chunk_token_weights = [1.0] + chunk_token_weights + [1.0]
            chunk_mask = [1] * len(chunk_token_ids)

            pad_length = self.max_token_count - len(chunk_token_ids)
            chunk_token_ids += [self.tokenizer.pad_token_id] * pad_length
            chunk_token_weights += [1.0] * pad_length
            chunk_mask += [self.padding_attention_mask_value] * pad_length

            all_token_ids += chunk_token_ids
            all_token_weights += chunk_token_weights
            all_masks += chunk_mask

            if self.truncate_to_model_max_length or len(remaining_token_ids) == 0:
                break

        all_token_ids_tensor = torch.tensor(all_token_ids, dtype=torch.long, device=device)
        all_per_token_weights_tensor = torch.tensor(all_token_weights,
                                                    dtype=self.get_dtype_for_device(device),
                                                    device=device)
        all_masks = torch.tensor(all_masks, dtype=torch.long, device=device)
        # print(f"assembled all_token_ids_tensor with shape {all_token_ids_tensor.shape}")
        return all_token_ids_tensor, all_per_token_weights_tensor, all_masks


    def build_weighted_embedding_tensor(self,
                                        token_ids: torch.Tensor,
                                        per_token_weights: torch.Tensor,
                                        attention_mask: Optional[torch.Tensor] = None,
                                        device: Optional[str] = None) -> torch.Tensor:
        """
        Build a tensor that embeds the passed-in token IDs and applies the given per_token weights
        
        :param token_ids: A tensor of shape `n*[self.max_length]` containing token IDs (ints) where n is some arbitrary
            integer (i.e. n==1 for shorter prompts, or it may be >1 if there are more than max_length tokens in the
            original prompt)
        :param per_token_weights: A tensor containing weights (floats), with the same shape as token_ids
        :param attention_mask: A tensor containing a mask (ints), with the same shape as token_ids, where 1 means use
            the corresponding token and 0 means ignore the corresponding token.

        :return: A tensor of shape `[1, token_ids.shape[0], token_dim]` representing the requested weighted embeddings
            where `token_dim` is 768 for SD1 and 1280 for SD2.
        """
        # print(f"building weighted embedding tensor for {tokens} with weights {token_weights}")
        if token_ids.shape[0] % self.max_token_count != 0:
            raise ValueError(f"token_ids has shape {token_ids.shape} - expected a multiple of {self.max_token_count}")

        if device is None:
            device = self.device

        chunk_start_index = 0
        empty_token_ids = torch.tensor([self.tokenizer.bos_token_id] +
                                       [self.tokenizer.eos_token_id] +
                                       [self.tokenizer.pad_token_id] * (self.max_token_count - 2),
                                       dtype=torch.int, device=device).unsqueeze(0)
        empty_z = self._encode_token_ids_to_embeddings(empty_token_ids)
        weighted_z = None

        chunk_size = self.max_token_count
        while chunk_start_index < token_ids.shape[0]:
            next_chunk_start_index = chunk_start_index+chunk_size
            chunk_per_token_weights = per_token_weights[chunk_start_index:next_chunk_start_index]
            chunk_token_ids = token_ids[chunk_start_index:next_chunk_start_index].unsqueeze(0)
            chunk_attention_mask = (
                attention_mask[chunk_start_index:next_chunk_start_index].unsqueeze(0)
                if attention_mask is not None
                else None
            )

            z = self._encode_token_ids_to_embeddings(chunk_token_ids, chunk_attention_mask)
            batch_weights_expanded = chunk_per_token_weights.reshape(
                chunk_per_token_weights.shape + (1,)).expand(z.shape).to(z)

            z_delta_from_empty = z - empty_z
            this_weighted_z = empty_z + (z_delta_from_empty * batch_weights_expanded)
            weighted_z = (
                this_weighted_z
                if weighted_z is None
                else torch.cat([weighted_z, this_weighted_z], dim=1)
            )
            chunk_start_index += chunk_size

        return weighted_z

    def _encode_token_ids_to_embeddings(self, token_ids: torch.Tensor,
                                        attention_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        needs_hidden_states = (self.returned_embeddings_type == ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED or
                               self.returned_embeddings_type == ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED)
        text_encoder_output = self.text_encoder(token_ids,
                                                attention_mask,
                                                output_hidden_states=needs_hidden_states,
                                                return_dict=True)
        if self.returned_embeddings_type is ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED:
            penultimate_hidden_state = text_encoder_output.hidden_states[-2]
            return penultimate_hidden_state
        elif self.returned_embeddings_type is ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED:
            penultimate_hidden_state = text_encoder_output.hidden_states[-2]
            return self.text_encoder.text_model.final_layer_norm(penultimate_hidden_state)
        elif self.returned_embeddings_type is ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED:
            # already normalized
            return text_encoder_output.last_hidden_state

        assert False, f"unrecognized ReturnEmbeddingsType: {self.returned_embeddings_type}"

    def _get_token_ranges_for_fragments(self, chunked_and_padded_token_ids: List[int], fragments: List[str]) -> List[Tuple[int, int]]:
        """
        Match token id sequences for the strings in `fragments` with token id sequences in `chunked_and_padded_token_ids`,
         taking into account any eos and bos markers that indicate `self.tokenizer.max_model_length`-sized chunks.

        :return: a list of tuples indicating start and end indices of each fragment's corresponding token id sequence in
         `chunked_and_padded_token_ids`.
        """
        per_fragment_token_ids = self.get_token_ids(fragments, include_start_and_end_markers=False)
        fragment_start = 0

        corresponding_indices = []
        for fragment_index, fragment_token_ids in enumerate(per_fragment_token_ids):
            if len(fragment_token_ids) == 0:
                corresponding_indices.append((None, None))
                continue
            if self.truncate_to_model_max_length and fragment_start >= self.tokenizer.model_max_length - 1:
                break
            # find the start
            while True:
                if fragment_start >= len(chunked_and_padded_token_ids)-1:
                    if self.truncate_to_model_max_length:
                        fragment_start = len(chunked_and_padded_token_ids)-1
                        break
                    else:
                        raise RuntimeError(
                            f"couldn't find start of token sequence for fragment at index {fragment_index} '{fragments[fragment_index]}'")
                if chunked_and_padded_token_ids[fragment_start] == fragment_token_ids[0]:
                    break
                fragment_start += 1
            # step through
            fragment_end = fragment_start
            fragment_relative_index = 0
            while True:
                if fragment_end >= len(chunked_and_padded_token_ids)-1:
                    if self.truncate_to_model_max_length:
                        fragment_end = len(chunked_and_padded_token_ids)-1
                        break
                    else:
                        raise RuntimeError(
                            f"couldn't find end of token sequence for fragment at index {fragment_index} '{fragments[fragment_index]}'")
                if not self.truncate_to_model_max_length and (
                        chunked_and_padded_token_ids[fragment_end] == self.tokenizer.eos_token_id
                        or chunked_and_padded_token_ids[fragment_end] == self.tokenizer.bos_token_id
                ):
                    # bos/eos: chunk boundaries
                    fragment_end += 1
                elif chunked_and_padded_token_ids[fragment_end] == fragment_token_ids[fragment_relative_index]:
                    # matching token
                    fragment_relative_index += 1
                    if fragment_relative_index == len(fragment_token_ids):
                        break
                    fragment_end += 1
                else:
                    raise RuntimeError(
                        f"token sequence mismatch for fragment at index {fragment_index} '{fragments[fragment_index]}':"
                        f"expected {fragment_token_ids}, found {chunked_and_padded_token_ids[fragment_start:fragment_end + 1]}")

            corresponding_indices.append((fragment_start, fragment_end))
            fragment_start = fragment_end + 1

        return corresponding_indices

    @classmethod
    def _pad_conditioning_tensors_to_same_length(cls, conditionings: List[torch.Tensor],
                                                 emptystring_conditioning: torch.Tensor
                                                 ) -> List[torch.Tensor]:
        if not all([len(c.shape) in [2, 3] for c in conditionings]):
            raise ValueError(
                "Conditioning tensors must all have either 2 dimensions (unbatched) or 3 dimensions (batched)")

        # ensure all conditioning tensors are 3 dimensions
        conditionings = [c.unsqueeze(0) if len(c.shape) == 2 else c for c in conditionings]
        c0_shape = conditionings[0].shape

        if not all([c.shape[0] == c0_shape[0] and c.shape[2] == c0_shape[2] for c in conditionings]):
            raise ValueError(
                f"All conditioning tensors must have the same batch size ({c0_shape[0]}) and number of embeddings per token ({c0_shape[1]}")

        if len(emptystring_conditioning.shape) == 2:
            emptystring_conditioning = emptystring_conditioning.unsqueeze(0)

        if not all(c.shape[1] % emptystring_conditioning.shape[1] == 0 for c in conditionings):
            raise ValueError(
                f"Token length of all conditioning tensors must be a multiple of the empty string conditioning token length ({emptystring_conditioning.shape[1]})"
            )

        empty_z = torch.cat([emptystring_conditioning] * c0_shape[0])
        max_token_count = max([c.shape[1] for c in conditionings])
        # if necessary, pad shorter tensors out with an emptystring tensor
        for i, c in enumerate(conditionings):
            while c.shape[1] < max_token_count:
                c = torch.cat([c, empty_z], dim=1)
                conditionings[i] = c
        return conditionings


    def remove_bos_eos(self, token_ids: list[Union[int,Any]]):
        if len(token_ids)>0 and type(token_ids[0]) is int:
            if self.needs_bos and token_ids[0] != self.tokenizer.bos_token_id:
                raise ValueError(f"attempt to remove bos marker from a token sequence that does not have it: {token_ids}")
            if self.needs_eos and token_ids[-1] != self.tokenizer.eos_token_id:
                raise ValueError(f"attempt to remove eos marker from a token sequence that does not have it: {token_ids}")
        start_idx = 1 if self.needs_bos else 0
        end_idx = -1 if self.needs_eos else None
        return token_ids[start_idx:end_idx]

    def add_bos_eos_if_required(self, token_ids: list[Union[int,Any]], bos_value=None, eos_value=None):
        if len(token_ids)>0 and type(token_ids[0]) is int:
            if self.needs_bos and token_ids[0] == self.tokenizer.bos_token_id:
                raise ValueError(f"asked to prepend bos marker to a token sequence that already has it")
            if self.needs_eos and token_ids[-1] == self.tokenizer.eos_token_id:
                raise ValueError(f"asked to append eos marker to a token sequence that already has it")
        prepend = ([self.tokenizer.bos_token_id if bos_value is None else bos_value]
                   if self.needs_bos
                   else [])
        append = ([self.tokenizer.eos_token_id if eos_value is None else eos_value]
                  if self.needs_eos
                  else [])
        return prepend + token_ids + append


class EmbeddingsProviderMulti:

    def __init__(self,
                tokenizers: CLIPTokenizer,
                text_encoders: Union[CLIPTextModel, CLIPTextModelWithProjection], # convert a list of int token ids to a tensor of embeddings
                textual_inversion_manager: BaseTextualInversionManager = None,
                dtype_for_device_getter: Callable[[torch.device], torch.dtype] = lambda device: torch.float32,
                truncate: bool = True,
                padding_attention_mask_value: int = 1,
                downweight_mode: DownweightMode = DownweightMode.MASK,
                returned_embeddings_type: Union[List[ReturnedEmbeddingsType], ReturnedEmbeddingsType] = ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
                 requires_pooled_mask: List[bool] = None
                ):

        requires_pooled_mask = [] if requires_pooled_mask is None else requires_pooled_mask
        returned_embeddings_type = len(text_encoders) * [returned_embeddings_type] if not isinstance(returned_embeddings_type, (list,tuple)) else returned_embeddings_type

        self.embedding_providers = [
            EmbeddingsProvider(tokenizer, text_encoder, textual_inversion_manager, dtype_for_device_getter, truncate, padding_attention_mask_value, downweight_mode, returned_embeddings_type)
            for tokenizer, text_encoder, returned_embeddings_type in zip(tokenizers, text_encoders, returned_embeddings_type)
        ]
        self.requires_pooled_mask = requires_pooled_mask

    @property
    def text_encoder(self):
        return self.embedding_providers[0].text_encoder

    @property
    def tokenizer(self):
        return self.embedding_providers[0].tokenizer

    def get_token_ids(self, *args, **kwargs):
        # get token ids does not use padding. The padding ID is the only ID that can differ between tokenizers
        # so for simplicity, we just return `get_token_ids` of the first tokenizer
        return self.embedding_providers[0].get_token_ids(*args, **kwargs)

    def get_pooled_embeddings(
        self, texts: List[str], attention_mask: Optional[torch.Tensor] = None, device: Optional[str] = None
    ) -> Optional[torch.Tensor]:

        pooled = [self.embedding_providers[provider_index].get_pooled_embeddings(texts, attention_mask, device=device)
                  for provider_index, requires_pooled in enumerate(self.requires_pooled_mask) if requires_pooled]

        if len(pooled) == 0:
            return None

        return torch.cat(pooled, dim=-1)

    def get_embeddings_for_weighted_prompt_fragments(self,
                                                     text_batch: List[List[str]],
                                                     fragment_weights_batch: List[List[float]],
                                                     should_return_tokens: bool = False,
                                                     device='cpu',
                                 ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        outputs = [provider.get_embeddings_for_weighted_prompt_fragments(text_batch, fragment_weights_batch, should_return_tokens=should_return_tokens, device=device) for provider in self.embedding_providers]

        text_embeddings_list = []
        tokens = []

        for output in outputs:
            text_embeddings_list.append(output[0])

            if should_return_tokens:
                tokens.append(output[1])

        text_embeddings = torch.cat(text_embeddings_list, dim=-1)

        if should_return_tokens:
            return text_embeddings, tokens
        else:
            return text_embeddings

