from typing import Union, Optional
from unittest.mock import Mock, MagicMock

import torch
from torch import nn

KNOWN_WORDS = ['a', 'b', 'c']
KNOWN_WORDS_TOKEN_IDS = [0, 1, 2]
UNKNOWN_WORDS = ['d', 'e', 'f']

class DummyEmbeddingsList(list):
    def __getattr__(self, name):
        if name == 'num_embeddings':
            return len(self)
        elif name == 'weight':
            return self
        elif name == 'data':
            return self

def make_dummy_embedding(embedding_length):
    return torch.randn([embedding_length])
class Object(object):
    pass


class NullTransformer:
    device = 'cpu'


class DummyTransformer:

    def __init__(self, device="cpu", text_model_max_length=77, embedding_length=768):
        self.text_model_max_length = text_model_max_length
        self.embedding_length = embedding_length
        self.embeddings = DummyEmbeddingsList([make_dummy_embedding(self.embedding_length) for _ in range(len(KNOWN_WORDS)+3)])
        self.device = device

    def resize_token_embeddings(self, new_size=None):
        if new_size is None:
            return self.embeddings
        else:
            while len(self.embeddings) > new_size:
                self.embeddings.pop(-1)
            while len(self.embeddings) < new_size:
                self.embeddings.append(make_dummy_embedding(self.embedding_length))

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], output_hidden_states: bool=False, return_dict: bool=True):
        if input_ids.shape[0] > 1:
            raise AssertionError("for unit testing, only batch size =1 is supported")
        all_embeddings = torch.cat([e.unsqueeze(0) for e in self.embeddings]).to(self.device)
        embeddings = torch.index_select(all_embeddings, dim=0, index=input_ids.to(self.device).squeeze(0)
                                        ).unsqueeze(0)
        if attention_mask is not None:
            # this is not expected to match what Transformers actually does
            embeddings = embeddings * attention_mask.unsqueeze(2).expand(embeddings.shape)
        if not return_dict:
            return [embeddings, torch.empty_like(embeddings)]

        class EmbeddingsObject:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state

            def __getitem__(self, item):
                assert item == 0
                return self.last_hidden_state

            @property
            def hidden_states(self):
                return [-self.last_hidden_state, self.last_hidden_state]

            @property
            def text_embeds(self):
                return self.last_hidden_state[:, -1, :]

        o = EmbeddingsObject(embeddings)
        return o

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        return self.forward(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=kwargs.pop("output_hidden_states", False))

    @property
    def text_model(self):
        tm = Mock()
        tm.final_layer_norm = nn.LayerNorm(normalized_shape=[self.text_model_max_length, self.embedding_length])
        return tm


class DummyTokenizer():
    def __init__(self, model_max_length=77):
        self.tokens = KNOWN_WORDS.copy() + ["<|bos|>", "<|pad|>", "<|eos|>"]
        self.bos_token_id = len(self.tokens)-3
        self.pad_token_id = len(self.tokens)-2
        self.eos_token_id = len(self.tokens)-1
        self.unk_token_id = self.eos_token_id
        self.model_max_length = model_max_length

    def __call__(self, fragments, **kwargs):
        tokenized = [[self.bos_token_id] + [self.tokens.index(w) for w in fragment.split(" ")] + [self.eos_token_id]
                     if len(fragment)>0 else [self.bos_token_id] + [self.eos_token_id]
                                           for fragment in fragments]
        default_truncation = False
        if kwargs.get('truncation', default_truncation):
            max_length = kwargs.get('max_length', self.model_max_length)
            tokenized = [x[0:self.model_max_length-1] + [self.eos_token_id] if len(x)>max_length
                         else x
                         for x in tokenized]
        padding_strategy = kwargs.get('padding', 'do_not_pad')
        if padding_strategy not in ['do_not_pad', 'max_length']:
            raise Exception(f"for unit tests only 'do_not_pad' and 'max_length' is supported as a padding strategy (got '{padding_strategy}')")

        if padding_strategy == "max_length":
            tokenized = [(tokens[:-1] + (self.model_max_length - len(tokens)) * [self.pad_token_id] + tokens[1:]) for tokens in tokenized]

        return {'input_ids': tokenized}

    def convert_tokens_to_ids(self, token_str):
        try:
            return self.tokens.index(token_str)
        except ValueError:
            return self.unk_token_id

    def add_tokens(self, token_str):
        if token_str in self.tokens:
            return 0
        self.tokens.append(token_str)
        return 1
