from typing import Union

import torch

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

def make_dummy_embedding():
    return torch.randn([768])

class DummyTransformer:


    def __init__(self):
        self.embeddings = DummyEmbeddingsList([make_dummy_embedding() for _ in range(len(KNOWN_WORDS)+2)])

    def resize_token_embeddings(self, new_size=None):
        if new_size is None:
            return self.embeddings
        else:
            while len(self.embeddings) > new_size:
                self.embeddings.pop(-1)
            while len(self.embeddings) < new_size:
                self.embeddings.append(make_dummy_embedding())

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, input_ids: torch.Tensor, return_dict: bool=False) -> torch.Tensor:
        if return_dict:
            raise AssertionError("for unit testing, return_dict must be false")
        if input_ids.shape[0] > 1:
            raise AssertionError("for unit testing, only batch size =1 is supported")
        return torch.index_select(torch.cat(self.embeddings), dim=0, index=input_ids.squeeze(0)).unsqueeze(0)

    def __call__(self, input_ids, **kwargs):
        return self.forward(input_ids=input_ids)

class DummyTokenizer():
    def __init__(self):
        self.tokens = KNOWN_WORDS.copy() + ["<|bos|>", "<|eos|>"]
        self.bos_token_id = len(self.tokens)-2
        self.eos_token_id = len(self.tokens)-2
        self.pad_token_id = self.eos_token_id
        self.unk_token_id = self.eos_token_id
        self.model_max_length = 77

    def __call__(self, fragments, **kwargs):
        return {'input_ids': [[self.tokens.index(w) for w in fragment.split(" ")]
                                           for fragment in fragments]}

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


class DummyClipEmbedder:
    def __init__(self):
        self.max_length = 77
        self.transformer = DummyTransformer()
        self.tokenizer = DummyTokenizer()
        self.position_embeddings_tensor = torch.randn([77,768], dtype=torch.float32)

    def position_embedding(self, indices: Union[list,torch.Tensor]):
        if type(indices) is list:
            indices = torch.tensor(indices, dtype=int)
        return torch.index_select(self.position_embeddings_tensor, 0, indices)
