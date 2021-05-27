import torch
from torch import nn


class MultiHeadAttentionLocal(nn.Module):
    def __init__(self, heads: int, embedding_size: int, device: str):
        super(MultiHeadAttentionLocal, self).__init__()
        self.heads = heads
        self.embedding_size = embedding_size
        self.head_size = self.embedding_size // self.heads
        self.__device = device
        assert embedding_size % heads == 0

        self.queryTrans = nn.Linear(self.embedding_size, self.embedding_size)
        self.keyTrans   = nn.Linear(self.embedding_size, self.embedding_size)
        self.valueTrans = nn.Linear(self.embedding_size, self.embedding_size)
        self.linearOut  = nn.Linear(self.embedding_size, self.embedding_size)


    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        assert query.shape[0] == key.shape[0] == value.shape[0]
        batch_size = query.shape[0]

        query_len = query.shape[1]
        key_len = key.shape[1]
        value_len = value.shape[1]

        # TODO test performance of prior LN
        # Linear Transformation
        query = self.queryTrans(query)
        key   = self.keyTrans(key)
        value = self.valueTrans(value)

        # Reshape to multi heads
        query = query.reshape(batch_size, query_len, self.heads, self.head_size)
        key   = key.reshape  (batch_size, key_len,   self.heads, self.head_size)
        value = value.reshape(batch_size, value_len, self.heads, self.head_size)

        # (QK^T / sqrt(d_k))V
        coff = torch.einsum('bqhl,bkhl->bqhk', [query, key])
        d_k = key.shape[3]
        attention = torch.softmax(coff / (d_k ** (1/2)), dim = 3)

        ans = torch.einsum('bqhk,bkhl->bqhl', [attention, value]).reshape(batch_size, query_len, self.embedding_size)
        ans = self.linearOut(ans)
        return ans


class TextClassificationModel(nn.Module):
    pass