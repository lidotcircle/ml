import torch
import torch.nn as nn
from typing import List


class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, embedding_size: int, device: str):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.embedding_size = embedding_size
        self.head_size = self.embedding_size // self.heads
        assert embedding_size % heads == 0

        self.queryTrans = nn.Linear(self.embedding_size, self.embedding_size).to(device)
        self.keyTrans   = nn.Linear(self.embedding_size, self.embedding_size).to(device)
        self.valueTrans = nn.Linear(self.embedding_size, self.embedding_size).to(device)
        self.linearOut  = nn.Linear(self.embedding_size, self.embedding_size).to(device)


    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        assert query.shape[0] == key.shape[0] == value.shape[0]
        batch_size = query.shape[0]

        query_len = query.shape[1]
        key_len = key.shape[1]
        value_len = value.shape[1]

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


class EncoderBlock(nn.Module):
    def __init__(self, heads: int, embedding_size: int, expansion: int, dropout: float, device: str):
        super(EncoderBlock, self).__init__()
        self.heads = heads
        self.embedding_size = embedding_size

        self.dropout = nn.Dropout(dropout).to(device)
        self.selfAttention = MultiHeadAttention(self.heads, self.embedding_size, device)
        self.norm1 = nn.LayerNorm(self.embedding_size).to(device)
        self.fcff = nn.Sequential(
                nn.Linear(self.embedding_size, expansion * self.embedding_size).to(device),
                nn.ReLU().to(device),
                nn.Linear(expansion * self.embedding_size, self.embedding_size).to(device)
                ).to(device)
        self.norm2 = nn.LayerNorm(self.embedding_size).to(device)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        v = self.selfAttention(src, src, src)
        v = self.dropout(self.norm1(src + v))

        b = self.fcff(v)
        b = self.dropout(self.norm2(b + v))
        return b


class DecoderBlock(nn.Module):
    def __init__(self, heads: int, embedding_size: int, expansion: int, dropout: float, device: str):
        super(DecoderBlock, self).__init__()
        self.heads = heads
        self.embedding_size = embedding_size

        self.dropout = nn.Dropout(dropout).to(device)
        self.norm1 = nn.LayerNorm(self.embedding_size).to(device)
        self.norm2 = nn.LayerNorm(self.embedding_size).to(device)
        self.norm3 = nn.LayerNorm(self.embedding_size).to(device)
        self.attention = MultiHeadAttention(self.heads, self.embedding_size, device)
        # TODO mask ??
        self.selfAttention = MultiHeadAttention(self.heads, self.embedding_size, device)
        self.fcff = nn.Sequential(
                nn.Linear(self.embedding_size, expansion * self.embedding_size).to(device),
                nn.ReLU().to(device),
                nn.Linear(expansion * self.embedding_size, self.embedding_size).to(device),
                ).to(device)

    def forward(self, target: torch.Tensor, encSrc: torch.Tensor) -> torch.Tensor:
        # TODO mask
        v = self.selfAttention(target, target, target)
        v = self.dropout(self.norm1(target + v))

        b = self.attention(target, encSrc, encSrc)
        b = self.dropout(self.norm2(v + b))

        n = self.fcff(b)
        n = self.dropout(self.norm3(b + n))
        return n


class Decoder(nn.Module):
    def __init__(self, heads: int, embedding_size: int, expansion: int, dropout: float, layers: int, device: str):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
                DecoderBlock(
                    heads, 
                    embedding_size, 
                    expansion, 
                    dropout,
                    device) for _ in range(0,layers)
                ])

    def forward(self, target: torch.Tensor, encSrc: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            target = layer(target, encSrc)
        return target


class Encoder(nn.Module):
    def __init__(self, heads: int, embedding_size: int, expansion: int, dropout: float, layers: int, device: str):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
                EncoderBlock(
                    heads, 
                    embedding_size, 
                    expansion, 
                    dropout,
                    device) for _ in range(0,layers)
                ])

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            src = layer(src)
        return src


class Transformer(nn.Module):
    def __init__(self, 
            sourceWordCount: int,
            targetWordCount: int,
            sourceSentenceMaxLength: int = 1000,
            targetSentenceMaxLength: int = 1000,
            heads: int = 8, 
            embedding_size: int = 64, 
            expansion: int = 4, 
            dropout: float = 0.2, 
            layers: int = 6,
            device: str = 'cpu'
            ):
        super(Transformer, self).__init__()
        assert targetWordCount > 0
        self.srcEmbedMatrix = nn.Linear(sourceWordCount, embedding_size).to(device)
        self.dstEmbedMatrix = nn.Linear(targetWordCount, embedding_size).to(device)
        self.srcPostionEmbedding = nn.Linear(sourceSentenceMaxLength, embedding_size).to(device)
        self.dstPostionEmbedding = nn.Linear(targetSentenceMaxLength, embedding_size).to(device)
        self.sourceWordCount = sourceWordCount
        self.targetWordCount = targetWordCount
        self.sourceSentenceMaxLength = sourceSentenceMaxLength
        self.targetSentenceMaxLength = targetSentenceMaxLength
        self.__device = device

        self.encoder = Encoder(heads, embedding_size, expansion, dropout, layers, device)
        self.decoder = Decoder(heads, embedding_size, expansion, dropout, layers, device)
        self.linearOut = nn.Linear(embedding_size, targetWordCount).to(device)
        # self.softmax = nn.Softmax(dim = 2).to(device)
    
    def __position_tensor(self, tensor: torch.Tensor, posLength: int) -> torch.Tensor:
        s = [tensor.shape[0] * tensor.shape[1], posLength]
        l1 = []
        l2 = []
        val = [ 1 ] * tensor.shape[0] * tensor.shape[1]
        for _ in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                l1.append(len(l1))
                l2.append(j)
        return torch.sparse_coo_tensor([l1, l2], val, s, dtype=torch.float)

    def __word_index_tensor(self, tensor: torch.Tensor, wordCount: int) -> torch.Tensor:
        s = [tensor.shape[0] * tensor.shape[1], wordCount]
        l1 = []
        l2 = []
        val = [ 1 ] * tensor.shape[0] * tensor.shape[1]
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                l1.append(len(l1))
                l2.append(tensor[i][j])
        return torch.sparse_coo_tensor([l1, l2], val, s, dtype=torch.float)

    def forward(self, src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert len(src.shape) == 2 and len(target.shape) == 2
        src_shape = src.shape
        target_shape = target.shape
        src_idx = self.__word_index_tensor(src, self.sourceWordCount)
        src_pos = self.__position_tensor(src, self.sourceSentenceMaxLength)
        trg_idx = self.__word_index_tensor(target, self.targetWordCount)
        trg_pos = self.__position_tensor(target, self.targetSentenceMaxLength)

        src_idx = self.srcEmbedMatrix(src_idx.to(self.__device))
        src_pos = self.srcPostionEmbedding(src_pos.to(self.__device))
        src = src_idx + src_pos
        src = src.reshape(src_shape[0], src_shape[1], src.shape[1])
        trg_idx = self.dstEmbedMatrix(trg_idx.to(self.__device))
        trg_pos = self.dstPostionEmbedding(trg_pos.to(self.__device))
        target = trg_idx + trg_idx
        target = target.reshape(target_shape[0], target_shape[1], target.shape[1])

        srcEnc = self.encoder(src)
        out = self.decoder(target, srcEnc)
        # return self.softmax(self.linearOut(out))
        return self.linearOut(out)

