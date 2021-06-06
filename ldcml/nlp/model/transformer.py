import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, List


__tri_mask_store = {}
def get_mask_tensor(row: int, col: int, device: str) -> Tensor:
    key = f"{row}-{col}-{device}"
    if key in __tri_mask_store:
        return __tri_mask_store[key]
    ans = torch.triu(torch.full((row, col), True)) == False
    ans = ans.to(device)
    __tri_mask_store[key] = ans
    return ans


class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, embedding_size: int):
        """ multi-head attention

        Parameters
        ---------
        heads: int
            how many heads
        embedding_size: int
            vector dimension
        """
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.embedding_size = embedding_size
        self.head_size = self.embedding_size // self.heads
        assert embedding_size % heads == 0
        self.__sqrt_embedding_size = (embedding_size / heads) ** 0.5

        self.queryTrans = nn.Linear(self.embedding_size, self.embedding_size)
        self.keyTrans   = nn.Linear(self.embedding_size, self.embedding_size)
        self.valueTrans = nn.Linear(self.embedding_size, self.embedding_size)
        self.linearOut  = nn.Linear(self.embedding_size, self.embedding_size)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, masked = None) -> Tensor:
        """
        Parameters
        ----------
        masked: Tensor
            mask attention, for example used to prevent nth query 
            from attenting to no-first-n key-value pair
        """
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
        key   = key  .reshape(batch_size, key_len,   self.heads, self.head_size)
        value = value.reshape(batch_size, value_len, self.heads, self.head_size)

        # (QK^T / sqrt(d_k))V
        coff: Tensor = torch.einsum('bqhl,bkhl->bhqk', [query, key])
        if masked is not None:
            coff.masked_fill_(masked, -1e9)
        attention = torch.softmax(torch.div(coff, self.__sqrt_embedding_size), dim = 3)

        ans = torch.einsum('bhqk,bkhl->bqhl', [attention, value])
        ans = ans.reshape(batch_size, query_len, self.embedding_size)
        ans = self.linearOut(ans)
        return ans


class EncoderBlock(nn.Module):
    def __init__(self, heads: int, embedding_size: int, expansion: int, dropout: float):
        super(EncoderBlock, self).__init__()
        self.heads = heads
        self.embedding_size = embedding_size

        self.dropout = nn.Dropout(dropout)
        self.selfAttention = MultiHeadAttention(self.heads, self.embedding_size)
        self.norm1 = nn.LayerNorm(self.embedding_size)
        self.fcff = nn.Sequential(
                nn.Linear(self.embedding_size, expansion * self.embedding_size),
                nn.ReLU(),
                nn.Linear(expansion * self.embedding_size, self.embedding_size)
                )
        self.norm2 = nn.LayerNorm(self.embedding_size)

    def forward(self, src: Tensor) -> Tensor:
        v = self.selfAttention(src, src, src)
        v = self.dropout(self.norm1(src + v))

        b = self.fcff(v)
        b = self.dropout(self.norm2(b + v))
        return b


class DecoderBlock(nn.Module):
    def __init__(self, heads: int, embedding_size: int, 
                 expansion: int, dropout: float, device: str):
        super(DecoderBlock, self).__init__()
        self.heads = heads
        self.embedding_size = embedding_size
        self.__device = device

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(self.embedding_size)
        self.norm2 = nn.LayerNorm(self.embedding_size)
        self.norm3 = nn.LayerNorm(self.embedding_size)
        self.attention = MultiHeadAttention(self.heads, self.embedding_size)
        self.selfAttention = MultiHeadAttention(self.heads, self.embedding_size)
        self.fcff = nn.Sequential(
                nn.Linear(self.embedding_size, expansion * self.embedding_size),
                nn.ReLU(),
                nn.Linear(expansion * self.embedding_size, self.embedding_size),
                )

    def forward(self, target: Tensor, encSrc: Tensor) -> Tensor:
        row = col = target.shape[1]
        masked = get_mask_tensor(row, col, self.__device)
        v = self.selfAttention(target, target, target, masked)
        v = self.dropout(self.norm1(target + v))

        b = self.attention(target, encSrc, encSrc)
        b = self.dropout(self.norm2(v + b))

        n = self.fcff(b)
        n = self.dropout(self.norm3(b + n))
        return n


class Decoder(nn.Module):
    def __init__(self, heads: int, embedding_size: int,
                 expansion: int, dropout: float, layers: int, device: str):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
                DecoderBlock(
                    heads, 
                    embedding_size, 
                    expansion, 
                    dropout,
                    device) for _ in range(0,layers)
                ])

    def forward(self, target: Tensor, encSrc: Tensor) -> Tensor:
        for layer in self.layers:
            target = layer(target, encSrc)
        return target


class Encoder(nn.Module):
    def __init__(self, heads: int, embedding_size: int,
                 expansion: int, dropout: float, layers: int):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
                EncoderBlock(
                    heads, 
                    embedding_size, 
                    expansion, 
                    dropout) for _ in range(0,layers)
                ])

    def forward(self, src: Tensor) -> Tensor:
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
            embed_grad: bool = True,
            device: str = 'cpu'
            ):
        super(Transformer, self).__init__()
        assert targetWordCount > 0
        self.srcEmbedMatrix = nn.Linear(sourceWordCount, embedding_size)
        self.dstEmbedMatrix = nn.Linear(targetWordCount, embedding_size)
        self.srcPostionEmbedding = nn.Linear(sourceSentenceMaxLength, embedding_size)
        self.dstPostionEmbedding = nn.Linear(targetSentenceMaxLength, embedding_size)
        self.srcEmbedMatrix.requires_grad_(embed_grad)
        self.dstEmbedMatrix.requires_grad_(embed_grad)
        self.srcPostionEmbedding.requires_grad_(embed_grad)
        self.dstPostionEmbedding.requires_grad_(embed_grad)

        self.sourceWordCount = sourceWordCount
        self.targetWordCount = targetWordCount
        self.sourceSentenceMaxLength = sourceSentenceMaxLength
        self.targetSentenceMaxLength = targetSentenceMaxLength
        self.__device = device

        self.encoder = Encoder(heads, embedding_size, expansion, dropout, layers)
        self.decoder = Decoder(heads, embedding_size, expansion, dropout, layers, device)
        self.linearOut = nn.Linear(embedding_size, targetWordCount)
        # self.softmax = nn.Softmax(dim = 2)

    @staticmethod
    def __linear2tensor(linear: nn.Linear) -> Tensor:
        linear = linear.to("cpu")
        tensor = Tensor(linear.in_features, linear.out_features)
        for i in range(linear.in_features):
            idx = torch.sparse_coo_tensor([[i]], [1], [linear.in_features], dtype=torch.float)
            val = linear(idx)
            tensor[i] = val
        return tensor

    def embedMatrics(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        a = Transformer.__linear2tensor(self.srcEmbedMatrix)
        b = Transformer.__linear2tensor(self.srcPostionEmbedding)
        c = Transformer.__linear2tensor(self.dstEmbedMatrix)
        d = Transformer.__linear2tensor(self.dstPostionEmbedding)
        return a, b, c, d
    
    def embedSrcAndTrg(self, batch_size: int, src: Tensor, srcpos: Tensor, trg: Tensor, trgpos: Tensor) -> Tuple[Tensor, Tensor]:
        src_idx = torch.sparse_coo_tensor(src[0:2],    src[2],    [src.shape[1], self.sourceWordCount], dtype=torch.float)
        src_pos = torch.sparse_coo_tensor(srcpos[0:2], srcpos[2], [src.shape[1], self.sourceSentenceMaxLength], dtype=torch.float)
        trg_idx = torch.sparse_coo_tensor(trg[0:2],    trg[2],    [trg.shape[1], self.targetWordCount], dtype=torch.float)
        trg_pos = torch.sparse_coo_tensor(trgpos[0:2], trgpos[2], [trg.shape[1], self.targetSentenceMaxLength], dtype=torch.float)

        src_idx = src_idx.to(self.__device)
        src_pos = src_pos.to(self.__device)
        trg_idx = trg_idx.to(self.__device)
        trg_pos = trg_pos.to(self.__device)

        src_idx = self.srcEmbedMatrix(src_idx)
        src_pos = self.srcPostionEmbedding(src_pos)
        src = src_idx + src_pos
        assert src.shape[0] % batch_size == 0
        xseq_length = src.shape[0] // batch_size
        src = src.reshape(batch_size, xseq_length, src.shape[1])

        trg_idx = self.dstEmbedMatrix(trg_idx)
        trg_pos = self.dstPostionEmbedding(trg_pos)
        trg = trg_idx + trg_idx
        assert trg.shape[0] % batch_size == 0
        trgseq_length = trg.shape[0] // batch_size
        trg = trg.reshape(batch_size, trgseq_length, trg.shape[1])
        return src, trg
    
    def forward(self, batch_size: int, src: Tensor, srcpos: Tensor, trg: Tensor, trgpos: Tensor) -> Tensor:
        if batch_size > 0:
            src, trg = self.embedSrcAndTrg(batch_size, src, srcpos, trg, trgpos)

        srcEnc = self.encoder(src)
        out = self.decoder(trg, srcEnc)
        return self.linearOut(out)


def __position_tensor(shape0: int, shape1: int) -> Tensor:
    l1 = []
    l2 = []
    val = [ 1 ] * shape0 * shape1
    for _ in range(shape0):
        for j in range(shape1):
            l1.append(len(l1))
            l2.append(j)
    return torch.tensor([l1, l2, val])

def __word_index_tensor(tensor: List[List[int]]) -> Tensor:
    l1 = []
    l2 = []
    shape0 = len(tensor)
    shape1 = len(tensor[0])
    val = [ 1 ] * shape0 * shape1
    for i in range(shape0):
        for j in range(shape1):
            l1.append(len(l1))
            l2.append(tensor[i][j])
    return torch.tensor([l1, l2, val])

def generate_batch(x: List[List[int]], trg: List[List[int]], y: List[List[int]]) -> Tuple[int, Tensor, List[any]]:
    assert len(x) == len(trg) == len(y)
    batch_size = len(x)
    xword = __word_index_tensor(x)
    xpos  = __position_tensor(len(x), len(x[0]))
    trgword = __word_index_tensor(trg)
    trgpos  = __position_tensor(len(trg), len(trg[0]))
    return batch_size, torch.tensor(y, dtype=torch.long), [batch_size, xword, xpos, trgword, trgpos]
