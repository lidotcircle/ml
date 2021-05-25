import torch
import torch.nn as nn


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
            heads: int = 8, 
            embedding_size: int = 64, 
            expansion: int = 4, 
            dropout: float = 0.2, 
            layers: int = 6,
            device: str = 'cpu'
            ):
        super(Transformer, self).__init__()
        assert targetWordCount > 0
        self.srcEmbedMatrix = nn.Linear(sourceWordCount + 1, embedding_size).to(device)
        self.dstEmbedMatrix = nn.Linear(targetWordCount + 1, embedding_size).to(device)
        # self.srcEmbedMatrix.requires_grad_(False)
        # self.dstEmbedMatrix.requires_grad_(False)
        self.encoder = Encoder(heads, embedding_size, expansion, dropout, layers, device)
        self.decoder = Decoder(heads, embedding_size, expansion, dropout, layers, device)
        self.linearOut = nn.Linear(embedding_size, targetWordCount).to(device)
        # self.softmax = nn.Softmax(dim = 2).to(device)

    def forward(self, src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        src = self.srcEmbedMatrix(src)
        target = self.dstEmbedMatrix(target)
        srcEnc = self.encoder(src)
        out = self.decoder(target, srcEnc)
        # return self.softmax(self.linearOut(out))
        return self.linearOut(out)

