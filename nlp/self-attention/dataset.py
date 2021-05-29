import pre_run
from typing import List, Tuple, Iterator
import random

import torch
from torch.utils.data import Dataset


class CnEnDataset(Dataset):
    def __init__(self, embedding_size: int, batch_size: int,
                       sample_begin: int = 0, sample_end: int = -1):
        self.finished_epoch = 0
        self.len_idx: List[int] = []
        self.mapstore = {}
        self.__bos = pre_run.BOS
        self.__eos = pre_run.EOS
        self.embedding_size = embedding_size
        assert self.embedding_size > 1

        self.en_tokens, self.cn_tokens = pre_run.load_tokens()
        self.en_sentences, self.cn_sentences = pre_run.load_dataset(sample_begin, sample_end)
        assert len(self.en_sentences) == len(self.cn_sentences)

        for i in range(len(self.en_sentences)):
            en_sent = self.en_sentences[i]
            cn_sent = self.cn_sentences[i]
            last = 0
            if len(self.len_idx) > 0:
                last = self.len_idx[-1]
            en_sent_len = len(en_sent)
            cn_sent_len = len(cn_sent)
            self.len_idx.append(last + cn_sent_len + 1)
            if not en_sent_len in self.mapstore:
                self.mapstore[en_sent_len] = {}
            storel1 = self.mapstore[en_sent_len]
            for v in range(0, cn_sent_len + 1):
                if not v in storel1:
                    storel1[v] = []
                storel1[v].append(i)
        self.batch_size = batch_size
        self.batchs = list(self.__batch_index_generate())

    def adjust_batch_size(self, newbatch_size):
        self.batch_size = newbatch_size
        self.batchs = list(self.__batch_index_generate())

    def cn_bos(self):
        return self.__bos

    def cn_eos(self):
        return self.__eos
   
    def en_tokens_count(self) -> int:
       return len(self.en_tokens)
    
    def cn_tokens_count(self) -> int:
        return len(self.cn_tokens)

    def cn_idx_list2sentence(self, idxlist: List[int]):
        ll = map(lambda v: self.cn_tokens[v], idxlist)
        return ''.join(ll)

    def __en_sample_index2tensor(self, idx: int) -> torch.Tensor:
        en_sent = self.en_sentences[idx]
        return self.__list2tensor(en_sent, len(self.en_tokens))
    
    def __cn_sample_index2tensor(self, idx: int, _len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cn_sent = self.cn_sentences[idx]
        assert 0 <= _len <= len(cn_sent)
        trg: List[int] = cn_sent[0:_len+1]
        trg.insert(0, self.__bos)
        if _len == len(cn_sent):
            trg.append(self.__eos)
        y = list(trg)
        trg.pop()
        y.pop(0)
        size = len(self.cn_tokens)
        return self.__list2tensor(trg, size), torch.tensor(y)

    def en_tensor2sentence(self, tensor: torch.Tensor) -> str:
        assert len(tensor.shape) == 2
        pos = []
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                if tensor[i][j] != 0:
                    pos.append(j)
                    break
        text = map(lambda v: self.en_tokens[v], pos)
        return " ".join(text)

    def cn_tensor2sentence(self, tensor: torch.Tensor) -> str:
        assert len(tensor.shape) == 2
        pos = []
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                if tensor[i][j] != 0:
                    pos.append(j)
                    break
        text = map(lambda v: self.cn_tokens[v], pos)
        return "".join(text)

    def cn_scalar2word(self, scalar: torch.Tensor) -> str:
        if type(scalar) == torch.Tensor:
            assert len(scalar.shape) == 0
            scalar = scalar.tolist()
        return self.cn_tokens[scalar]
    
    def __list2tensor(self, vallist: List[int], _: int):
        return torch.tensor(vallist)

    def embed_x(self, xlist: List[int]):
        return self.__list2tensor(xlist, len(self.en_tokens))

    def embed_trg(self, trglist: List[int]):
        trgl = list(trglist)
        trgl.insert(0, self.__bos)
        return self.__list2tensor(trgl, len(self.cn_tokens))

    def __pair_idx2idx(self, l1: int, l2: int) -> int:
        return self.len_idx[l1] - len(self.cn_sentences[l1]) - 1 + l2

    def __idx2pair_idx(self, idx: int) -> Tuple[int, int]:
        min = 0
        max = len(self.len_idx)
        while max > min:
            avg = (min + max) // 2
            high = self.len_idx[avg]
            lv = len(self.cn_sentences[avg])
            low = high - lv - 1
            if low <= idx < high:
                return avg, idx - low
            if low > idx:
                max = avg
            else:
                min = avg

    def testIndexEx(self):
        for i in range(len(self)):
            l1, l2 = self.__idx2pair_idx(i)
            j = self.__pair_idx2idx(l1, l2)
            if i != j:
                print(i, l1, l2, j)
            assert  j == i

    def __len__(self):
        return self.len_idx[-1]

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index < 0:
            index = len(self) + index
        a, b = self.__idx2pair_idx(index)
        x = self.__en_sample_index2tensor(a)
        trg, y = self.__cn_sample_index2tensor(a, b)
        return x, trg, y

    def __batch_index_generate(self) -> Iterator[List[int]]:
        for l1 in self.mapstore.keys():
            l1store = self.mapstore[l1]
            for l2 in l1store.keys():
                vv = l1store[l2]
                read = 0
                while len(vv) > read:
                    ans = []
                    n = min(self.batch_size, len(vv) - read)
                    pairidx = vv[read:read+n]
                    read = read + n
                    for idx in pairidx:
                        ans.append(self.__pair_idx2idx(idx, l2))
                    yield ans

    def batchSampler(self, epcho: int, suffle: bool = True) -> Iterator[List[int]]:
        while epcho > 0:
            epcho = epcho - 1
            listme = list(range(len(self.batchs)))
            while len(listme) > 0:
                n = 0
                if suffle:
                    n = random.randrange(0, len(listme))
                n = listme.pop(n)
                if len(listme) == 0:
                    self.finished_epoch = self.finished_epoch + 1
                yield self.batchs[n]

    def batchLen(self):
        return len(self.batchs)
