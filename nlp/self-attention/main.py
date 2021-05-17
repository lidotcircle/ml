#!/usr/bin/env python

from transformer import Transformer
from torchtext.data import get_tokenizer
from typing import List, Tuple, Generator, Iterator
from pathlib import Path
import io
import math
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class CnEnDataset(Dataset):
    def __init__(self, file: str, embedding_size: int, device: str, batch_size: int, max_sample: int = -1):
        self.file = file
        self.device = device
        self.len_idx: List[int] = []
        self.mapstore = {}
        self.index_me = None
        self.en_tokenizer = get_tokenizer('basic_english', language = 'en')
        self.en_word_to_idx = {}
        self.en_idx_to_word = [ ]
        self.cn_word_to_idx = {}
        self.cn_idx_to_word = [ ]
        self.__bos_symbol = "<BOS>"
        self.__eos_symbol = "<EOS>"
        self.__cn_append_word(self.__bos_symbol)
        self.__cn_append_word(self.__eos_symbol)
        self.__en_append_word(self.__bos_symbol)
        self.__en_append_word(self.__eos_symbol)
        self.embedding_size = embedding_size
        assert self.embedding_size > 1
        with io.open(file, mode="r", encoding="utf-8") as datafile:
            self.pair_dataset : List[Tuple[str, str]] = list(
            map(
                lambda pair: (pair[0], pair[1]),
                filter(
                    lambda pair: len(pair) == 2,
                    map(
                        lambda data: data.split('\t')[0:2],
                        datafile.read().split('\n')
                    )
                )
            )
        )[0:max_sample]
        for i in range(len(self.pair_dataset)):
            pair = self.pair_dataset[i]
            last = 0
            if len(self.len_idx) > 0:
                last = self.len_idx[-1]
            self.len_idx.append(last + len(pair[1]) + 1)
            en_words = list(self.en_tokenizer(pair[0]))
            for en_word in en_words:
                self.__en_append_word(en_word)
            for cn_word in pair[1]:
                self.__cn_append_word(cn_word)
            l1 = len(en_words)
            l2 = len(pair[1])
            if not l1 in self.mapstore:
                self.mapstore[l1] = {}
            storel1 = self.mapstore[l1]
            for v in range(l2 + 1):
                if not v in storel1:
                    storel1[v] = []
                storel1[v].append(i)
        self.batch_size = batch_size
        self.batchs = list(self.__batch_index_generate())
        self.__warm_index()

    def targetWordCount(self):
        return len(self.cn_idx_to_word)

    def sourceWordCount(self):
        return len(self.en_idx_to_word)

    def cn_bos(self):
        return self.en_word_to_idx[self.__bos_symbol]

    def cn_eos(self):
        return self.en_word_to_idx[self.__eos_symbol]

    def __en_append_word(self, en_word: str):
        if not en_word in self.en_word_to_idx:
            self.en_word_to_idx[en_word] = len(self.en_idx_to_word)
            self.en_idx_to_word.append(en_word)

    def __cn_append_word(self, cn_word: str):
        if not cn_word in self.cn_word_to_idx:
            self.cn_word_to_idx[cn_word] = len(self.cn_idx_to_word)
            self.cn_idx_to_word.append(cn_word)

    def __warm_index(self):
        self.index_me: List[Tuple[List[int], List[int], List[int]]] = []
        for en_s, cn_s in self.pair_dataset:
            en_idx = list(map(lambda word: self.en_word_to_idx[word], self.en_tokenizer(en_s)))
            cn_idx = [ self.cn_bos() ]
            cn_words = list(cn_s)
            cn_words.append(self.__eos_symbol)
            for word in cn_words:
                trg = list(cn_idx)
                cn_idx.append(self.cn_word_to_idx[word])
                y   = list(cn_idx)
                y.pop(0)
                self.index_me.append((en_idx, trg, y))
    
    def __en_idx_to_tensor(self, en_idx: Iterator[int]) -> torch.Tensor:
        l1 = []
        l2 = []
        indices = [l1, l2]
        values = []
        i = 0
        for idx in en_idx:
            l1.append(i)
            l2.append(idx)
            l1.append(i)
            l2.append(self.sourceWordCount())
            values.append(1)
            values.append(i)
            i = i + 1
        return torch.sparse_coo_tensor(indices, values, (len(en_idx), self.sourceWordCount() + 1)).float().to_dense()

    def __cn_trg_idx_to_tensor(self, cn_idx: Iterator[int]) -> torch.Tensor:
        l1 = []
        l2 = []
        indices = [l1, l2]
        values = []
        i = 0
        for idx in cn_idx:
            l1.append(i)
            l1.append(i)
            l2.append(idx)
            l2.append(self.targetWordCount())
            values.append(1)
            values.append(i)
            i = i + 1
        return torch.sparse_coo_tensor(indices, values, (len(cn_idx), self.targetWordCount() + 1)).float().to_dense()

    def __cn_y_idx_to_tensor(self, cn_y: Iterator[int]) -> torch.Tensor:
        return torch.tensor(list(cn_y))

    def pair_idx2idx(self, l1: int, l2: int) -> int:
        e = self.len_idx[l1] - len(self.pair_dataset[l1][1]) - 1 + l2
        return e

    def idx2pair_idx(self, idx: int) -> Tuple[int, int]:
        min = 0
        max = len(self.len_idx)
        while max > min:
            avg = (min + max) // 2
            high = self.len_idx[avg]
            lv = len(self.pair_dataset[avg][1])
            low = high - lv - 1
            if low <= idx < high:
                return avg, idx - low
            if low > idx:
                max = avg
            else:
                min = avg

    def testIndexEx(self):
        for i in range(len(self)):
            l1, l2 = self.idx2pair_idx(i)
            j = self.pair_idx2idx(l1, l2)
            if i != j:
                print(i, l1, l2, j)
            assert  j == i

    def __len__(self):
        return self.len_idx[-1]

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index < 0:
            index = len(self) + index
        assert self.index_me is not None
        x, trg, y = self.index_me[index]
        x_tensor = self.__en_idx_to_tensor(x)
        trg_tensor = self.__cn_trg_idx_to_tensor(trg)
        y_tensor = self.__cn_y_idx_to_tensor(y)
        return x_tensor, trg_tensor, y_tensor

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
                        ans.append(self.pair_idx2idx(idx, l2))
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
                yield self.batchs[n]


BATCH_SIZE = 300
LEARNING_RATE = 0.01
TRAIN_EPCHO = 1000
EMBEDDING_SIZE = 256


def train(dataloader: DataLoader, model: nn.Module, loss_fn, optimizer):
    print("begin training")
    i = 0
    for x, trg, y in dataloader:
        x = x.to(device)
        trg = trg.to(device)
        y = y.to(device)
        pred = model(x, trg)
        pred = pred.reshape(pred.shape[0] * pred.shape[1], pred.shape[2])
        y    = y.reshape(y.shape[0] * y.shape[1])
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        i = i + 1
        if i % 100 == 0:
            print(f"device: {device}, batch: {i}, loss: {loss:>7f}")
        if i % 1000 == 0:
            save_model(model)

def load_model(model: nn.Module):
    file = Path("model.pth")
    if file.is_file():
        print("load model")
        model.load_state_dict(torch.load(file))

def save_model(model: nn.Module):
    print("save model")
    torch.save(model.state_dict(), "model.pth")

def translate(model: nn.Module, dataset: CnEnDataset, eng_sentence: str) -> str:
    x = dataset.embed_a_case(eng_sentence, "")
    x = torch.tensor(x).to(device)
    eos = dataset.cn_eos()
    query = []
    while True:
        trg = dataset.getCnTarget(query)
        pred = model(x, trg)

device = "cuda" if torch.cuda.is_available() else "cpu"
if __name__ == '__main__':
    dataset = CnEnDataset("../../datasets/cmn-eng/cmn.txt", EMBEDDING_SIZE, device, BATCH_SIZE, -1)
    # non-leaf tensor can't cross process boundary
    # crash when num_workers > 4 in windows "OSError: (WinError 1455) The paging file is too small for this operation to complete"
    dataloader = DataLoader(dataset, batch_sampler=dataset.batchSampler(TRAIN_EPCHO), num_workers=2)
    # dataset.testIndexEx()

    model = Transformer(dataset.sourceWordCount(), dataset.targetWordCount(), 
                        heads = 8, embedding_size = EMBEDDING_SIZE, expansion = 4,
                        dropout = 0.2, layers = 6, device = device)
    load_model(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE)
    train(dataloader, model, loss_fn, optimizer)
