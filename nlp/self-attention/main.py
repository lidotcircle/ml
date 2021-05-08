#!/usr/bin/env python

from transformer import Transformer
from torchtext.data import get_tokenizer
from typing import List, Tuple, Generator, Iterator
from pathlib import Path
import io
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class CnEnDataset(Dataset):
    def __init__(self, file: str, embedding_size: int, device: str, max_sample: int = -1):
        self.file = file
        self.device = device
        self.__len = 0
        self.dataset: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
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
        for pair in self.pair_dataset:
            self.__len = self.__len + len(pair[1]) + 1
            for en_word in self.en_tokenizer(pair[0]):
                self.__en_append_word(en_word)
            for cn_word in pair[1]:
                self.__cn_append_word(cn_word)
        self.en_embed = nn.Embedding(len(self.en_idx_to_word) + 1, self.embedding_size - 1)
        self.cn_embed = nn.Embedding(len(self.cn_idx_to_word) + 1, self.embedding_size - 1)

    def cn_bos(self):
        return self.en_word_to_idx[self.__bos_symbol]

    def cn_eos(self):
        return self.en_word_to_idx[self.__eos_symbol]

    def __en_append_word(self, en_word: str):
        if not en_word in self.en_word_to_idx:
            self.en_idx_to_word.append(en_word)
            self.en_word_to_idx[en_word] = len(self.en_idx_to_word)

    def __cn_append_word(self, cn_word: str):
        if not cn_word in self.cn_word_to_idx:
            self.cn_idx_to_word.append(cn_word)
            self.cn_word_to_idx[cn_word] = len(self.cn_idx_to_word)
    
    def toword(self, value: float) -> Tuple[int, str]:
        val = math.floor(value)
        if value - val > 0.5:
            val = val + 1
        if val < 0 or val > len(self.cn_idx_to_word):
            return val, "FAIL"
        return val, self.cn_idx_to_word[val]

    def getCnTarget(self, targets: List[float]) -> torch.Tensor:
        targets.insert(0, self.cn_bos())
        concat = []
        for i in range(len(targets)):
            self.embed_position(self.cn_embed, targets[i], i)
        return torch.stack(concat, dim = 0).to(self.device)

    def embed_position(self, embed, word_idx: int, pos: int):
        return torch.cat((embed(torch.tensor(word_idx)), torch.tensor([ pos ])), 0)

    def embed_a_case(self, src_sentence: str, trg_sentence: str) -> Tuple[List, List[Tuple[List, List]]]:
        trg_y: List[Tuple[List, List]] = []

        src_words: List[str] = self.en_tokenizer(src_sentence)
        src_embed = []
        for i in range(len(src_words)):
            val = src_words[i]
            embed = self.embed_position(self.en_embed, self.en_word_to_idx[val], i + 1)
            src_embed.append(embed.tolist())

        target_words: List[str] = list(map(lambda v: v, trg_sentence))
        target_words.append(self.__eos_symbol)
        for i in range(len(target_words)):
            y = []
            trg = [ self.embed_position(self.cn_embed, self.cn_word_to_idx[self.__bos_symbol], 0).tolist() ]
            for j in range(i + 1):
                word_idx = self.cn_word_to_idx[target_words[j]]
                y.append([ word_idx ])
                if j < i:
                    trg.append(self.embed_position(self.cn_embed, word_idx, j + 1).tolist())

            trg_y.append((trg, y))

        return src_embed, trg_y

    def __append_a_case(self, x: List, trg_y: List[Tuple[List, List]]):
        for trg, y in trg_y:
            self.dataset.append((
                torch.tensor(x).to(self.device),
                torch.tensor(trg).to(self.device),
                torch.tensor(y).to(self.device)
                ))

    def __init_dataset(self, start: int, end: int):
        self.dataset: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for src, target in self.pair_dataset[start:end]:
            x, trg_y = self.embed_a_case(src, target)
            self.__append_a_case(x, trg_y)
        self.dataset.sort(key = lambda tp: len(tp[0]) * 2048 + len(tp[1]))

    def __len__(self):
        return self.__len

    def __getitem__(self, index):
        return self.dataset[index]

    def batchSampler(self, batch_size: int, epcho: int, division: int, switch_epcho: int) -> Iterator[List[int]]:
        osize = math.ceil(len(self.pair_dataset) / division)

        while epcho > 0:
            epcho = epcho - switch_epcho
            os = 0
            while os < len(self.pair_dataset):
                self.__init_dataset(os, os + osize)
                os = os + osize
                se = switch_epcho
                while se > 0:
                    se = se - 1
                    start: int = 0
                    while start < len(self.dataset):
                        begin = start
                        ans = []
                        m = self.dataset[start][0].shape[0]
                        n = self.dataset[start][1].shape[0]
                        while start < len(self.dataset) \
                            and start < begin + batch_size \
                            and self.dataset[start][0].shape[0] == m \
                            and self.dataset[start][1].shape[0] == n:
                            ans.append(start)
                            start = start + 1
                        yield ans


BATCH_SIZE = 2048
LEARNING_RATE = 0.002
TRAIN_EPCHO = 1000
EMBEDDING_SIZE = 512


def train(dataloader: DataLoader, model: nn.Module, loss_fn, optimizer):
    print("begin training")
    i = 0
    for x, tgr, y in dataloader:
        x = x.to(device)
        tgr = tgr.to(device)
        y = y.to(device)
        pred = model(x, tgr)
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
    dataset = CnEnDataset("../../datasets/cmn-eng/cmn.txt", EMBEDDING_SIZE, device, -1)
    dataloader = DataLoader(dataset, batch_sampler=dataset.batchSampler(BATCH_SIZE, TRAIN_EPCHO, 30, 10))

    model = Transformer(heads = 8, embedding_size = EMBEDDING_SIZE, expansion = 4, dropout = 0.2, layers = 6, device = device)
    load_model(model)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE)
    train(dataloader, model, loss_fn, optimizer)
