#!/usr/bin/env python

from transformer import Transformer
from torchtext.data import get_tokenizer
from typing import List, Tuple, Generator, Iterator
import io

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class CnEnDataset(Dataset):
    def __init__(self, file: str, embedding_size: int, device: str, max_sample: int = -1):
        self.file = file
        self.device = device
        self.dataset: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.en_tokenizer = get_tokenizer('basic_english', language = 'en')
        self.en_word_to_idx = {}
        self.en_idx_to_word = [ '' ]
        self.cn_word_to_idx = {}
        self.cn_idx_to_word = [ '' ]
        self.__bos_symbol = "<BOS>"
        self.__eos_symbol = "<EOS>"
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
            for en_word in self.en_tokenizer(pair[0]):
                self.__en_append_word(en_word)
            for cn_word in pair[1]:
                self.__cn_append_word(cn_word)
        self.__cn_append_word(self.__bos_symbol)
        self.__cn_append_word(self.__eos_symbol)
        self.__en_append_word(self.__bos_symbol)
        self.__en_append_word(self.__eos_symbol)
        self.en_embed = nn.Embedding(len(self.en_idx_to_word) + 1, self.embedding_size - 1)
        self.cn_embed = nn.Embedding(len(self.cn_idx_to_word) + 1, self.embedding_size - 1)
        self.__init_dataset()

    def __en_append_word(self, en_word: str):
        if not en_word in self.en_word_to_idx:
            self.en_idx_to_word.append(en_word)
            self.en_word_to_idx[en_word] = len(self.en_idx_to_word)

    def __cn_append_word(self, cn_word: str):
        if not cn_word in self.cn_word_to_idx:
            self.cn_idx_to_word.append(cn_word)
            self.cn_word_to_idx[cn_word] = len(self.cn_idx_to_word)

    def embed_a_case(self, src_sentence: str, trg_sentence: str) -> Tuple[List, List[Tuple[List, List]]]:
        targets: List[Tuple[List, List]] = []
        src_words: List[str] = self.en_tokenizer(src_sentence)
        target_words: List[str] = list(map(lambda v: v, trg_sentence))

        src_embed = []
        for i in range(len(src_words)):
            val = src_words[i]
            embed = self.en_embed(torch.tensor(self.en_word_to_idx[val]))
            embed = torch.cat((embed, torch.tensor([i + 1])), 0)
            src_embed.append(embed.tolist())

        for i in range(len(target_words)):
            v = [ torch.cat((self.cn_embed(torch.tensor(self.cn_word_to_idx[self.__bos_symbol])), torch.tensor([ 0 ])), 0).tolist() ]
            for j in range(i + 1):
                embed = self.cn_embed(torch.tensor(self.cn_word_to_idx[target_words[i]]))
                v.append(torch.cat((embed, torch.tensor([j + 1])), 0).tolist())
            target = list(v)
            output = list(v)
            target.pop()
            output.pop(0)
            targets.append((target, output))
            if len(v) == len(target_words) + 1:
                target = list(v)
                output = list(v)
                output.pop(0)
                t = torch.tensor(self.cn_word_to_idx[self.__eos_symbol])
                output.append(torch.cat((self.cn_embed(t), torch.tensor([ -1 ])), 0).tolist())
                targets.append((target, output))

        return src_embed, targets

    def __append_a_case(self, x: List, trg_y: List[Tuple[List, List]]):
        for trg, y in trg_y:
            self.dataset.append((
                torch.tensor(x).to(self.device),
                torch.tensor(trg).to(self.device),
                torch.tensor(y).to(self.device)
                ))

    def __init_dataset(self):
        for src, target in self.pair_dataset:
            x, trg_y = self.embed_a_case(src, target)
            self.__append_a_case(x, trg_y)
        self.dataset.sort(key = lambda tp: len(tp[0]) * 2048 + len(tp[1]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def batchSampler(self, batch_size: int) -> Iterator[List[int]]:
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
    for epcho in range(TRAIN_EPCHO):
        j = 0
        for x, tgr, y in dataloader:
            pred = model(x, tgr)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i = i + 1
            j = j + 1
            if i % 100 == 0:
                print(f"device: {device}, epcho: {epcho}, batch: {j}, loss: {loss:>7f}")

def load_model(model: nn.Module):
    model.load_state_dict(torch.load("model.pth"))
    return model

def save_model(model: nn.Module):
    torch.save(model.state_dict(), "model.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"
if __name__ == '__main__':
    dataset = CnEnDataset("../../datasets/cmn-eng/cmn.txt", EMBEDDING_SIZE, device, 2000)
    dataloader = DataLoader(dataset, batch_sampler=dataset.batchSampler(BATCH_SIZE))
    dataloader_fake = list(dataloader.__iter__())
    model = Transformer(heads = 8, embedding_size = EMBEDDING_SIZE, expansion = 4, dropout = 0.2, layers = 6, device = device)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE)
    train(dataloader_fake, model, loss_fn, optimizer)
