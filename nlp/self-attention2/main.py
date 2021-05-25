#!/usr/bin/env python

from transformer import Transformer
import pre_run
import csv
from torchtext.data import get_tokenizer
from typing import List, Tuple, Generator, Iterator
from pathlib import Path
import io
import sys
import math
import random

import numpy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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
        y: List[int] = cn_sent[0:min(_len + 1, len(cn_sent))]
        y.insert(0, self.__bos)
        if _len == len(cn_sent):
            y.append(self.__eos)
        trg = list(y)
        trg.pop()
        y.pop(0)
        size = len(self.cn_tokens)
        return self.__list2tensor(trg, size), torch.tensor(y)
    
    def __list2tensor(self, vallist: List[int], sparsesize: int):
        l1 = []
        l2 = []
        indices = [l1, l2]
        values = []
        for i in range(len(vallist)):
            l1.append(i)
            l2.append(vallist[i])
            l1.append(i)
            l2.append(sparsesize)
            values.append(1)
            values.append(i)
        return torch.sparse_coo_tensor(indices, values, (len(vallist), sparsesize + 1)).float()

    def embed_x(self, xlist: List[int]):
        return self.__list2tensor(xlist, len(self.en_tokens))

    def embed_trg(self, trglist: List[int]):
        trgl = list(trglist)
        trgl.insert(0, self.__bos)
        return self.__list2tensor(trgl, len(self.cn_tokens))

    def idx2tensor(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(idx.shape) == 1:
            sam_idx, trg_len = idx.tolist()
            x = self.__en_sample_index2tensor(sam_idx)
            trg, y = self.__cn_sample_index2tensor(sam_idx, trg_len)
            return x, trg, y
        else:
            indices = torch.unbind(idx, dim = 0)
            xl = []
            trgl = []
            yl = []
            for ix in indices:
                x, trg, y = self.idx2tensor(ix)
                xl.append(x)
                trgl.append(trg)
                yl.append(y)
            x = torch.stack(xl, dim = 0)
            trg = torch.stack(trgl, dim = 0)
            y = torch.stack(yl, dim = 0)
            return x, trg, y

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

    def __getitem__(self, index) -> torch.Tensor:
        if index < 0:
            index = len(self) + index
        a, b = self.__idx2pair_idx(index)
        return torch.tensor([a, b])

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


BATCH_SIZE = 500
LEARNING_RATE = 0.08
TRAIN_EPCHO = 1000
GONE_EPOCH = 0
EMBEDDING_SIZE = 280


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = numpy.average(values, weights=weights)
    # Fast and numerically precise:
    variance = numpy.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))

def train(dataloader: DataLoader, model: nn.Module, loss_fn, optimizer, scheduler):
    print("begin training")
    dataset: CnEnDataset = dataloader.dataset
    i = 0
    current_epoch = 0
    sample_count = 0
    loss_list = []
    loss_weight = []
    batch_len = dataset.batchLen()
    general_loss_list = []
    current_best_loss = -1
    for indices in dataloader:
        x, trg, y = dataset.idx2tensor(indices)
        x = x.to(device).to_dense()
        trg = trg.to(device).to_dense()
        y = y.to(device)
        pred = model(x, trg)
        pred = pred.reshape(pred.shape[0] * pred.shape[1], pred.shape[2])
        y    = y.reshape(y.shape[0] * y.shape[1])
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        i = i + 1
        current_batch_size = x.shape[0]
        sample_count = sample_count + current_batch_size
        loss_list.append(float(loss))
        loss_weight.append(current_batch_size)
        epoch_finished = i % batch_len == 0
        if epoch_finished or len(loss_list) == 100:
            loss_mean, loss_std = weighted_avg_and_std(loss_list, loss_weight)
            loss_list = []
            loss_weight = []
            general_loss_list.append(loss_mean)
            print(f"device: {device}, epoch: {current_epoch}, batch size: {BATCH_SIZE}, batch: {i}, total sample: {sample_count}, loss mean: {loss_mean:>7f}, loss_std: {loss_std:>7f}")
            sample_count = 0
        if epoch_finished:
            current_epoch = current_epoch + 1
            loss_mean = numpy.mean(general_loss_list)
            if current_best_loss < 0:
                current_best_loss = loss_mean
            if loss_mean < current_best_loss:
                print(f"save current best, loss mean: {loss_mean}")
                save_model(model, loss_mean)
                current_best_loss = loss_mean
            else:
                save_model(model)
            general_loss_list = []
            if scheduler is not None:
                scheduler.step()

def load_model(model: nn.Module):
    file = Path("model.pth")
    if file.is_file():
        print("load model")
        model.load_state_dict(torch.load(file))

def save_model(model: nn.Module, postfix: str = ''):
    print("save model")
    torch.save(model.state_dict(), f"model{postfix}.pth")

def list_max_index(l: List[float]):
    m = 0
    for i in range(len(l)):
        if l[i] > l[m]:
            m = i
    return m

def translate(model: nn.Module, dataset: CnEnDataset, eng_sentence: str) -> str:
    src = pre_run.en_tokenizer(eng_sentence)
    src = dataset.embed_x(src).unsqueeze(0).to_dense().to(device)
    eng_len = src.shape[1]
    trg = []
    while len(trg) == 0 or trg[-1] != dataset.cn_eos():
        trg = dataset.embed_trg(trg).unsqueeze(0).to_dense().to(device)
        pred = model(src, trg)
        trg = list(map(list_max_index, pred.squeeze(0).tolist()))
        if len(trg) > eng_len * 20:
            print(f"maybe translate fail!!! '{eng_sentence}'")
            break
    trg.pop()
    return dataset.cn_idx_list2sentence(trg)

device = "cuda" if torch.cuda.is_available() else "cpu"
if __name__ == '__main__':
    dataset = CnEnDataset(EMBEDDING_SIZE, BATCH_SIZE)
    # non-leaf tensor can't cross process boundary
    # crash when num_workers > 4 in windows "OSError: (WinError 1455) The paging file is too small for this operation to complete"
    dataloader = DataLoader(dataset, batch_sampler=dataset.batchSampler(TRAIN_EPCHO, suffle=True), num_workers=2)

    model = Transformer(dataset.en_tokens_count(), dataset.cn_tokens_count(), 
                        heads = 8, embedding_size = EMBEDDING_SIZE, expansion = 4,
                        dropout = 0.2, layers = 6, device = device)
    load_model(model)
    if len(sys.argv) == 1:
        loss_fn = nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
        optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE)
        # scheduler = None
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.995 ** (epoch + GONE_EPOCH) * LEARNING_RATE)
        print(f"learning rate: {LEARNING_RATE}, embedding size: {EMBEDDING_SIZE}, batch size: {BATCH_SIZE}")
        train(dataloader, model, loss_fn, optimizer, scheduler)
    else:
        ss = sys.argv[1:]
        sentence = ' '.join(ss)
        print(f"Origin: '{sentence}'")
        trans = translate(model, dataset, sentence)
        print(f"Target: '{trans}'")
