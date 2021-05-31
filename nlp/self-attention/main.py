#!/usr/bin/env python

from transformer import Transformer
from dataset import CnEnDataset
import pre_run
from typing import List, Tuple, Generator, Iterator
from pathlib import Path
import csv
import io
import sys
import time
import math

import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


BATCH_SIZE = 300
LEARNING_RATE = 0.08
TRAIN_EPCHO = 1000
GONE_EPOCH = 0
EMBEDDING_SIZE = 100


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = numpy.average(values, weights=weights)
    # Fast and numerically precise:
    variance = numpy.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))

def train(dataset: Dataset, model: nn.Module, loss_fn, optimizer, scheduler):
    print("begin training")
    # non-leaf tensor can't cross process boundary
    # crash when num_workers > 4 in windows "OSError: (WinError 1455) The paging file is too small for this operation to complete"
    dataloader = DataLoader(dataset, 
                            batch_sampler=dataset.batchSampler(TRAIN_EPCHO, suffle=True), 
                            collate_fn=wrap_collate_fn, 
                            num_workers=2)
    i = 0
    current_epoch = 0
    sample_count = 0
    loss_list = []
    loss_weight = []
    batch_len = dataset.batchLen()
    general_loss_list = []
    current_best_loss = -1
    last_save = time.time()
    for batch_size, xseq_length, trgseq_length, x, trg, y in dataloader:
        y = y.to(device)
        pred = model(batch_size, xseq_length, trgseq_length, x, trg)
        pred = pred.reshape(pred.shape[0] * pred.shape[1], pred.shape[2])
        y    = y.reshape(y.shape[0] * y.shape[1])
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        i = i + 1
        current_batch_size = batch_size
        sample_count = sample_count + current_batch_size
        loss_list.append(float(loss))
        loss_weight.append(current_batch_size)
        epoch_finished = i % batch_len == 0

        if (time.time() - last_save) > 15 * 60:
            last_save = time.time()
            save_model(model)

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
            print(f"loss mean: {loss_mean}")
            if current_best_loss < 0:
                current_best_loss = loss_mean
            if loss_mean < current_best_loss:
                print(f"save current best")
                save_model(model, loss_mean)
                current_best_loss = loss_mean
            else:
                save_model(model)
            general_loss_list = []
            if scheduler is not None:
                scheduler.step()

def load_model(model: nn.Module):
    file = Path("saved_model/model.pth")
    if file.is_file():
        print("load model")
        model.load_state_dict(torch.load(file))

def save_model(model: nn.Module, postfix: str = ''):
    print("save model")
    torch.save(model.state_dict(), f"saved_model/model{postfix}.pth")

def list_max_index(l: List[float]):
    m = 0
    for i in range(len(l)):
        if l[i] > l[m]:
            m = i
    return m

def translate(model: nn.Module, dataset: CnEnDataset, src_sentence: str) -> str:
    src = pre_run.en_tokenizer(src_sentence)
    src = dataset.embed_x(pre_run.en_tokenizer(src_sentence))
    eng_len = src.shape[0]
    trg_list = []
    while len(trg_list) == 0 or trg_list[-1] != dataset.cn_eos():
        trg = dataset.embed_trg(trg_list)
        batch_size, xseq_length, trgseq_length, x, trg, _ = wrap_collate_fn([(src, trg, torch.tensor([]))])
        pred = model(batch_size, xseq_length, trgseq_length, x, trg)
        pred = torch.softmax(pred, dim = 2).squeeze(0).tolist()
        pred = pred[-1]
        y = list_max_index(pred)
        trg_list.append(y)
        if len(trg_list) > eng_len * 20:
            print(f"maybe translate fail!!! '{src_sentence}'")
            break
    trg_list.pop()
    return dataset.cn_idx_list2sentence(trg_list)


def position_tensor(sentenceLength: int, posLength: int) -> torch.Tensor:
    s = [sentenceLength, posLength]
    l1 = []
    l2 = []
    val = [ 1 ] * sentenceLength
    for i in range(sentenceLength):
        l1.append(len(l1))
        l2.append(i)
    return torch.sparse_coo_tensor([l1, l2], val, s, dtype=torch.float)


def __position_tensor(tensor: torch.Tensor) -> torch.Tensor:
    l1 = []
    l2 = []
    val = [ 1 ] * tensor.shape[0] * tensor.shape[1]
    for _ in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            l1.append(len(l1))
            l2.append(j)
    return torch.tensor([l1, l2, val])

def __word_index_tensor(tensor: torch.Tensor) -> torch.Tensor:
    l1 = []
    l2 = []
    val = [ 1 ] * tensor.shape[0] * tensor.shape[1]
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            l1.append(len(l1))
            l2.append(tensor[i][j])
    return torch.tensor([l1, l2, val])

def __save_tensor2csv(t: torch.Tensor, fn: str, header: List[str] = None, vmap = None):
    with io.open(fn, "w", encoding="utf-8-sig", newline="") as sf:
        writer = csv.writer(sf)
        l = t.tolist()

        if vmap is not None:
            for i in range(len(l)):
                for j in range(len(l[0])):
                    l[i][j] = vmap(l[i][j])

        if header is not None:
            for i in range(len(header)):
                l[i].insert(0, header[i])
            header.insert(0, "")
            writer.writerow(header)

        writer.writerows(l)

def save_embed_matrix(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor):
    __save_tensor2csv(a, "./running_data/src_embed")
    __save_tensor2csv(b, "./running_data/src_position")
    __save_tensor2csv(c, "./running_data/trg_embed")
    __save_tensor2csv(d, "./running_data/trg_position")

__global_running_transformer: Transformer = None
def wrap_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    x0, trg0, _ = batch[0]
    batch_size = len(batch)
    xseq_length = x0.shape[0]
    trgseq_length = trg0.shape[0]
    x, trg, y = zip(*batch)
    x = torch.stack(x, dim = 0)
    trg = torch.stack(trg, dim = 0)
    y = torch.stack(y, dim= 0)
    xword = __word_index_tensor(x)
    xpos = __position_tensor(x)
    trgword = __word_index_tensor(trg)
    trgpos = __position_tensor(trg)
    args = [ batch_size, xseq_length, trgseq_length, torch.stack([xword, xpos], dim = 0), torch.stack([trgword, trgpos], dim = 0), y ]
    if __global_running_transformer is None:
        return args
    else:
        print("yes")
        return __global_running_transformer.embedSrcAndTrg(*args)

device = "cuda" if torch.cuda.is_available() else "cpu"
if __name__ == '__main__':
    embed_grad = True
    dataset = CnEnDataset(EMBEDDING_SIZE, BATCH_SIZE)
    model = Transformer(dataset.en_tokens_count(), dataset.cn_tokens_count(), 
                        heads = 5, embedding_size = EMBEDDING_SIZE, expansion = 4,
                        dropout = 0.05, layers = 6, device = device, embed_grad = embed_grad)
    load_model(model)
    if not embed_grad:
        dataset.set_embed_matrics(*model.embedMatrics())

    if len(sys.argv) == 2 and sys.argv[1] == "-d":
        swords = dataset.en_tokens[0:800]
        __save_tensor2csv(dataset.get_src_word_distances(swords), "./running_data/src_word_distance.csv", swords, lambda v: v if v < 0.2 else "")

        twords = dataset.cn_tokens[0:800]
        __save_tensor2csv(dataset.get_trg_word_distances(twords), "./running_data/trg_word_distance.csv", twords, lambda v: v if v < 0.2 else "")

        __save_tensor2csv(dataset.get_src_pos_distances(list(range(100))), "./running_data/src_pos_distance.csv", list(range(100)), lambda v: v if True or v < 0.2 else "")
        __save_tensor2csv(dataset.get_trg_pos_distances(list(range(100))), "./running_data/trg_pos_distance.csv", list(range(100)), lambda v: v if True or v < 0.2 else "")
    elif len(sys.argv) == 1:
        loss_fn = nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
        optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE)
        # scheduler = None
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.995 ** (epoch + GONE_EPOCH) * LEARNING_RATE)
        print(f"learning rate: {LEARNING_RATE}, embedding size: {EMBEDDING_SIZE}, batch size: {BATCH_SIZE}")
        if not embed_grad:
            __global_running_transformer = model

        while True:
            try:
                train(dataset, model, loss_fn, optimizer, scheduler)
            except RuntimeError as e:
                if 'out of meomory' in str(e):
                    print("|Warning: out of memory")
                    for p in model.parameters():
                        if p.grad is not None:
                            del p.grad
                    torch.cuda.empty_cache()
                    BATCH_SIZE = BATCH_SIZE * 0.8
                    dataset.adjust_batch_size(BATCH_SIZE)
                else:
                    raise e
    else:
        ss = sys.argv[1:]
        sentence = ' '.join(ss)
        print(f"Origin: '{sentence}'")
        trans = translate(model, dataset, sentence)
        print(f"Target: '{trans}'")
