#!/usr/bin/env python

from transformer import Transformer
from dataset import CnEnDataset
import pre_run
from typing import List, Tuple, Generator, Iterator
from pathlib import Path
import sys
import math

import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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
    for idx in dataloader:
        x, trg, y = dataloader.dataset.idx2tensor(idx)
        x = x.to(device)
        trg = trg.to(device)
        y = y.to(device)
        # print(f"src: <{dataset.en_tensor2sentence(x[0])}>, trg: <{dataset.cn_tensor2sentence(trg[0])}>, y: <{dataset.cn_scalar2word(y[0][-1])}>")
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

def translate(model: nn.Module, dataset: CnEnDataset, eng_sentence: str) -> str:
    src = pre_run.en_tokenizer(eng_sentence)
    src = dataset.embed_x(src).unsqueeze(0).to_dense().to(device)
    eng_len = src.shape[1]
    trg_list = []
    while len(trg_list) == 0 or trg_list[-1] != dataset.cn_eos():
        trg = dataset.embed_trg(trg_list).unsqueeze(0).to_dense().to(device)
        pred = model(src, trg)
        pred = torch.softmax(pred, dim = 2).squeeze(0).tolist()
        pred = pred[-1]
        y = list_max_index(pred)
        trg_list.append(y)
        if len(trg_list) > eng_len * 20:
            print(f"maybe translate fail!!! '{eng_sentence}'")
            break
    trg_list.pop()
    return dataset.cn_idx_list2sentence(trg_list)

device = "cuda" if torch.cuda.is_available() else "cpu"
if __name__ == '__main__':
    dataset = CnEnDataset(EMBEDDING_SIZE, BATCH_SIZE)
    # non-leaf tensor can't cross process boundary
    # crash when num_workers > 4 in windows "OSError: (WinError 1455) The paging file is too small for this operation to complete"
    dataloader = DataLoader(dataset, batch_sampler=dataset.batchSampler(TRAIN_EPCHO, suffle=True), num_workers=2)

    model = Transformer(dataset.en_tokens_count(), dataset.cn_tokens_count(), 
                        heads = 5, embedding_size = EMBEDDING_SIZE, expansion = 4,
                        dropout = 0.08, layers = 6, device = device)
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
