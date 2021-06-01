#!/usr/bin/env python

from transformer import Transformer
from torchtext.data.metrics import bleu_score
from dataset import CnEnDataset
import pre_run
from typing import List, Tuple, Generator, Iterator
from pathlib import Path
import csv
import io
import sys
import time
import math
import json

import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


BATCH_SIZE = 300
LEARNING_RATE = 0.06
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

def translate(model: nn.Module, dataset: CnEnDataset, src_sentence: str, silent: bool = False) -> str:
    src = pre_run.en_tokenizer(src_sentence)
    src = dataset.embed_x(pre_run.en_tokenizer(src_sentence))
    eng_len = src.shape[0]
    trg_list = []
    while len(trg_list) == 0 or trg_list[-1] != dataset.cn_eos():
        trg = dataset.embed_trg(trg_list)
        batch_size, xseq_length, trgseq_length, x, trg, _ = wrap_collate_fn([(src, trg, torch.tensor([]))])
        pred = model(batch_size, xseq_length, trgseq_length, x, trg)
        pred = torch.softmax(pred, dim = 2)
        y = torch.argmax(pred[0][-1])
        trg_list.append(y)
        if len(trg_list) > eng_len * 20:
            if not silent:
                print(f"maybe translate fail!!! '{src_sentence}'")
            break
    trg_list.pop()
    return dataset.cn_idx_list2sentence(trg_list)

len_limit = [ 1, 4, 8, 8, 10, 10, 12, 14, 16 ]
len_limit = len_limit + [ 1.3 * (i + len(len_limit)) for i in range(len(len_limit), 1000) ]
def __eval_multiple(model: nn.Module, sentences: List[List[int]], bos: int, eos: int, batch_size: int) -> List[List[int]]:
    sentences_with_trg = [ (sentence, [ bos ]) for sentence in sentences ]
    queue = set()
    indecis = { }
    finished_sentence = 0
    def append_idx(i: int):
        m, n = sentences_with_trg[i]
        m, n = len(m), len(n)
        if m not in indecis:
            indecis[m] = {}
        if n not in indecis[m]:
            indecis[m][n] = []
        indecis[m][n].append(i)
        queue.add((m, n))

    for i in range(len(sentences_with_trg)):
        append_idx(i)

    prev_perc = 0
    while len(queue) > 0:
        finished_perc = finished_sentence / len(sentences_with_trg) * 100
        if (finished_perc - prev_perc) > 5:
            print(f"finish {finished_perc:<2}%")
            prev_perc = finished_perc

        m, n = queue.pop()
        li = indecis[m][n]
        indecis[m][n] = li[batch_size:-1]
        if len(li) > batch_size:
            queue.add((m, n))
        batchli = li[0:batch_size]
        size = len(batchli)
        src = torch.Tensor(2, 3, m * size)
        trg = torch.Tensor(2, 3, n * size)
        for i in range(size):
            s, t = sentences_with_trg[batchli[i]]
            for j in range(m):
                src[0][0][i * m + j] = i * m + j
                src[0][1][i * m + j] = s[j]
                src[0][2][i * m + j] = 1
                src[1][0][i * m + j] = i * m + j
                src[1][1][i * m + j] = j
                src[1][2][i * m + j] = 1
            for j in range(len(t)):
                trg[0][0][i * n + j] = i * n + j
                trg[0][1][i * n + j] = t[j]
                trg[0][2][i * n + j] = 1
                trg[1][0][i * n + j] = i * n + j
                trg[1][1][i * n + j] = j
                trg[1][2][i * n + j] = 1
        # ignore softmax
        pred = model(size, m, n, src, trg)
        try:
            sss = torch.argmax(pred, dim=2)
        except RuntimeError as e:
            for idx in batchli:
                _, s = sentences_with_trg[idx]
                s.pop(0)
                sentences_with_trg[idx] = ([], s)
            print("catch a error: ", e)

        for i in range(size):
            s, o = sentences_with_trg[batchli[i]]
            sentence = list(sss[i])
            finish = sentence[-1] == eos
            if finish:
                finished_sentence = finished_sentence + 1
                sentence.pop()
            else:
                sentence.insert(0, bos)
                o.append(sentence[-1])
            sentences_with_trg[batchli[i]] = (s, o)
            if not finish and len(sentence) < len_limit[len(s)]:
                append_idx(batchli[i])
            else:
                o.pop(0)

    return [ v for _, v in sentences_with_trg ]

def eval_bleu(model: nn.Module, dataset: CnEnDataset, test_cases: List[Tuple[str, List[str]]]):
    src_sentences = [ src for _, _, src, _, _, _ in test_cases ]
    candidates = __eval_multiple(model, src_sentences, dataset.cn_bos(), dataset.cn_eos(), 300)
    candidates = [ [ dataset.cn_tokens[v] for v in s ] for s in candidates ]
    sentences_refs = [ refs for _, _, _, _, refs, _ in test_cases ]
    score = bleu_score(candidates, sentences_refs)
    print(f"bleu score: {score}")
    return score


def translate_multiple(model: nn.Module, sentences: List[str], bos: int, eos: int, batch_size: int) -> List[List[int]]:
    _, cn_tokens = pre_run.load_tokens()
    tokenized_sentence = [ pre_run.en_tokenizer(sentence) for sentence in sentences ]
    ooo = __eval_multiple(model, tokenized_sentence, bos, eos, batch_size)
    vvv = [ "".join([ cn_tokens[i] for i in l ]) for l in ooo ]
    maxlen = 0
    for s in sentences:
        maxlen = max(maxlen, len(s))
    for i in range(len(sentences)):
        sentences[i] = sentences[i] + " " * (maxlen - len(sentences[i]))
    for ss, rr in zip(sentences, vvv):
        print(f"['{ss}' => '{rr}']")


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
                        dropout = 0.12, layers = 6, device = device, embed_grad = embed_grad)
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
    elif len(sys.argv) == 2 and sys.argv[1] == "-e":
        print("Evaluating BLEU Score")
        testcases = pre_run.load_testcases()
        eval_bleu(model, dataset, testcases)
    elif len(sys.argv) > 1 and sys.argv[1] == "-t":
        assert len(sys.argv) > 2
        translate_multiple(model, sys.argv[2:], dataset.cn_bos(), dataset.cn_eos(), 300)
    elif len(sys.argv) > 1 and sys.argv[1] == "-f":
        assert len(sys.argv) > 2
        with io.open(sys.argv[2], "r", encoding="utf-8") as sourcetext:
            lines = sourcetext.read().split("\n")
            translate_multiple(model, lines, dataset.cn_bos(), dataset.cn_eos(), 300)
    elif len(sys.argv) == 1:
        loss_fn = nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
        optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE, momentum=0.02)
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
