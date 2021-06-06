#!/usr/bin/env python

from ....nlp.model.transformer import Transformer, generate_batch
from ....utils import weighted_avg_and_std
from ....dataset.nlp import UnpackedSentencePairDataset
from ....utils.logger import CsvLogger

from torchtext.data.metrics import bleu_score
from typing import List, Dict
from pathlib import Path
import signal
import json
import csv
import io
import os
import sys
import time

import numpy
import torch
import torch.nn as nn
import nltk
from torch.utils.data import DataLoader


BATCH_SIZE = 300
LEARNING_RATE = 0.06
TRAIN_EPCHO = 1000
GONE_EPOCH = 0
EMBEDDING_SIZE = 100


# can't pickle lambda
def cn_tokenizer(v):
    return list(v)

def collate_fn_wrap(batch):
    return generate_batch(*zip(*batch))

class TranslationSession():
    def __init__(self, device: str = 'cpu'):
        self.gone_epoch = 0
        self.sample_count = 0
        self.current_best_loss = -1
        self.workdir = os.path.dirname(__file__)
        self.dataset = UnpackedSentencePairDataset(os.path.join(self.workdir, "running_data"), 
                                            "source.txt", "target.txt", 
                                            nltk.word_tokenize, cn_tokenizer)

        embed_grad = True
        heads = 5
        embedding_size = EMBEDDING_SIZE
        expansion_rate = 4
        dropout = 0.05
        layers = 6
        self.__model_version = f"h{heads}-embedding{embedding_size}-expansion{expansion_rate}-dropout{dropout}-layers{layers}"
        self.model = Transformer(self.dataset.source_token_count(), self.dataset.target_token_count(), 
                            heads = heads, embedding_size = embedding_size, expansion = expansion_rate,
                            dropout = dropout, layers = layers, device = device, embed_grad = embed_grad).to(device)
        self.load_model()
        self.__dumpme = Path(f"session{self.__model_version}.log")
        if self.__dumpme.is_file():
            with io.open(self.__dumpme, "r", encoding="utf-8") as dff:
                obj = json.loads(dff.read())
                self.gone_epoch = obj["gone_epoch"]
                self.sample_count = obj["sample_count"]
                self.current_best_loss = obj["current_best_loss"]

        self.logger = CsvLogger(columns=["epoch", "batch size", "loss"], filename=f"lossinfo{self.__model_version}.csv")
        self.bleu_logger = CsvLogger(columns=["epoch", "bleu"], filename=f"bleuinfo{self.__model_version}.csv")
        self.len_limit = [ 1, 4, 8, 8, 10, 10, 12, 14, 16 ]
        self.len_limit = self.len_limit + [ 1.3 * (i + len(self.len_limit)) for i in range(len(self.len_limit), 1000) ]

    def train(self, epoch: int):
        def signal_int_handler(signo, stkf):
            self.save_model()
            exit(0)
        origin_handler = signal.signal(signal.SIGINT, signal_int_handler)

        loss_fn   = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr = LEARNING_RATE, momentum=0.02)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.995 ** (epoch + GONE_EPOCH) * LEARNING_RATE)
        dataloader = DataLoader(self.dataset,
                                batch_sampler=self.dataset.batchSampler(BATCH_SIZE, epoch, suffle=True), 
                                collate_fn=collate_fn_wrap, 
                                num_workers=2)
        loss_list = []
        loss_weight = []
        general_loss_list = []
        dataset_len = len(self.dataset)
        last_save = time.time()
        for current_batch_size, y, args in dataloader:
            pred = self.model(*args)
            pred = pred.reshape(pred.shape[0] * pred.shape[1], pred.shape[2])
            y = y.reshape(y.shape[0] * y.shape[1])
            y = y.to(device)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.sample_count = self.sample_count + current_batch_size
            epoch_finished = self.sample_count % dataset_len == 0
            loss_list.append(float(loss))
            loss_weight.append(current_batch_size)
            self.logger.info(self.gone_epoch, current_batch_size, float(loss))

            if (time.time() - last_save) > 30 * 60:
                last_save = time.time()
                self.save_model()

            if epoch_finished or len(loss_list) == 100:
                loss_mean, loss_std = weighted_avg_and_std(loss_list, loss_weight)
                loss_list = []
                loss_weight = []
                general_loss_list.append(loss_mean)
                print(f"device: {device}, epoch: {self.gone_epoch}, total sample: {self.sample_count}, loss mean: {loss_mean:>7f}, loss_std: {loss_std:>7f}")
            if epoch_finished:
                self.gone_epoch = self.gone_epoch + 1
                loss_mean = numpy.mean(general_loss_list)
                general_loss_list = []
                print(f"loss mean: {loss_mean}")
                if self.current_best_loss < 0:
                    self.current_best_loss = loss_mean
                if loss_mean < self.current_best_loss:
                    print(f"save current best")
                    self.save_model(loss_mean)
                    self.current_best_loss = loss_mean
                if scheduler is not None:
                    scheduler.step()
        signal.signal(signal.SIGINT, origin_handler)

    def load_model(self):
        file = Path(os.path.join(self.workdir, "saved_model", f"model{self.__model_version}.pth"))
        if file.is_file():
            print("load model")
            self.model.load_state_dict(torch.load(file))

    def save_model(self, postfix: str = ''):
        print("save model")
        fn = os.path.join(self.workdir, "saved_model", f"model{self.__model_version}{postfix}.pth")
        fndir = os.path.dirname(fn)
        if not Path(fndir).is_dir():
            os.mkdir(fndir)
        torch.save(self.model.state_dict(), fn)
        obj = {}
        obj["sample_count"] = self.sample_count
        obj["gone_epoch"]   = self.gone_epoch
        obj["current_best_loss"]   = self.current_best_loss
        with io.open(self.__dumpme, "w", encoding="utf-8") as ddf:
            ddf.write(json.dumps(obj))

    def __eval_multiple(self, sentences: List[List[int]], batch_size: int, silent: bool = False) -> List[List[int]]:
        bos = self.dataset.bos()
        eos = self.dataset.eos()
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
            if not silent and (finished_perc - prev_perc) > 5:
                print(f"finish {finished_perc:>.2f}%")
                prev_perc = finished_perc

            m, n = queue.pop()
            li = indecis[m][n]
            indecis[m][n] = li[batch_size:]
            if len(li) > batch_size:
                queue.add((m, n))
            batchli = li[0:batch_size]
            size = len(batchli)
            src    = torch.Tensor(3, m * size)
            srcpos = torch.Tensor(3, m * size)
            trg    = torch.Tensor(3, n * size)
            trgpos = torch.Tensor(3, n * size)
            for i in range(size):
                s, t = sentences_with_trg[batchli[i]]
                for j in range(m):
                    src[0][i * m + j] = i * m + j
                    src[1][i * m + j] = s[j]
                    src[2][i * m + j] = 1
                    srcpos[0][i * m + j] = i * m + j
                    srcpos[1][i * m + j] = j
                    srcpos[2][i * m + j] = 1
                for j in range(len(t)):
                    trg[0][i * n + j] = i * n + j
                    trg[1][i * n + j] = t[j]
                    trg[2][i * n + j] = 1
                    trgpos[0][i * n + j] = i * n + j
                    trgpos[1][i * n + j] = j
                    trgpos[2][i * n + j] = 1
            pred = self.model(size, src, srcpos, trg, trgpos)
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
                    sentence.pop()
                else:
                    sentence.insert(0, bos)
                    o.append(sentence[-1])
                sentences_with_trg[batchli[i]] = (s, o)
                if not finish and len(sentence) < self.len_limit[len(s)]:
                    append_idx(batchli[i])
                else:
                    finished_sentence = finished_sentence + 1
                    o.pop(0)

        return [ v for _, v in sentences_with_trg ]

    def eval_bleu(self, test_cases: Dict = None, silent: bool = False):
        if test_cases is None:
            test_cases = self.dataset.validation_stuff()
        src_sentences: List[List[int]] = [ pair["source_val"] for pair in test_cases ]
        candidates = self.__eval_multiple(src_sentences, 300, silent)
        candidates = [ [ self.dataset.target_value2token(v) for v in s ] for s in candidates ]
        sentences_refs = [ pair["targets_tokens"] for pair in test_cases ]
        return bleu_score(candidates, sentences_refs)

    def train_and_bleu(self, epoch: int, interval: int):
        gone = 0
        best_bleu = 0
        while gone < epoch:
            rv = min(interval, epoch - gone)
            self.train(rv)
            gone = gone + rv
            print("eval bleu")
            bleu = self.eval_bleu(None, True)
            print(f"epoch: {gone}, bleu: {bleu}")
            self.bleu_logger.info(gone, bleu)
            if bleu > best_bleu:
                self.save_model(f"bleu{bleu}")
                best_bleu = bleu

    def translate(self, sentences: List[str], joinv: str = '') -> List[str]:
        tokenized_sentence = [ self.dataset.source_sentence2val(sentence) for sentence in sentences ]
        ooo = self.__eval_multiple(tokenized_sentence, 300, True)
        vvv = [ self.dataset.target_val2sentence(l, joinv) for l in ooo ]
        return vvv

    def translate_and_print(self, sentences: List[str], joinv: str = ''):
        vvv = self.translate(sentences, joinv)
        maxlen = 0
        for s in sentences:
            maxlen = max(maxlen, len(s))
        for i in range(len(sentences)):
            sentences[i] = sentences[i] + " " * (maxlen - len(sentences[i]))
        for ss, rr in zip(sentences, vvv):
            print(f"['{ss}' => '{rr}']")


def __save_tensor2csv(t: torch.Tensor, fn: str, header: List[str] = None, vmap = None):
    assert len(t.shape) == 2
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


device = "cuda" if torch.cuda.is_available() else "cpu"
if __name__ == '__main__':
    session = TranslationSession(device)

    if len(sys.argv) == 1:
        print(f"learning rate: {LEARNING_RATE}, embedding size: {EMBEDDING_SIZE}, batch size: {BATCH_SIZE}")

        while True:
            try:
                session.train_and_bleu(TRAIN_EPCHO, 1)
            except RuntimeError as e:
                if 'out of meomory' in str(e):
                    print("|Warning: out of memory")
                    for p in session.model.parameters():
                        if p.grad is not None:
                            del p.grad
                    torch.cuda.empty_cache()
                    BATCH_SIZE = BATCH_SIZE * 0.8
                    session.dataset.adjust_batch_size(BATCH_SIZE)
                else:
                    raise e
    elif len(sys.argv) == 2 and sys.argv[1] == "-e":
        print("Evaluating BLEU Score")
        score = session.eval_bleu()
        print(f"bleu score: {score}")
    elif len(sys.argv) > 1 and sys.argv[1] == "-t":
        assert len(sys.argv) > 2
        session.translate_and_print(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "-f":
        assert len(sys.argv) > 2
        with io.open(sys.argv[2], "r", encoding="utf-8") as sourcetext:
            lines = sourcetext.read().split("\n")
            session.translate_and_print(lines)
    else:
        print(f"usage: {sys.argv[0]} [-etf] <options>")
        exit(1)
