#!/usr/bin/env python

from transformer import Transformer
from torchtext.data import get_tokenizer
from typing import List, Tuple, Generator

import torch
import torch.nn as nn


datas = open('../../datasets/cmn-eng/cmn.txt')
dataset: List[Tuple[str, str]] = list(
        map(
            lambda pair: (pair[0], pair[1]),
            filter(
                lambda pair: len(pair) == 2,
                map(
                    lambda data: data.split('\t')[0:2],
                    datas.read().split('\n')
                )
            )
        )
    )

en_tokenizer = get_tokenizer('basic_english', language = 'en')
en_word_to_idx = {}
en_idx_to_word = [ '' ]
cn_word_to_idx = {}
cn_idx_to_word = [ '' ]
en_sentence_max_len = 1
cn_sentence_max_len = 1
for pair in dataset:
    en_sentence_max_len = max(en_sentence_max_len, len(pair[0]))
    cn_sentence_max_len = max(cn_sentence_max_len, len(pair[1]))
    for voc in en_tokenizer(pair[0]):
        if not voc in en_word_to_idx:
            en_idx_to_word.append(voc)
            en_word_to_idx[voc] = len(en_idx_to_word)
    for v in pair[1]:
        if not v in cn_word_to_idx:
            cn_idx_to_word.append(v)
            cn_word_to_idx[v] = len(cn_idx_to_word)

en_idx_to_word.append("<BOS>")
cn_idx_to_word.append("<BOS>")
en_word_to_idx['<BOS>'] = len(en_idx_to_word)
cn_word_to_idx['<BOS>'] = len(cn_idx_to_word)
en_idx_to_word.append("<EOS>")
cn_idx_to_word.append("<EOS>")
en_word_to_idx['<EOS>'] = len(en_idx_to_word)
cn_word_to_idx['<EOS>'] = len(cn_idx_to_word)

en_embed = nn.Embedding(len(en_idx_to_word) + 1, 63)
cn_embed = nn.Embedding(len(cn_idx_to_word) + 1, 63)
# en_pos_embed = nn.Embedding(en_sentence_max_len, 1)
# cn_pos_embed = nn.Embedding(cn_sentence_max_len, 1)

BATCH_SIZE = 20
LEARNING_RATE = 0.002

def embed_a_case(case: Tuple[str, str]) -> Tuple[List, List[Tuple[List, List]]]:
    targets: List[Tuple[List, List]] = []
    src_words: List[str] = en_tokenizer(case[0])
    target_words: List[str] = list(map(lambda v: v, case[1]))

    src_embed = []
    for i in range(len(src_words)):
        val = src_words[i]
        embed = en_embed(torch.tensor(en_word_to_idx[val]))
        embed = torch.cat((embed, torch.tensor([i + 1])), 0)
        src_embed.append(embed.tolist())

    for i in range(len(target_words)):
        v = [ torch.cat((cn_embed(torch.tensor(cn_word_to_idx["<BOS>"])), torch.tensor([ 0 ])), 0).tolist() ]
        for j in range(i + 1):
            embed = cn_embed(torch.tensor(cn_word_to_idx[target_words[i]]))
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
            t = torch.tensor(cn_word_to_idx["<EOS>"])
            output.append(torch.cat((cn_embed(t), torch.tensor([ -1 ])), 0).tolist())
            targets.append((target, output))

    return src_embed, targets

def sort_cases_output(outputs: List[Tuple[List, List[Tuple[List, List]]]]) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    outputs.sort(key = lambda val: len(val[0]))
    store = {}
    for val in outputs:
        src_val = val[0]
        src_len = len(src_val)
        if not src_len in store:
            store[src_len] = {}
        store2 = store[src_len]
        for v2 in val[1]:
            trg_len = len(v2[0])
            assert len(v2[0]) == len(v2[1])
            if not trg_len in store2:
                store2[trg_len] = ([], [], [])
            store3 = store2[trg_len]
            store3[0].append(src_val)
            store3[1].append(v2[0])
            store3[2].append(v2[1])

    ans = []
    for k1 in store:
        for k2 in store[k1]:
            tp = store[k1][k2]
            ans.append((
                torch.tensor(tp[0]), 
                torch.tensor(tp[1]), 
                torch.tensor(tp[2])
            ))
    return ans


def make_batch(thelist: List[int]) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    return sort_cases_output(list(
        map(lambda idx: embed_a_case(dataset[idx]), thelist))
    )


def embededDataset() -> Generator[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], None, None]:
    start = 0
    end = len(dataset)
    while start <= end:
        thissize = min(end - start, BATCH_SIZE)
        yield make_batch(list(range(start, start + thissize)))
        start = start + thissize


device = "cuda" if torch.cuda.is_available() else "cpu"
def train(model: nn.Module, loss_fn, optimizer):
    i = 0
    for epcho in range(0, 100):
        j = 0
        for batch in embededDataset():
            for src, target, y in batch:
                src    = src.to(device)
                target = target.to(device)
                y      = y.to(device)

                pred = model(src, target)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                i = i + 1
                j = j + 1
                if i % 100 == 0:
                    print(f"device: {device}, epcho: {epcho}, batch: {j}, loss: {loss:>7f}")

def load_model():
    model = Transformer()
    model.load_state_dict(torch.load("model.pth"))
    return model

def save_model(model: nn.Module):
    torch.save(model.state_dict(), "model.pth")

if __name__ == '__main__':
    model = Transformer(heads = 8, embedding_size = 64, expansion = 4, dropout = 0.2, layers = 6)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE)
    train(model, loss_fn, optimizer)

