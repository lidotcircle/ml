import torch
import math
import random
import pre_run
from typing import List, Tuple, Iterator


class TextClassificationDataset():
    def __init__(self, train_perc: float = 0.9):
        self.__dataset = pre_run.load_dataset()
        random.shuffle(self.__dataset)
        self.__train = True
        self.__train_len = math.floor(train_perc * len(self.__dataset))
        _, self.__tokens = pre_run.load_labels_and_tokens()

    def toggleEvalMode(self):
        self.__train = not self.__train

    def __len__(self):
        if self.__train:
            return self.__train_len
        else:
            return len(self.__dataset) - self.__trian_len

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        if index < 0:
            index = index + len(self)
        if index > len(self):
            raise "out of range"
        if not self.__train:
            index = index + self.__train
        item: List[int] = self.__dataset[index]
        item.pop(0)
        size = len(self.__tokens) + 1
        label = item[0]
        return self.__list2tensor(item, size), torch.tensor(label)

    # z0 z1 .... zn
    def __list2tensor(self, vallist: List[int], size: int):
        l1 = []
        l2 = []
        indices = [l1, l2]
        values = []
        for i in range(1, len(vallist) + 1):
            l1.append(i)
            l1.append(i)
            l2.append(vallist[i - 1])
            l2.append(size - 1)
            values.append(1)
            values.append(i)
        return torch.sparse_coo_tensor(indices, values, (len(vallist) + 1, size)).float().to_dense()

    def sampler(self, batch_size: int, epoch: int, shuffle: bool) -> Iterator[List[int]]:
        base = 0 if self.__train else self.__train_len
        ivv = { }
        for i in range(len(self)):
            sentence = self.__dataset[base + i]
            l = len(sentence)
            if l not in ivv:
                ivv[l] = []
            ivv[l].append(i)
        samelist = [ ]
        for key in ivv.keys():
            samelist.append(ivv[key])
        for _ in range(epoch):
            bl = list(samelist)
            if shuffle:
                random.shuffle(bl)
            while len(bl) > 0:
                f: List[int] = bl[0]
                if len(f) <= batch_size:
                    bl.pop(0)
                else:
                    bl[0] = f[batch_size:]
                    f = f[0:batch_size]
                yield f
