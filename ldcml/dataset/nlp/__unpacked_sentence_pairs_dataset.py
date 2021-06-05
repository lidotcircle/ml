from typing import Callable, List, Tuple, Iterator
from .__sentence_pair_dataset import SentencePairDataset
import random


class UnpackedSentencePairDataset(SentencePairDataset):
    def __init__(self, workdir: str, source_sentence_file: str, target_sentence_file: str,
                 source_tokenizer: Callable[[str], List[int]],  target_tokenizer: Callable[[str], List[int]]):
        super().__init__(workdir, source_sentence_file, target_sentence_file, source_tokenizer, target_tokenizer)
        self.__len_accu = []
        for i in range(super().__len__()):
            l = 0
            if len(self.__len_accu) > 0:
                l = self.__len_accu[-1]
            self.__len_accu.append(l + len(super().__getitem__(i)[1]) + 1)
        self.__len = self.__len_accu[-1]

    def __pair_idx2idx(self, l1: int, l2: int) -> int:
        return self.__len_accu[l1] - len(super().__getitem__(l1)[1]) - 1 + l2

    def __idx2pair_idx(self, idx: int) -> Tuple[int, int]:
        min = 0
        max = len(self.__len_accu)
        while max > min:
            avg = (min + max) // 2
            high = self.__len_accu[avg]
            lv = len(super().__getitem__(avg)[1])
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
        return self.__len

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int], List[int]]:
        if idx < 0:
            idx = len(self) + idx
        if idx >= len(self):
            raise RuntimeError("out of range")
        l1, l2 = self.__idx2pair_idx(idx)
        x, y = super().__getitem__(l1)
        assert 0 <= l2 <= len(y)
        isend = l2 == len(y)
        y   = y[0:l2 + 1]
        trg = list(y)
        trg.insert(0, self.bos())
        if isend:
            y.append(self.eos())
        else:
            trg.pop()
        return x, trg, y

    def batchSampler(self, batch_size: int, epcho: int, suffle: bool = True) -> Iterator[List[int]]:
        store = {}
        for i in range(super().__len__()):
            x, y = super().__getitem__(i)
            for j in range(len(y) + 1):
                k = f"{len(x)}-{j}"
                if k not in store:
                    store[k] = []
                store[k].append(self.__pair_idx2idx(i, j))
        same_shape = list(store.values())
        batchs = []
        for ss in same_shape:
            while len(ss) > 0:
                batchs.append(ss[0:batch_size])
                ss = ss[batch_size:]
        while epcho > 0:
            epcho = epcho - 1
            listme = list(range(len(batchs)))
            while len(listme) > 0:
                n = 0 if not suffle else random.randrange(0, len(listme))
                n = listme.pop(n)
                yield batchs[n]
