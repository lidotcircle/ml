from genericpath import exists
from sys import path
from typing import List, Tuple, Iterator, Set, Dict, Callable
import random
import io
import os
from pathlib import Path
import pathlib
import base64
import json
import csv
import math


__BOS = "<BOS>"
__EOS = "<EOS>"


def _process_lines(lines: List[str], fnumeric_csv: str, ftoken_list: str,
                                 tokenizer: Callable[[str], List[str]], ignoretrain: Set[int] = {}):
    tokens = { }
    token_list = [ __BOS, __EOS ]
    csv_lines = []
    length_info = {}
    for i in range(len(lines)):
        line = lines[i]
        csv_line = []
        for token in tokenizer(line):
            if token not in tokens:
                tokens[token] = len(token_list)
                token_list.append(token)
            csv_line.append(tokens[token])
        lk = len(csv_line)
        if lk not in length_info:
            length_info[lk] = 0
        length_info[lk] = length_info[lk] + 1
        if i not in ignoretrain:
            csv_lines.append(csv_line)

    with io.open(ftoken_list, "w", encoding="utf-8") as tokenfile:
        tokens_list = list(map(lambda v: base64.b64encode(v.encode("utf-8")).decode("utf-8"), token_list))
        tokenfile.write("\n".join(tokens_list))
    
    with io.open(fnumeric_csv, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_lines)


def _load_token_list(ftoken_list: str) -> List[str]:
    with io.open(ftoken_list, mode="r", encoding = "utf-8") as ftoken:
        tokens = ftoken.read().split("\n")
        return list(map(lambda v: base64.b64decode(v.encode("utf-8")).decode("utf-8"), tokens))


def _load_numeric_csv(csv_file: str) -> List[List[int]]:
    with io.open(csv_file, mode="r", encoding="utf-8") as en_ds:
        reader = csv.reader(en_ds)
        return [[int(char) for char in line] for line in reader if len(line) > 0]

class _ignore_blank_token():
    def __init__(self, f: Callable[[str], List[str]]):
        self.__f = f

    def __call__(self, s: str) -> List[str]:
        return list(filter(lambda v: v.strip() != '', self.__f(s)))


class SentencePairDataset():
    def __init__(self, workdir: str, source_sentence_file: str, target_sentence_file: str,
                 source_tokenizer: Callable[[str], List[str]], target_tokenizer: Callable[[str], List[str]],
                 valid_perc = 0.1):
        if not pathlib.Path(workdir).is_dir():
            os.mkdir(workdir)

        self.__data_is_loaded = False
        self.__source_sentence_file = source_sentence_file if os.path.isabs(source_sentence_file) \
            else os.path.join(workdir, source_sentence_file)
        self.__target_sentence_file = target_sentence_file if os.path.isabs(target_sentence_file) \
            else os.path.join(workdir, target_sentence_file)
        self.__f_source_csv_data = os.path.join(workdir, "dataset_" + Path(source_sentence_file).stem + ".csv")
        self.__f_target_csv_data = os.path.join(workdir, "dataset_" + Path(target_sentence_file).stem + ".csv")
        self.__f_source_token_data = os.path.join(workdir, "token_" + Path(source_sentence_file).stem + ".txt")
        self.__f_target_token_data = os.path.join(workdir, "token_" + Path(target_sentence_file).stem + ".txt")
        self.__valid_perc = valid_perc
        self.__source_tokenizer = _ignore_blank_token(source_tokenizer)
        self.__target_tokenizer = _ignore_blank_token(target_tokenizer)
        self.__f_validation_data = os.path.join(workdir, "validation_pair" + Path(source_sentence_file).stem + ".txt")
        self.__validationv = None
        self.__data_valid = False
        self.__load_token()

    def __generate_data(self):
        with io.open(self.__source_sentence_file, "r", encoding="utf-8") as sfile:
            slines = sfile.read().split("\n")
        with io.open(self.__target_sentence_file, "r", encoding="utf-8") as sfile:
            tlines = sfile.read().split("\n")
        assert len(slines) == len(tlines)

        vv = list(range(len(slines)))
        testidx = []
        for _ in range(math.floor(self.__valid_perc * len(slines))):
            d = random.randrange(len(vv))
            testidx.append(vv.pop(d))
        
        _process_lines(slines, self.__f_source_csv_data,
                        self.__f_source_token_data, self.__source_tokenizer,
                        testidx)
        _process_lines(tlines, self.__f_target_csv_data,
                        self.__f_target_token_data, self.__target_tokenizer,
                        testidx)
        self.__data_valid = True
        self.__load_token()
        validation_list = []
        tvtv = {}
        for i in testidx:
            sline = slines[i]
            tline = tlines[i]
            if sline not in tvtv:
                tvtv[sline] = []
            tvtv[sline].append(tline)

        for source_line in tvtv.keys():
            sline_tokens = self.__source_tokenizer(source_line)
            sline_val    = [ self.source_token2value(t) for t in sline_tokens ]
            target_lines = tvtv[source_line]
            target_lines_tokens = [ self.__target_tokenizer(tline) for tline in target_lines ]
            target_lines_val    = [ [ self.target_token2value(t) for t in line ] for line in target_lines_tokens]
            validation_list.append({
                "source": sline, 
                "source_tokens": sline_tokens,
                "source_val": sline_val,
                "targets": target_lines,
                "targets_tokens": target_lines_tokens,
                "targets_val": target_lines_val
                })
        with io.open(self.__f_validation_data, "w", encoding="utf-8") as vfile:
            vfile.write(json.dumps(validation_list))

    def validation_stuff(self) -> Dict:
        if self.__validationv != None:
            return self.__validationv
        
        fn = Path(self.__f_validation_data)
        if fn.is_file():
            self.__generate_data()

        with io.open(fn, "r", encoding="utf-8") as vfile:
            self.__validationv = json.loads(vfile.read())
            return self.__validationv

    def __is_data_valid(self):
        if self.__data_valid:
            return True
        last = os.path.getmtime(self.__source_sentence_file)
        last = max(os.path.getmtime(self.__target_sentence_file), last)
        if not os.path.exists(self.__f_source_csv_data) or \
           os.path.getmtime(self.__f_source_csv_data) < last or \
           not os.path.exists(self.__f_target_csv_data) or \
           os.path.getmtime(self.__f_target_csv_data) < last or \
           not os.path.exists(self.__f_source_token_data) or \
           os.path.getmtime(self.__f_source_token_data) < last or \
           not os.path.exists(self.__f_target_token_data) or \
           os.path.getmtime(self.__f_target_token_data) < last or \
           not os.path.exists(self.__f_validation_data) or \
           os.path.getmtime(self.__f_validation_data) < last:
            return False
        self.__data_valid = True
        return True

    def __load_token(self):
        if not self.__is_data_valid():
            self.__generate_data()
        self.__source_token: List[str] = _load_token_list(self.__f_source_token_data)
        self.__target_token: List[str] = _load_token_list(self.__f_target_token_data)
        self.__source_token_map = dict()
        self.__target_token_map = dict()
        for i in range(len(self.__source_token)):
            self.__source_token_map[self.__source_token[i]] = i
        for i in range(len(self.__target_token)):
            self.__target_token_map[self.__target_token[i]] = i

    def source_tokenizer2int(self, source: str) -> List[int]:
        return list(map(
            lambda v: self.__source_token_map[v], 
            self.__source_tokenizer(source)))

    def target_tokenizer2int(self, target: str) -> List[int]:
        return list(map(
            lambda v: self.__target_token_map[v], 
            self.__target_tokenizer(target)))

    def bos(self) -> int:
        return 0

    def eos(self) -> int:
        return 1
    
    def source_token_count(self):
        return len(self.__source_token)

    def target_token_count(self):
        return len(self.__target_token)

    def source_token2value(self, token: str) -> int:
        return self.__source_token_map[token]

    def target_token2value(self, token: str) -> int:
        return self.__target_token_map[token]

    def source_value2token(self, val: int) -> str:
        return self.__source_token[val]

    def target_value2token(self, val: int) -> str:
        return self.__target_token[val]

    def source_sentence2val(self, sentence: str) -> List[int]:
        return [ self.__source_token_map[t] for t in self.__source_tokenizer(sentence) ]

    def target_sentence2val(self, sentence: str) -> List[int]:
        return [ self.__target_token_map[t] for t in self.__target_tokenizer(sentence) ]

    def source_val2sentence(self, vals: List[int], joinv = ' '):
        tlist = [ self.__source_token[v] for v in vals ]
        return joinv.join(tlist)

    def target_val2sentence(self, vals: List[int], joinv = ' '):
        tlist = [ self.__target_token[v] for v in vals ]
        return joinv.join(tlist)

    def __load_csv_data(self):
        if self.__data_is_loaded:
            return
        if not self.__is_data_valid():
            self.__generate_data()
        self.__source_data = _load_numeric_csv(self.__f_source_csv_data)
        self.__target_data = _load_numeric_csv(self.__f_target_csv_data)
        assert len(self.__source_data) == len(self.__target_data)
        self.__data_is_loaded = True

    def __len__(self) -> int:
        self.__load_csv_data()
        return len(self.__source_data)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        self.__load_csv_data()
        return self.__source_data[idx], self.__target_data[idx]
