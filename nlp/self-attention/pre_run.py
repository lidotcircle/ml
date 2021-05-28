import nltk
import io
import csv
from typing import Tuple, List


__en_dataset = "./running_data/cmn-en.txt"
__cn_dataset = "./running_data/cmn-cn.txt"

__en_tokens = "./running_data/en_tokens"
__cn_tokens = "./running_data/cn_tokens"

__en_csv_dataset = "./running_data/en_dataset.csv"
__cn_csv_dataset = "./running_data/cn_dataset.csv"

__BOS = "<BOS>"
__EOS = "<EOS>"
BOS = 0
EOS = 1

# require download nltk punkt
def __process_en():
    tokens = { }
    token_list = [ __BOS, __EOS ]
    csv_lines = []
    with io.open(__en_dataset, mode="r", encoding="utf-8") as datafile:
        lines = datafile.read().split("\n")
        for line in lines:
            csv_line = []
            for token in nltk.word_tokenize(line):
                token = token.lower()
                if token not in tokens:
                    tokens[token] = len(token_list)
                    token_list.append(token)
                csv_line.append(tokens[token])
            csv_lines.append(csv_line)

    with io.open(__en_tokens, "w", encoding="utf-8") as tokenfile:
        for token in token_list:
            tokenfile.write(token + "\n")
    
    with io.open(__en_csv_dataset, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_lines)


def __process_cn():
    tokens = { }
    token_list = [ __BOS, __EOS ]
    csv_lines = []
    with io.open(__cn_dataset, mode="r", encoding="utf-8") as datafile:
        lines = datafile.read().split("\n")
        for line in lines:
            csv_line = []
            for token in line:
                if token.strip() == "":
                    continue
                if token not in tokens:
                    tokens[token] = len(token_list)
                    token_list.append(token)
                csv_line.append(tokens[token])
            csv_lines.append(csv_line)

    with io.open(__cn_tokens, "w", encoding="utf-8") as tokenfile:
        for token in token_list:
            tokenfile.write(token + "\n")
    
    with io.open(__cn_csv_dataset, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_lines)


def __process_data():
    __process_en()
    __process_cn()


def load_dataset(start: int = 0, end: int = -1) -> Tuple[List[List[int]], List[List[int]]]:
    with io.open(__en_csv_dataset, mode="r", encoding="utf-8") as en_ds:
        reader = csv.reader(en_ds)
        en_sentences = [[int(char) for char in line] for line in reader if len(line) > 0]

    with io.open(__cn_csv_dataset, mode="r", encoding="utf-8") as cn_ds:
        reader = csv.reader(cn_ds)
        cn_sentences = [[int(char) for char in line] for line in reader if len(line) > 0]

    assert len(en_sentences) == len(cn_sentences)
    return en_sentences[start:end], cn_sentences[start:end]


def load_tokens() -> Tuple[List[str], List[str]]:
    with io.open(__en_tokens, mode="r", encoding = "utf-8") as en_tokens:
        en_lines = en_tokens.read().split("\n")[0:-1]
    with io.open(__cn_tokens, mode="r", encoding = "utf-8") as cn_tokens:
        cn_lines = cn_tokens.read().split("\n")[0:-1]
    return en_lines, cn_lines


def __get_en_token_map():
    __en_token_map = {}
    en_lines, _ = load_tokens()
    for i in range(len(en_lines)):
        __en_token_map[en_lines[i]] = i
    return __en_token_map


def en_tokenizer(sentence: str) -> List[int]:
    __en_token_map = __get_en_token_map()
    return list(map(
        lambda word: __en_token_map[word.lower()],
        nltk.word_tokenize(sentence)
    ))


def iseos(sent: List[int]):
    return sent[-1] == EOS

def insertBOS(sent: List[int]):
    sent.insert(0, BOS)
    return sent


if __name__ == "__main__":
    __process_data()
