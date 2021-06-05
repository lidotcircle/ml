import io
import sys
import os
import pathlib
import csv
import base64
from typing import List, Tuple

__cdir = os.path.dirname(os.path.abspath(__file__))
__token_array = os.path.join(__cdir, "running_data/tokens.txt")
__label_array = os.path.join(__cdir, "running_data/labels.txt")
__dataset = os.path.join(__cdir, "running_data/classifiction_dataset.csv")


def token_array():
    return __token_array

def label_array():
    return __label_array


def save_labels_and_tokens(labels: List[str], tokens: List[str]):
    with io.open(__token_array, "w", encoding="utf-8") as f:
        tokens = list(map(lambda v: base64.b64encode(v.encode("utf-8")).decode("utf-8"), tokens))
        f.write("\n".join(tokens))
    with io.open(__label_array, "w", encoding="utf-8") as f:
        f.write("\n".join(labels))


def load_labels_and_tokens() -> Tuple[List[str], List[str]]:
    labels = []
    tokens = []
    with io.open(__token_array, "r", encoding="utf-8") as f:
        tokens = f.read().split("\n")
        tokens = list(map(lambda v: base64.b64decode(v.encode("utf-8")).decode("utf-8"), tokens))
    with io.open(__label_array, "r", encoding="utf-8") as f:
        labels = f.read().split("\n")
    return labels, tokens


def add_sample(csv_writer: csv.DictWriter, tokens: List[str], labels: List[str], label: str, sentence: str):
    if not label in labels:
        labels.append(label)
    label_idx = labels.index(label) + 1
    line = [ label_idx ]
    for token in sentence:
        if not token in tokens:
            tokens.append(token)
        t = tokens.index(token)
        line.append(t)
    csv_writer.writerow(line)


def process_data(label_directories: List[str], max_size: int = -1) -> Tuple[List[str], List[str]]:
    dsf = io.open(__dataset, "w", encoding="utf-8", newline='')
    writer = csv.writer(dsf)
    labels = []
    tokens = []
    for labeldir in label_directories:
        dir = pathlib.Path(labeldir)
        label = dir.name
        print(f"label: {label}")
        n = 0
        for fl in os.listdir(dir):
            if max_size > 0 and n >= max_size:
                break
            n = n + 1
            fl = pathlib.Path(os.path.join(dir, fl))
            if not fl.is_file():
                continue
            fh = io.open(fl, "r", encoding = "utf-8")
            add_sample(writer, tokens, labels, label, fh.read())
    dsf.close()
    return labels, tokens


def load_dataset() -> List[List[int]]:
    with io.open(__dataset, "r", encoding="utf-8") as fl:
        reader = csv.reader(fl)
        return [[int(char) for char in line] for line in reader if len(line) > 0]


if __name__ == "__main__":
    n = 5000
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    labels, tokens = process_data(list(map(lambda v: "./running_data/" + v, ["财经", "彩票", "房产", "股票", "家居", "教育", "科技", "社会", "时尚"])), n)
    save_labels_and_tokens(labels, tokens)
