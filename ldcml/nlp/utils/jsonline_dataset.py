import io
import sys
import re
import json


valid_english_sentence = re.compile("^[a-zA-Z,&.\s\"0-9'?!]+$")
def __met(dataset: str, src_dataset: str, trg_dataset: str, n: int):
    try:
        ds = io.open(dataset, "r", encoding="utf-8")
        src = io.open(src_dataset, "w", encoding="utf-8")
        trg = io.open(trg_dataset, "w", encoding="utf-8")
        decoder = json.decoder.JSONDecoder()
        added = 0
        line = ds.readline()
        srcLines = []
        trgLines = []
        while line is not None and added < n:
            lo = decoder.decode(line)
            srcLine = lo["english"]
            trgLine = lo["chinese"]
            if valid_english_sentence.match(srcLine):
                srcLines.append(srcLine)
                trgLines.append(trgLine)
                added = added + 1
            line = ds.readline()
        src.write("\n".join(srcLines))
        trg.write("\n".join(trgLines))
    finally:
        ds.close()
        if src in locals():
            src.close()
        if trg in locals():
            trg.close()


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print(f"{sys.argv[0]} <dataset> <src_part> <trg_part> <max-n>")
        exit(1)
    else:
        __met(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
