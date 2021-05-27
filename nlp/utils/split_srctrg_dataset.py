import io
import sys
import re


valid_english_sentence = re.compile("^[a-zA-Z,&.\s]+$")
def split_dataset(dataset: str, src_dataset: str, trg_dataset: str):
    try:
        ds = io.open(dataset, "r", encoding="utf-8")
        src = io.open(src_dataset, "w", encoding="utf-8")
        trg = io.open(trg_dataset, "w", encoding="utf-8")
        lines = ds.readlines()
        dslen = len(lines) // 2
        for i in range(dslen):
            cni = i * 2
            eni = cni + 1
            ens = lines[eni]
            cns = lines[cni]
            if valid_english_sentence.match(ens):
                src.write(cns)
                trg.write(ens)
    finally:
        ds.close()
        if src in locals():
            src.close()
        if trg in locals():
            trg.close()


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f"{sys.argv[0]} <dataset> <src_part> <trg_part>")
        exit(1)
    else:
        split_dataset(sys.argv[1], sys.argv[2], sys.argv[3])
