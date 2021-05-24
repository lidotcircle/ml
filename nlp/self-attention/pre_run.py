import nltk
import numpy
import io
import sys
import pathlib
import csv


en_dataset = "./running_data/en_dataset"
cn_dataset = "./running_data/cn_dataset"
en_tokens = "./running_data/en_tokens"
cn_tokens = "./running_data/cn_tokens"

en_csv_dataset = "./running_data/en_dataset.csv"
cn_csv_dataset = "./running_data/cn_dataset.csv"


# require download nltk punkt
def process_en():
    tokens = { }
    token_list = [ "" ]
    csv_lines = []
    with io.open(en_dataset, mode="r", encoding="utf-8") as datafile:
        lines = datafile.readlines()
        for line in lines:
            csv_line = []
            for token in nltk.word_tokenize(line):
                if token not in tokens:
                    tokens[token] = len(token_list)
                    token_list.append(token)
                csv_line.append(tokens[token])
            csv_lines.append(csv_line)

    with io.open(en_tokens, "w", encoding="utf-8") as tokenfile:
        for token in token_list:
            tokenfile.write(token + "\n")
    
    with io.open(en_csv_dataset, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_lines)

def process_cn():
    tokens = { }
    token_list = [ "" ]
    csv_lines = []
    with io.open(cn_dataset, mode="r", encoding="utf-8") as datafile:
        lines = datafile.readlines()
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

    with io.open(cn_tokens, "w", encoding="utf-8") as tokenfile:
        for token in token_list:
            tokenfile.write(token + "\n")
    
    with io.open(cn_csv_dataset, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_lines)

def process_data():
    process_en()
    process_cn()

if __name__ == "__main__":
    process_data()
