import pandas as pd
import numpy as np
import re

keys = ['Id', 'Form', 'Lemma', 'UPosTag', 'XPosTagA', 'Feats', 'Head', 'DepRel', 'Deps', 'Misc']
prefix_len = len('# text = ')


def read_conllu(path, start = 0, end = -1):
    if end > 0 and start - end >= 0 : return
    file = open(path, encoding='utf-8')
    sentences = []
    all_rows = []
    i = 0
    keep = True
    for line in file:
        s_rows = []
        if i < start:
            for l in file:
                if l == '\n' or l == '\n\r':
                    break
        else:
            if i >= end : break
            row = []
            sentences.append(line[prefix_len:])
            for l in file:
                if l == '\n' or l == '\n\r' :break
                cells = l[:-1].split('\t')
                row = [i] + [v for v in cells]
                if row[4] == 'NOUN' or row[4] == 'VERB':
                    if re.match(r'\W', row[3], flags = re.UNICODE) :
                        keep = False
                s_rows.append(row)
        if keep: all_rows += s_rows
        else :
            sentences = sentences[:-1]
            keep = True
        i+=1
    frame = pd.DataFrame(data = all_rows)
    frame.columns = ['SId'] + keys
    frame.set_index(['SId', 'Id'], inplace = True)
    return frame, sentences

def read_conllu_from(file, batch_size):
    sentences = []
    all_rows = []
    i = 0
    keep = True
    for line in file:
        s_rows = []
        if i >= batch_size : break
        row = []
        sentences.append(line[prefix_len:])
        for l in file:
            if l == '\n' or l == '\n\r' :break
            cells = l[:-1].split('\t')
            row = [i] + [v for v in cells]
            if row[4] == 'NOUN' or row[4] == 'VERB':
                if re.match(r'\W', row[3], flags = re.UNICODE) :
                    keep = False
            s_rows.append(row)

        if keep: all_rows += s_rows
        else :
            sentences = sentences[:-1]
            keep = True
        i+=1
    frame = pd.DataFrame(data = all_rows)
    frame.columns = ['SId'] + keys
    frame.set_index(['SId', 'Id'], inplace = True)
    return frame, sentences, file
