import pandas as pd
import numpy as np

keys = ['Id', 'Form', 'Lemma', 'UPosTag', 'XPosTagA', 'Feats', 'Head', 'DepRel', 'Deps', 'Misc']
prefix_len = len('# text = ')

def read_conllu(path, start = 0, end = -1):
    if end > 0 and start - end >= 0 : return
    file = open(path, encoding='utf-8')
    sentences = []
    a = []
    i = 0
    for line in file:
        if i < start:
            for l in file:
                if l == '\n' or l == '\n\r':
                    break
        else:
            if i >= end : break
            b = []
            s = line[prefix_len:]
            sentences.append(s)
            for l in file:
                if l == '\n' or l == '\n\r' :break
                cells = l[:-1].split('\t')
                b = [i] + [v for v in cells]
                a.append(b)
        i+=1
    c = pd.DataFrame(data = a)
    c.columns = ['SId'] + keys
    c.set_index(['SId', 'Id'], inplace = True)
    return c, sentences
