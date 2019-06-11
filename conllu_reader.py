import pandas as pd
import numpy as np
import re
import os

keys = ['Id', 'Form', 'Lemma', 'UPosTag', 'XPosTagA', 'Feats', 'Head', 'DepRel', 'Deps', 'Misc']
keys2 = ['Id', 'Form', 'Lemma', 'UPosTag', 'XPosTagA', 'Feats', 'Head', 'DepRel', 'Deps', 'Misc', 'Mwe']
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

def read_conllu_from2(file, batch_size):
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
            if l == '\n' or l == '\n\r' or '#' in l :break
            cells = l[:-1].split('\t')
            row = [i] + [v for v in cells]
            if(len(row) < 4) : print(row)
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
    frame.columns = ['SId'] + keys2
    frame.set_index(['SId', 'Id'], inplace = True)
    return frame, sentences, file

def read_conllu_from(file, batch_size):
    '''
    read the next batch_size sentences of the file
    '''
    sentences = []
    all_rows = []
    s_rows = []
    i = 0
    keep = True
    new = True
    for line in file:
        # check the line is empty or is a 'comment' (begin with #)
        if line == '\n' or line == '\n\r' or '#' in line :
            # if the line contains 'text' then it is the begining of a new sentence
            if 'text' in line :
                # register the sentence
                sentences.append(line[prefix_len:])
                new = True
        elif new :
            new = False
            if keep: all_rows += s_rows
            else :
                sentences = sentences[:-1]
                keep = True
            
            s_rows = []
            row = []
            if i >= batch_size : break
            i+=1
        else:
            #put the content of the line in an array
            cells = line[:-1].split('\t')
            row = [i] + [v for v in cells]
            #throws away sentence if the word of this line is a noun or a verb that contains unhautorized char
            if row[4] == 'NOUN' or row[4] == 'VERB':
                if re.match(r'\W', row[3], flags = re.UNICODE) :
                    keep = False
            s_rows.append(row)
    if keep: all_rows += s_rows
    else : sentences = sentences[:-1]
    frame = pd.DataFrame(data = all_rows)
    if not frame.empty:
        frame.columns = ['SId'] + keys + (['Mwe'] if len(frame.columns ) > len(keys)+1 else [])
        frame.set_index(['SId', 'Id'], inplace = True)
        return frame, sentences, file
    else:
        return None, None, None
    
def file_batcher(file, batch_size):
    '''
    read the next batch_size sentences of the file
    '''
    sentences = []
    all_rows = []
    s_rows = []
    i = 0
    keep = True
    new = False
    for line in file:
        # check the line is empty or is a 'comment' (begin with #)
        if line == '\n' or line == '\n\r' or '#' in line :
            # if the line contains 'text' then it is the begining of a new sentence
            if 'text' in line :
                # register the sentence
                sentences.append(line[prefix_len:])
                new = True
        else:
            if new :
                new = False
                if keep: all_rows += s_rows
                else :
                    sentences = sentences[:-1]
                    keep = True
                
                s_rows = []
                row = []
                i+=1
                if i >= batch_size :
                    i = 0
                    if keep: all_rows += s_rows
                    else : sentences = sentences[:-1]
                    frame = pd.DataFrame(data = all_rows)
                    if not frame.empty:
                        frame.columns = ['SId'] + keys + (['Mwe'] if len(frame.columns ) > len(keys)+1 else [])
                        frame.set_index(['SId', 'Id'], inplace = True)
                        yield frame, sentences, file
                    else:
                        break  
            #put the content of the line in an array
            cells = line[:-1].split('\t')
            row = [i] + [v for v in cells]
            #throws away sentence if the word of this line is a noun or a verb that contains unhautorized char
            if row[4] == 'NOUN' or row[4] == 'VERB':
                if re.match(r'\W', row[3], flags = re.UNICODE) :
                    keep = False
            if '-' not in row[1]:
                s_rows.append(row)
    if keep: all_rows += s_rows
    else : sentences = sentences[:-1]
    frame = pd.DataFrame(data = all_rows)
    if not frame.empty:
        frame.columns = ['SId'] + keys + (['Mwe'] if len(frame.columns ) > len(keys)+1 else [])
        frame.set_index(['SId', 'Id'], inplace = True)
        yield frame, sentences, file


def corpus_batcher(corpus_dir_path, batch_size= 10_000):
    for path in os.listdir(corpus_dir_path):
        file = open(corpus_dir_path+'\\'+path, encoding='utf-8')
        for data, sentences, file in file_batcher(file, batch_size):
            yield data, sentences

