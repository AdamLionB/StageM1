import pandas as pd
import numpy as np
import re
import os

keys = ['Id', 'Form', 'Lemma', 'UPosTag', 'XPosTagA', 'Feats', 'Head', 'DepRel', 'Deps', 'Misc']
prefix_len = len('# text = ')


#TODOC 
def file_batcher(file : 'Iterator' , batch_size : int):
    """
    FR : Format le contenue du fichier CoNLL en  Dataframes contenants chacune au maxium
    'batch_size' phrases, 
    EN : 
    Params
    ------
        file : file
            FR :
            EN :
        batch_size : int
            FR :
            EN : 
    Yields
    ------
    DatFrame :
        FR : 
        EN :
    list[str] :
        FR :
        EN :
    file : str

    """
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
                if i % batch_size == 0:
                    if keep: all_rows += s_rows
                    else : sentences = sentences[:-1]
                    frame = pd.DataFrame(data = all_rows)
                    if not frame.empty:
                        frame.columns = ['SId'] + keys + (['Mwe'] if len(frame.columns ) > len(keys)+1 else [])
                        frame.set_index(['SId', 'Id'], inplace = True)
                        yield frame, sentences, file
                        all_rows = []
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

def corpus_batcher(corpus_dir_path : str, batch_size : int = 10_000):
    """
    FR : Parcour le corpus par lots de batch_size phrases
    EN : Parkour the corpus per batch of batch_size sentences
    Params
    ------
        corpus_dir_path : str\n
            FR : Emplacement du corpus\n
            EN : The corpus path\n
        batch_size : int default 10_000\n
            FR : Nombre de phrase à lire par lot\n
            EN : Number of sentence to be read per batch\n
    Yields
    ------
        data : DataFrame[('Sid', 'Id'), ...]
            FR : Tableau de tout les token du lot
            EN : Table of all the token in the batch
        sentences : list[str]
            FR : Liste des phrase du lot
            EN : List of the sentences of the batch
    Examples
    --------
    >>> buffer = []
    >>> for data, sentences in corpus_batcher('path'):
    >>>     buffer += do_something(data)
    >>> aggregate(buffer)
    """
    for path in os.listdir(corpus_dir_path):
        file = open(corpus_dir_path+'\\'+path, encoding='utf-8')
        for data, sentences, file in file_batcher(file, batch_size):
            yield data, sentences


