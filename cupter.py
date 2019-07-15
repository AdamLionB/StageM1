from conllu_reader import corpus_batcher
from collections import defaultdict
import pandas as pd

dic1 = {}
dic2 = defaultdict(list)

X = None
for data, sentence in corpus_batcher('Test'):
  X = data
  for (SId, Id),row in data.iterrows():
    if row['Mwe'] != '*' and row['Mwe'] != '_':
      for mwe in row['Mwe'].split(';'):
        tmp = mwe.split(':')
        if len(tmp) != 1:
          dic1[(SId, tmp[0])] = tmp[1]
        dic2[(SId, tmp[0])].append(row[['Lemma', 'UPosTag']].tolist())

#TODO refactor
#TODOC
def something(l):
  verb = None
  noun = None
  for lemme, pos in l:
    if pos == 'VERB' : verb = lemme
    elif pos == 'NOUN':
      if noun is None : noun = lemme
      else : return None
  return noun, verb

dic = {k : something(v) for k,v in dic2.items() if 'LVC' in dic1[k] }
dic = {k : v for k, v in dic.items() if v is not None}
dic = {v for k, v in dic.items() }
dic = pd.DataFrame(dic, columns=['NOUN', 'VERB'])
dic.to_csv("test_truth.csv", index=False)




