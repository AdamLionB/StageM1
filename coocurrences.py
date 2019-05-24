from sklearn.decomposition import TruncatedSVD
import os
from collections import defaultdict
import time
from conllu_reader import read_conllu_from
from itertools import product
from functools import reduce
import pickle
from collections import Counter
import traceback

directory = 'Corpus'

def f():
    return defaultdict(int)

def expression_s_coocurrences(expressions):
  x = defaultdict(f)
  for path in os.listdir(directory):
    batch_size = 10000
    batchs = 0
    first = time.time()
    file = open(directory+'/'+path, encoding='utf-8')
    k = 0
    m = 0
    try :
      while True:
        print(batchs, time.time()-first)
        data, sentence, file = read_conllu_from(file, batch_size)
        sentence= None
        batchs +=1
        data = data[['Lemma']].reset_index().to_numpy()
        temp = defaultdict(int)
        nb_words = defaultdict(f)
        for n,end_row in enumerate(data):
          if k != end_row[0] :
            #for word, occurences in temp.values():
            if len(nb_words) != 0 :
              C = defaultdict(lambda : reduce(lambda x,y : x*y, nb_words[exp].values()))
              for exp in expressions:
                if len(nb_words[exp]) != 0:
                  #print(temp)
                  for word, occurences in temp.items():
                    #print(len(x[exp]), x[exp][word], word)
                    x[exp][word] += occurences*C[exp]
            temp = defaultdict(int)
            nb_words = defaultdict(f)
            k = end_row[0]
            #m = n
          temp[end_row[2]]+=1
          #print(temp)
          for exp in expressions:
            for word in exp:
              if end_row[2] == word:
                nb_words[exp][word]+=1
          '''
          for start_row in data[m:n]:
            temp[s
            
            #k = end_row[0]
            
            nb_words = defaultdict(f)
            for start_row in data[m:n]:
              for exp in expressions:
                if all(word in data[m:n] for word in exp):
                  for word in exp:
                    if start_row[2] == word:
                      nb_words[exp][word]+=1
            if len(nb_words) != 0 :
              for start_row in data[m:n]:
                C = defaultdict(lambda : reduce(lambda x,y : x*y, nb_words[exp].values()))
                for exp in expressions:
                  x[exp][start_row[2]]+= C[exp]
          m = n
          '''
    except Exception as e:
      pass#traceback.print_exc()
  return x


def corpus_coocurrences():
    for path in os.listdir(directory):
        batch_size = 10000
        batchs = 0
        x = defaultdict(f)
        first = time.time()
        file = open(directory+'/'+path, encoding='utf-8')
        k = 0
        m = 0
        try :
            while True:
                print(batchs, time.time()-first)
                batchs+=1
                data, sentence, file = read_conllu_from(file, batch_size)
                sentence= None
                data = data[['Lemma']].reset_index().to_numpy()
                for n,row in enumerate(data):
                    if k != row[0] :
                        k = row[0]
                        m = n
                    for a in data[m:n]:
                        x[row[2]][a[2]]+=1
                        x[a[2]][row[2]]+=1
                        
        except Exception as e:
            pass
    return x

#data = expression_s_coocurrences([('prendre', 'décision'), ('faire','cuisine')])
#x= corpus_coocurrences()
#output = open('data.pkl', 'wb')
#pickle.dump(x, output)
#output.close()
