from conllu_reader import read_conllu
from nltk.stem import WordNetLemmatizer, SnowballStemmer, lancaster
import stanfordnlp
import pandas as pd
import numpy as np
import csv
import time
import os
#import nltk
#nltk.download('wordnet')
#stanfordnlp.download('fr')
#stanfordnlp.download('en')


sno = SnowballStemmer('english')
lan = lancaster.LancasterStemmer()
lan2 = lancaster.LancasterStemmer(strip_prefix_flag=True)
wnl = WordNetLemmatizer()
#stan = stanfordnlp.Pipeline(processors='tokenize,pos,lemma',lang="en")



functions = []
functions.append(lambda x : x)
functions.append(sno.stem)
functions.append(lan.stem)
#renaming a function
def lan2stem(args):
    return lan2.stem(args)
functions.append(lan2stem)
functions.append(wnl.lemmatize)
#functions.append(lambda x : stan(x).setences[0].words[0].lemma)

def test(word):
    return [f(word) for f in functions]

class Trie:
    def __init__(self, letter = '', deepness = 0):
        self.letter = letter
        self.children = {}
        self.values = {}
        self.deepness = deepness
    def add(self, transformation, transformed_word, word):
        if self.deepness == len(transformed_word):
            values = self.values.setdefault(transformation, []).append(word)
        else:
            l = word[self.deepness]
            child = self.children.setdefault(l, Trie(l, self.deepness+1)).add(transformation, transformed_word, word)
    def get(self, transformation, word):
        if self.deepness == len(transformed_word):
            return self.values.get(transformation)
        child = self.children.get(transformed_word[self.deepness+1])
        if child is not None:
            child.get(transformation, word)
    def __repr__(self):
        st = ''
        st += '-'*self.deepness +  self.letter + str(self.values) +'\n'
        for child in self.children.values():
            st += str(child)
        return st
'''
conllu = Conllu_file('fr-common_crawl-164.conllu', 10)


verbs = []
for n, sentence in enumerate(conllu):
    for word, cells in sentence.rows.items():
        if cells['UPosTag'] == 'VERB':
          verbs.append((n,word))
verbs = {conllu[n][word]['Form'] for n, word in verbs}
transformed_verbs = [test(verb) for verb in verbs]
trie = Trie()
for transformed_verb in transformed_verbs:
    for n,transfo in enumerate(transformed_verb[1:]):
        trie.add(functions[n+1].__qualname__, transfo, transformed_verb[0])

print(trie)


verbs = f.index.array
transformed_verbs = [test(verb) for verb in verbs]
trie = Trie()
for transformed_verb in transformed_verbs:
    for n,transfo in enumerate(transformed_verb[1:]):
        trie.add(functions[n+1].__qualname__, transfo, transformed_verb[0])

print(trie)

'''

def find_candidats():
    nb_insertion = 0
    with open('candidats.csv', 'w', encoding='utf-8') as f:
        pass #reset output file
    for path in os.listdir('Corpus'):
        batch_size = 100000
        batchs = 0
        sentences = []
        while len(sentences) == batch_size or batchs == 0:
            print(batchs)
            data, sentences = read_conllu('Corpus/fr-common_crawl-164.conllu',  batchs * batch_size, (batchs+1) * batch_size) #get a Dataframe representing the corpus
            batchs +=1
            verbs = data.loc[data.UPosTag == 'VERB'] #select the line with verbs
            nouns = data.loc[data.UPosTag == 'NOUN'] #select the line with nouns
            #keeps the couples noun-verb where the noun point to the verb
            candidats = (pd.merge(nouns, verbs, left_on=['SId', 'Head'], right_on=['SId', 'Id'], suffixes=['_n', '_v']))[['Lemma_n', 'Lemma_v']].values
            with open('candidats.csv', 'a', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=' ', lineterminator="\n")
                for c in candidats:
                    writer.writerow(c)
    print('done')        


def load_candidats():
    print('loading')
    start = time.time()
    with open('candidats.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter = ' ')
        verbs = {}
        nouns = {}
        couples = {}
        for row in reader:
            nouns.setdefault(row[0], [0, 0, 0, 0])[0]+=1
            verbs.setdefault(row[1], [0, 0, 0, 0])[0]+=1
            couples.setdefault((row[0], row[1]), [0, 0, 0, 0, 0, 0, 0, 0, 0])[0]+=1
        verbs = pd.DataFrame(data = verbs).transpose()
        verbs.index.name = 'VERB'
        verbs.set_axis(['N', 'P', 'T', 'supp'], axis = 1, inplace= True)
        
        nouns = pd.DataFrame(data = nouns).transpose()
        nouns.index.name = 'NOUN'
        nouns.set_axis(['N', 'P', 'T', 'supp'], axis = 1, inplace= True)
        
        couples = pd.DataFrame(data = couples).transpose()
        couples.index.names = ['NOUN', 'VERB']
        couples.set_axis(['N', 'P', 'T', 'supp', 'dist', 'P_obl', 'P_subj', 'P_obj', 'P_other'], axis = 1, inplace= True)
        print('loading done in :', (time.time() - start))
        return verbs, nouns, couples

def f(verbs, nouns, couples):
    print('f')
    start = time.time()
    for path in os.listdir('Corpus'):
        batch_size = 100000
        batchs = 0
        sentences = []
        size = 0
        n_v = 0
        n_n = 0
        n_s =0
        while len(sentences) == batch_size or batchs == 0:
            
            print(batchs, end = ' ')
            start2 = time.time()
            
            #get a Dataframe representing the corpus
            data, sentences =read_conllu('Corpus/fr-common_crawl-164.conllu',
                                         batchs * batch_size, (batchs+1) * batch_size)
            size += len(sentences)
            mid2= time.time()
            print(mid2 - start2, end =' ')

            batchs +=1

            v = data.loc[data.UPosTag == 'VERB'] 
            n = data.loc[data.UPosTag == 'NOUN']
            #c = pd.merge(n, v, left_on=['SId'], right_on=['SId'], suffixes = ['_n', '_v'])[['Lemma_n', 'Lemma_v']]
            s = pd.merge(n.reset_index(), v.reset_index(), left_on=['SId', 'Head'], right_on=['SId', 'Id'], suffixes = ['_n', '_v']).set_index('SId')
            

            n_v += len(v)
            n_n += len(n)
            n_s += len(s)


            # Pour chaque couple nom, verbe ; compte le nombre de phrase dans lesquelles on retouve au moins une fois le nom ET le verbe (lié ou non)
            t = (s[['Lemma_n', 'Lemma_v']]
                 .assign(T=1)                   # ajoute une colonne de 1
                 .groupby(['SId', 'Lemma_n', 'Lemma_v'])        # groupe par phrase, nom, verbe
                 .min()                                         # met chaque groupe à 1
                 .groupby(['Lemma_n', 'Lemma_v'])               # groupe par nom, verbe
                 .count())                                      # compte le nombre de phrase aggregé a chaque groupe.

            couples.update(                                     # met à jour les résultats déjà existant
                pd.DataFrame(                                   # met en forme
                    {'T' :
                     pd.concat(                                 # fusionne les resultats déjà existant aux nouveaux
                         [couples[['T']], t]     
                         , axis=1
                         , sort=True)
                     .fillna(0)                                 
                     .sum(axis=1)}))                            # somme les resultats
            

            # Pour chaque verbe ; compte le nombre de phrases dans lesquelles on retrouve au moins une fois le verbe
            t = v[['Lemma']].assign(T=1).groupby(['SId', 'Lemma']).min().groupby('Lemma').count()
            verbs.update(pd.DataFrame({'T' : pd.concat([verbs[['T']],t], axis=1, sort=True).fillna(0).sum(axis=1)}))

            # Pour chaque nom ; compte le nombre de phrases dans lesquelles on retrouve au moins une fois le nom
            t = n[['Lemma']].assign(T=1).groupby(['SId', 'Lemma']).min().groupby('Lemma').count()
            nouns.update(pd.DataFrame({'T' : pd.concat([nouns[['T']],t], axis=1, sort=True).fillna(0).sum(axis=1)}))


            t = s.assign(dist=np.abs(pd.to_numeric(s.Id_n) - pd.to_numeric(s.Id_v))).groupby(['Lemma_n', 'Lemma_v']).sum()
            couples.update(pd.DataFrame({'dist' : pd.concat([couples[['dist']], t], axis=1, sort=True).fillna(0).sum(axis=1)}))

            t = s[['DepRel_n', 'Lemma_n', 'Lemma_v']].assign(N_rel=0).groupby(['DepRel_n', 'Lemma_n', 'Lemma_v']).count()
            couples.update(pd.DataFrame({'P_obj' : pd.concat([couples['P_obj'], t.loc[('obj')]], axis=1, sort=True).fillna(0).sum(axis=1)}))
            couples.update(pd.DataFrame({'P_obl' : pd.concat([couples['P_obl'], t.loc[('obl')]], axis=1, sort=True).fillna(0).sum(axis=1)}))
            couples.update(pd.DataFrame({'P_subj' : pd.concat([couples['P_subj'], t.loc[('nsubj')]], axis=1, sort=True).fillna(0).sum(axis=1)}))
            print(time.time() - mid2)
        start2 = time.time()

        # Pour chaque couple nom, verbe ; calcule la probabilité qu'un couple nom, verbe quelconque soit le couple nom, verbe en question lié [p(n et v)]
        couples.update(pd.DataFrame({'P' : couples['N'] / n_s}))
            
        # Pour chaque verbe ; calcule la probabilité qu'un verbe quelconque soit le verbe en question lié à un nom quelconque [p(v)]
        verbs.update(pd.DataFrame({'P' : verbs['N'] / n_s}))

        # Pour chaque nom ; calcule la probabilité qu'un nom quelconque soit le nom en question lié à un verbe quelconque [p(n)]
        nouns.update(pd.DataFrame({'P' : nouns['N'] / n_s}))

        # Pour chaque verbe ; calcule la probabilité qu'une phrase quelconque contienne le verbe [supp(v)]
        verbs.update(pd.DataFrame({'supp' : verbs['T'] / size}))

        # Pour chaque nom ; calcule la probabilité qu'une phrase quelconque contienne le nom [supp(n)]
        nouns.update(pd.DataFrame({'supp' : nouns['T'] / size}))

        couples.update(pd.DataFrame({'dist' : couples['dist'] / couples['N']}))

        nouns.rename(columns=lambda x: str(x) + '_n', inplace = True)
        couples = pd.merge(couples.reset_index(), nouns, on='NOUN')
        nouns = None
        verbs.rename(columns=lambda x: str(x) + '_v', inplace = True)
        couples = pd.merge(couples.reset_index(), verbs, on='VERB')
        verbs = None
        couples.set_index(['NOUN', 'VERB'], inplace = True)


        # Pour chaque couple nom, verbe ; compte le nombre de phrase dans lesquelles on retrouve au moins une fois le nom OU le verbe (lié ou non)
        #couples.update(pd.DataFrame({'n_transaction_or' : couples['n_transaction_n'] + couples['n_transaction_v'] - couples['n_transaction_and']}))

        

        # Pour chaque couple nom, verbe ; calcule la probabilité qu'un nom quelconque soit le nom en question sachant le verbe
        couples = couples.assign(P_n_given_v = couples['P'] / couples['P_v'])

        # Pour chaque couple nom, verbe ; calcule la probabilité qu'un verbe quelconque soit le verbe en question sachant le nom
        couples = couples.assign(P_v_given_n = couples['P'] / couples['P_n'])

        
        couples = couples.assign(V_n = couples['P_n'] * (1 - couples['P_n']))

        couples = couples.assign(V_v = couples['P_v'] * (1 - couples['P_v']))

        couples = couples.assign(V = couples['P'] - (couples['P_n'] * couples['P_v']))

        couples = couples.assign(corr = couples['V'] / (np.sqrt(couples['V_n']) * np.sqrt(couples['V_v'])))

        

        # Pour chaque couple nom, verbe ; calcule le pmi
        couples = couples.assign(pmi = np.log(couples['P_n_given_v'] / couples['P_n']))
        
        # Pour chaque couple nom, verbe ; calcule la probabilité qu'une phrase quelconque contienne le nom ET le verbe (lié ou non) [supp(x -> y) || supp(y -> x)]
        couples.update(pd.DataFrame({'supp' : couples['T'] / size}))

        # Pour chaque couple nom, verbe ; calcule la probabilité qu'une phrase contenant le nom contienne aussi le verbe (lié ou non) [conf(x -> y)]
        couples = couples.assign(conf_n_v = couples['supp'] / couples['supp_n'])

        # Pour chaque couple nom, verbe ; calcule la probabilité qu'une phrase contenant le verbe contienne aussi le nom (lié ou non) [conf(y -> x)]
        couples = couples.assign(conf_v_n = couples['supp'] / couples['supp_v'])


        couples.update(pd.DataFrame({'P_obj' : couples['P_obj'] / couples['N']}))

        couples.update(pd.DataFrame({'P_obl' : couples['P_obl'] / couples['N']}))

        couples.update(pd.DataFrame({'P_subj' : couples['P_subj'] / couples['N']}))
        
        couples.update(pd.DataFrame({'P_other' : 1 - (couples['P_obj'] + couples['P_obl'] + couples['P_subj'])}))
        
        print(time.time() - start2)
        print('f done in :', (time.time() - start))
        print(size)
        return couples

find_candidats()
v, n, c = load_candidats()

'''vv, nn, cc'''
a = f(v, n, c)

truth = pd.DataFrame.from_csv('truth.csv', encoding='utf-8')
truth = truth.reset_index().set_index(['NOUN', 'VERB']).assign(isLVC='YES')
a = pd.merge(a, truth, how='left', left_index= True, right_index= True).fillna('NO')
'''
newcols = ['index', 'n_selected', 'p_selected', 'p_selected_v', 'p_selected_n', 'p_n_given_v',
           'p_v_given_n', 'pmi', 'supp', 'supp_v', 'supp_n', 'conf_n_v', 'conf_v_n',
           'n_transaction_and',  'n_transaction_or', 'n_selected_n', 'n_transaction_n',
           'n_selected_v',  'n_transaction_v', 'isLVC']
f = f[newcols]
'''
a.to_csv(path_or_buf='res2.csv')

















