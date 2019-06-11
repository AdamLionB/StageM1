from conllu_reader import read_conllu_from, corpus_batcher
import pandas as pd
import numpy as np
import csv
import time
import os


def find_candidats(corpus_dir_path, output_path):
    '''
    find all couples (verb, noun) in the corpus where the verb points toward the noun.
    Write the list of all those couples at the path given as output_path.
    this file will contain duplicates
    '''
    nb_insertion = 0 #TODO
    
    out = open(output_path, 'w', encoding='utf-8')
    writer = csv.writer(out, delimiter=' ', lineterminator="\n")

    for data, sentences in corpus_batcher(corpus_dir_path, batch_size= 100_000):
        verbs = data.loc[data.UPosTag == 'VERB'] #select the line with verbs
        nouns = data.loc[data.UPosTag == 'NOUN'] #select the line with nouns
        #keeps the couples noun-verb where the noun point to the verb
        candidats = (pd.merge(nouns, verbs, left_on=['SId', 'Head']
                              , right_on=['SId', 'Id']
                              , suffixes=['_n', '_v']))[['Lemma_n', 'Lemma_v']].values
        for c in candidats:
            writer.writerow(c)
    print('done')
  


def load_candidats(candidats_path):
    '''
    generate a dataframe for the nouns, verbs and couples (noun, verb)
    with a row per unique noun, verb, (noun,verb).
    Count the number of occurence of each noun, verb, (noun,verb) and
    set others columns to 0
    '''
    print('loading')
    start = time.time()
    with open(candidats_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter = ' ')
        verbs = {}
        nouns = {}
        couples = {}
        # add each candidat to a dict if they are not already in.
        # count the number of occurences
        for row in reader:
            nouns.setdefault(row[0], [0, 0, 0, 0])[0]+=1
            verbs.setdefault(row[1], [0, 0, 0, 0])[0]+=1
            couples.setdefault((row[0], row[1]), [0, 0, 0, 0, 0, 0, 0, 0, 0])[0]+=1

        #generate the verbs dataframe
        verbs = pd.DataFrame(data = verbs).transpose()
        verbs.index.name = 'VERB'
        verbs.set_axis(['N', 'P', 'T', 'supp'], axis = 1, inplace= True)

        #generate the nouns dataframe        
        nouns = pd.DataFrame(data = nouns).transpose()
        nouns.index.name = 'NOUN'
        nouns.set_axis(['N', 'P', 'T', 'supp'], axis = 1, inplace= True)

        #generate the candidates dataframe
        couples = pd.DataFrame(data = couples).transpose()
        couples.index.names = ['NOUN', 'VERB']
        couples.set_axis(['N', 'P', 'T', 'supp', 'dist', 'P_obl', 'P_subj', 'P_obj', 'P_other'], axis = 1, inplace= True)
        print('loading done in :', (time.time() - start))
        return verbs, nouns, couples



def add_to_col(df_to_update, to_add, col_to_update_name):
    df_to_update.update(
        pd.DataFrame({col_to_update_name :
             pd.concat(
                 [df_to_update[[col_to_update_name]],to_add]
                 , axis=1
                 , sort=True)
             .fillna(0)
             .sum(axis=1)
            })
        )


def compute_measures(verbs, nouns, couples):
    '''
    Réalise plusieurs mesures sur les nom, verbe et couples (nom, verbe) du corpus
    '''
    print('computing measure')
    size, n_v, n_n, n_s = (0, 0, 0, 0)
    for data, sentences in corpus_batcher('Train', batch_size= 100_000):
        '''
        Les diverses comptes sont réalisés sur des batchs du corpus,
        la df 't' sert de tampon afin d'ajouter le resultat des comptes au df concerné
        '''
        # isole les verbes, nom, (nom, verbe) du batch dans des df
        v = data.loc[data.UPosTag == 'VERB'] 
        n = data.loc[data.UPosTag == 'NOUN']
        s = pd.merge(n.reset_index(), v.reset_index()
                     , left_on=['SId', 'Head']
                     , right_on=['SId', 'Id']
                     , suffixes = ['_n', '_v']).set_index('SId')

        # Compte le nombre de phrases, de verbe, de nom, et de couple (non unique)
        size += len(sentences)
        n_v += len(v)
        n_n += len(n)
        n_s += len(s)


        #### Patterne mining ####
        # Premier exemple détaillé pour mieux comprendre la syntaxe pandas

        # Pour chaque couple nom, verbe ; compte le nombre de phrase dans lesquelles on retouve au moins une fois le nom ET le verbe (lié ou non)
        t = (s[['Lemma_n', 'Lemma_v']]                      # isole les colonnes 'Lemma_n' et 'Lemma_v' dans une df (avec 'Sid' et 'Id' car ce sont des indexes)
             .assign(T=1)                                   # ajoute une colonne T remplie 1 à la df
             .groupby(['SId', 'Lemma_n', 'Lemma_v'])        # groupe par phrase, nom, verbe 
             .min()                                         # met chaque groupe T à 1
             .groupby(['Lemma_n', 'Lemma_v'])               # groupe par nom, verbe
             .count())                                      # compte le nombre de phrase aggregé à chaque groupe.
        add_to_col(couples, t, 'T')                         # ajoute le résultat précédent à la df couples
        # Pour chaque verbe ; compte le nombre de phrases dans lesquelles on retrouve au moins une fois le verbe
        t = v[['Lemma']].assign(T=1).groupby(['SId', 'Lemma']).min().groupby('Lemma').count()
        add_to_col(verbs, t, 'T')
        # Pour chaque nom ; compte le nombre de phrases dans lesquelles on retrouve au moins une fois le nom
        t = n[['Lemma']].assign(T=1).groupby(['SId', 'Lemma']).min().groupby('Lemma').count()
        add_to_col(nouns, t, 'T')


        #### Distance nom, verbe ####
        
        # Pour chaque couple (nom, verbe) unique; fait la somme des distance (nombre de mots) entre le nom est le verbe des instances de (nom, verbe)
        t = s.assign(dist=np.abs(pd.to_numeric(s.Id_n) - pd.to_numeric(s.Id_v))).groupby(['Lemma_n', 'Lemma_v']).sum()
        add_to_col(couples, t, 'dist')


        #### Fréquence relation ####

        # Pour chaque couple (nom, verbe) ; compte le nombre de fois où le couple est lié par une relation de type
        # obj, obl ou subj
        t = s[['DepRel_n', 'Lemma_n', 'Lemma_v']].assign(N_rel=0).groupby(['DepRel_n', 'Lemma_n', 'Lemma_v']).count()
        add_to_col(couples, t.loc[('obj')], 'P_obj')
        add_to_col(couples, t.loc[('obl')], 'P_obl')
        add_to_col(couples, t.loc[('nsubj')], 'P_subj')


    #### Probabilitées ####

    # Pour chaque couple nom, verbe ; calcule la probabilité qu'un couple nom, verbe quelconque soit le couple nom, verbe en question lié [p(n et v)]
    couples.update(pd.DataFrame({'P' : couples['N'] / n_s}))   
    # Pour chaque verbe ; calcule la probabilité qu'un verbe quelconque soit le verbe en question lié à un nom quelconque [p(v)]
    verbs.update(pd.DataFrame({'P' : verbs['N'] / n_s}))
    # Pour chaque nom ; calcule la probabilité qu'un nom quelconque soit le nom en question lié à un verbe quelconque [p(n)]
    nouns.update(pd.DataFrame({'P' : nouns['N'] / n_s}))


    #### Patterne mining ####
    
    # Pour chaque verbe ; calcule la probabilité qu'une phrase quelconque contienne le verbe [supp(v)]
    verbs.update(pd.DataFrame({'supp' : verbs['T'] / size}))
    # Pour chaque nom ; calcule la probabilité qu'une phrase quelconque contienne le nom [supp(n)]
    nouns.update(pd.DataFrame({'supp' : nouns['T'] / size}))


    #### Distance nom, verbe ####
    
    # Pour chaque couple (nom, verbe) ; calcule la distanc moyenne entre le nom et le verbe
    couples.update(pd.DataFrame({'dist' : couples['dist'] / couples['N']}))


    #### Mise en commun ####

    #fusionne les mesures relatives aux noms et au verbes aux mesure de couples (star schéma to 
    nouns.rename(columns=lambda x: str(x) + '_n', inplace = True)
    couples = pd.merge(couples.reset_index(), nouns, on='NOUN')
    nouns = None
    verbs.rename(columns=lambda x: str(x) + '_v', inplace = True)
    couples = pd.merge(couples.reset_index(), verbs, on='VERB')
    verbs = None
    couples.set_index(['NOUN', 'VERB'], inplace = True)


    #### Probablitées ####
    
    # Pour chaque couple nom, verbe ; calcule la probabilité qu'un nom quelconque soit le nom en question sachant le verbe
    couples = couples.assign(P_n_given_v = couples['P'] / couples['P_v'])
    # Pour chaque couple nom, verbe ; calcule la probabilité qu'un verbe quelconque soit le verbe en question sachant le nom
    couples = couples.assign(P_v_given_n = couples['P'] / couples['P_n'])
    # Pour chaque nom ; calcule la variance de la probabilité du nom
    couples = couples.assign(V_n = couples['P_n'] * (1 - couples['P_n']))
    # Pour chaque verbe ; calcule la variance de la probabilité du verbe
    couples = couples.assign(V_v = couples['P_v'] * (1 - couples['P_v']))
    # Pour chaque couple (nom, verbe) ; calcule la covariance de la probabilité du nom et la probabilité du verbe
    couples = couples.assign(V = couples['P'] - (couples['P_n'] * couples['P_v']))
    # Pour chaque cuple (nom, verbe) ; calcule la corrélation entre la probabilité du nom et la probabilité du verbe
    couples = couples.assign(corr = couples['V'] / (np.sqrt(couples['V_n']) * np.sqrt(couples['V_v'])))
    # Pour chaque couple nom, verbe ; calcule le pmi
    couples = couples.assign(pmi = np.log(couples['P_n_given_v'] / couples['P_n']))

    #### Patterne mining ####
    
    # Pour chaque couple nom, verbe ; calcule la probabilité qu'une phrase quelconque contienne le nom ET le verbe (lié ou non) [supp(x -> y) || supp(y -> x)]
    couples.update(pd.DataFrame({'supp' : couples['T'] / size}))
    # Pour chaque couple nom, verbe ; calcule la probabilité qu'une phrase contenant le nom contienne aussi le verbe (lié ou non) [conf(x -> y)]
    couples = couples.assign(conf_n_v = couples['supp'] / couples['supp_n'])
    # Pour chaque couple nom, verbe ; calcule la probabilité qu'une phrase contenant le verbe contienne aussi le nom (lié ou non) [conf(y -> x)]
    couples = couples.assign(conf_v_n = couples['supp'] / couples['supp_v'])


    #### Fréquence relation ####
    #Pour chaque couple nom, verbe ; calcule la fréquence à laquelle le nom est objet du verbe
    couples.update(pd.DataFrame({'P_obj' : couples['P_obj'] / couples['N']}))
    #Pour chaque couple nom, verbe ; calcule la fréquence à laquelle le nom est oblique du verbe
    couples.update(pd.DataFrame({'P_obl' : couples['P_obl'] / couples['N']}))
    #Pour chaque couple nom, verbe ; calcule la fréquence à laquelle le nom est sujet du verbe
    couples.update(pd.DataFrame({'P_subj' : couples['P_subj'] / couples['N']}))
    #Pour chaque couple nom, verbe ; calcule la fréquence à laquelle le nom n'est, ni objet, oblique ou sujet du verbe
    couples.update(pd.DataFrame({'P_other' : 1 - (couples['P_obj'] + couples['P_obl'] + couples['P_subj'])}))
    
    return couples.fillna(0)


def f():
    d = {}
    for data, sentences in corpus_batcher('Train'):
        v = data.loc[data.UPosTag == 'VERB'] 
        n = data.loc[data.UPosTag == 'NOUN']
        s = pd.merge(n.reset_index(), v.reset_index()
                     , left_on=['SId', 'Head']
                     , right_on=['SId', 'Id']
                     , suffixes = ['_n', '_v']).set_index('SId')
        s[['Id_n', 'Id_v']]
        res = []
        temp = []
        si = -1
        gen = g(s)
        for (Sid, Id), pos in data['UPosTag'].iteritems():
            while(si < Sid):
                si, j = next(gen)
            for jj in j:
                if int(Id) > jj[0] and int(Id) < jj[1]:
                    jj[2].append(pos)
            res.append(j)
        return res
                
        


def g(s):
    old = s.iloc[0].name
    temp = []
    for sid, idn, idv in s[['Id_n', 'Id_v']].itertuples():
        if sid != old :
            old = sid
            yield sid, sorted(temp, key= lambda x : int(x[0]))
            temp = []
        else:
            temp.append((min(int(idn),int(idv)), max(int(idn),int(idv)), [],))
    yield temp
            
x = f()
#find_candidats('Train', 'test.txt')
#v, n, c = load_candidats('test.txt')
#x = compute_measures(v,n,c)
'''vv, nn, cc'''
#a = f(v, n, c)

#truth = pd.DataFrame.from_csv('truth.csv', encoding='utf-8')
#truth = truth.reset_index().set_index(['NOUN', 'VERB']).assign(isLVC='YES')
#a = pd.merge(a, truth, how='left', left_index= True, right_index= True).fillna('NO')






















