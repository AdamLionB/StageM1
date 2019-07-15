from conllu_reader import corpus_batcher
from collections import defaultdict
import pandas as pd
import numpy as np


     

# TODOC
# TODO remove disk thing
# ORDER 1_1_1_1
# ORDER 1_2_1
def find_candidats(corpus_dir_path):  
    '''
    find all couples (verb, noun) in the corpus where the verb points toward the noun.
    Write the list of all those couples at the path given as output_path.
    this file will contain duplicates
    '''
    nb_insertion = 0 #TODO manage insertion
    
    verbs = {}
    nouns = {}
    couples = {}

    for data, sentences in corpus_batcher(corpus_dir_path, batch_size= 100_000):
        v = data.loc[data.UPosTag == 'VERB'] #select the line with verbs
        n = data.loc[data.UPosTag == 'NOUN'] #select the line with nouns
        #keeps the couples noun-verb where the noun point to the verb
        candidats = (pd.merge(n, v, left_on=['SId', 'Head']
                              , right_on=['SId', 'Id']
                              , suffixes=['_n', '_v']))[['Lemma_n', 'Lemma_v']]
        
        # add each candidat to a dict if they are not already in.
        # count the number of occurences
        for row in candidats.to_numpy():
            nouns.setdefault(row[0], [0, 0, 0, 0])[0]+=1
            verbs.setdefault(row[1], [0, 0, 0, 0])[0]+=1
            couples.setdefault((row[0], row[1]), [0, 0, 0, 0, 0, 0, 0, 0, 0])[0]+=1
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
    #print(couples['N'])
    return verbs, nouns, couples
    
  

# TODOC
#ORDER 1_1_1_2
def compute_features(corpus_dir_path, verbs, nouns, couples):
    '''
    Réalise plusieurs mesures sur les nom, verbe et couples (nom, verbe) du corpus
    '''
    print('computing measure')
    size, n_v, n_n, n_s = (0, 0, 0, 0)
    for data, sentences in corpus_batcher(corpus_dir_path, batch_size= 100_000):
        '''
        Les diverses comptes sont réalisés sur des batchs du corpus,
        la df 't' sert de tampon afin d'ajouter le resultat des comptes au df concerné
        '''
        #### Setup ####
        
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
        #print(t)
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
    #print(couples[['P_obj', 'N']].sort_values('N'))
    couples.update(pd.DataFrame({'P_obj' : couples['P_obj'] / couples['N']}))
    #Pour chaque couple nom, verbe ; calcule la fréquence à laquelle le nom est oblique du verbe
    couples.update(pd.DataFrame({'P_obl' : couples['P_obl'] / couples['N']}))
    #Pour chaque couple nom, verbe ; calcule la fréquence à laquelle le nom est sujet du verbe
    couples.update(pd.DataFrame({'P_subj' : couples['P_subj'] / couples['N']}))
    #Pour chaque couple nom, verbe ; calcule la fréquence à laquelle le nom n'est, ni objet, oblique ou sujet du verbe
    couples.update(pd.DataFrame({'P_other' : 1 - (couples['P_obj'] + couples['P_obl'] + couples['P_subj'])}))
    
    return couples.fillna(0)


##### PATRON #######
# TODOC
#ORDER 1_1_2_1
#ORDER 1_1_3_1
def compute_patrons(corpus_dir_path):
    """
    FR :

    EN :
    Returns
    -------
    ??
        FR :

        EN :
    """
    d = {}
    for data, sentences in corpus_batcher(corpus_dir_path):
        verbes = data.loc[data.UPosTag == 'VERB'] 
        nouns = data.loc[data.UPosTag == 'NOUN']
        couples = pd.merge(nouns.reset_index(), verbes.reset_index()
                     , left_on=['SId', 'Head']
                     , right_on=['SId', 'Id']
                     , suffixes = ['_n', '_v']).set_index('SId')
        all_patrons = []
        temp = []
        current_sid = -1
        gen = couples_per_sentences(couples)
        for (Sid, Id), pos in data['UPosTag'].iteritems():
            try :
                while(current_sid < Sid):
                    current_sid, j = next(gen)
                    all_patrons.append((current_sid,j))
                    #print(si, Sid)
                for jj in j:
                    if int(Id) > jj[0] and int(Id) < jj[1]:
                        jj[2].append(pos)
            except :
                pass
            
            
        return all_patrons



# TODOC     
# ORDER 1_1_2_1_1 
# ORDER 1_1_3_1_1         
def couples_per_sentences(couples):
    """
    FR : Génère pour chaque phrase une liste des couples (nom,verbe) trié par ordre d'apparition du premier
    element du couple

    EN :
    Parameters
    ----------
    couples : Dataframe
        FR : Ensemble des couples (nom,verbe) -unique- du corpus

        EN : 
    Yields
    ------
    int : 
        FR : identifiant de la phrase

        EN :
    list[(str,str,list)] :
        FR : 

        EN :
    """
    old = couples.iloc[0].name
    temp = []
    for sid, idn, idv in couples[['Id_n', 'Id_v']].itertuples():
        idn = int(idn)
        idv = int(idv)
        if sid != old :
            yield old, sorted(temp, key= lambda x : int(x[0]))
            old = sid
            temp = []
        else:
            temp.append((min(idn,idv), max(idn,idv), [],))
    yield old, sorted(temp, key= lambda x : int(x[0]))

# TODOC
# ORDER 1_1_2_2
def get_most_frequent_patrons(all_patrons):
    d = defaultdict(int)
    for i in all_patrons:
        for j in i[1]:
            d[tuple(j[2])]+=1
    return [a[0] for a in sorted([(k,v) for k,v in d.items()], key = lambda x: x[1], reverse=True)[:20]]

# TODOC
# TODO rename
#ORDER 1_1_3_1
def p(corpus_dir_path, a, b):
    e = defaultdict(lambda : defaultdict(int))
    n = 0
    for data, sentences in corpus_batcher(corpus_dir_path):
        last = data.iloc[-1].name[0]
        for i, j in a[n:]:
            if i > last:
                break
            for k, l, m in j:
                if tuple(m) in b:
                    # TODO why try ?
                    try:
                        a1 = data.loc[(i,str(k))]['Lemma']
                        a2 = data.loc[(i,str(l))]['Lemma']
                        if data.loc[(i,str(k))]['UPosTag'] == 'NOUN':
                            a1, a2 = a2, a1
                        e[(a1, a2)][tuple(m)] +=1
                    except:
                        print('-------',i,k,l,m)
            n+=1
    res = pd.DataFrame.from_dict(e, orient='index').fillna(0)
    res.columns = b
    res.index.names = ('VERB', 'NOUN')
    return res






# TODO RENAME
# TODOC
class patronateur():
    #ORDER 1_1_2
    def fit(self, corpus_dir_path):
        all_patrons = compute_patrons(corpus_dir_path)
        self.patrons = get_most_frequent_patrons(all_patrons)
    #ORDER 1_1_3
    def to_rename(self, corpus_dir_path):
        all_patrons = compute_patrons(corpus_dir_path)
        return p(corpus_dir_path, all_patrons, self.patrons)


# TODOC
#ORDER 1_1_1_2_1
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


# TODOC
# ORDER 1_1_1
def get_features(corpus_dir_path : str) -> pd.DataFrame:
    """
        FR : \n
        EN :\n
        Params
        ------
            corpus : DataFrame\n
                FR :\n
                EN :\n
        Returns
        -------
            features : DataFrame\n
                FR :\n
                EN :\n
    """
    v, n, c = find_candidats(corpus_dir_path)
    features = compute_features(corpus_dir_path, v,n,c)
    return features
























