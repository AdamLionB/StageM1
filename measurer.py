from conllu_reader import corpus_batcher
from collections import defaultdict
import pandas as pd
import numpy as np

# ORDER 1_1_1_1
# ORDER 1_2_1
def find_candidats(corpus_dir_path : str) -> (pd.DataFrame) : 
    """
    FR : Génère une DataFrame de tous les couples (nom, verbe) du corpus pour lesquelles le
    verbs pointe sur le nom, et compte le nombre d'occurence de chaque candidats\n 
    EN : Creates the DataFrame of all the (noun, verb) in the corpus where the verb points towards
    the noun and count the number of occurences of each candidates.\n
    Params
    ------
    corpus_dir_path :\n
        FR : Emplacement du corpus\n
        EN : The corpus path\n
    Returns
    -------
    candidats : DataFrame[('NOUN', 'VERB') : (9,)]\n
        FR : Tableau des candidats, toutes colonnes à 0 sauf, la première représentant le
        nombre d'occurence du candidat\n
        EN : Table of the candidats, all columns at 0 except the first describing the
        number of occurences of the candidat.\n   
    """ 
    #ONEDAY manage insertion
    candidates = {}
    for data, sentences in corpus_batcher(corpus_dir_path, batch_size= 100_000):
        verbs = data.loc[data.UPosTag == 'VERB'] #select the line with verbs
        nouns = data.loc[data.UPosTag == 'NOUN'] #select the line with nouns
        #keeps the couples noun-verb where the noun point to the verb
        candidats = (pd.merge(nouns, verbs, left_on=['SId', 'Head']
                              , right_on=['SId', 'Id']
                              , suffixes=['_n', '_v']))[['Lemma_n', 'Lemma_v']]
        # count the number of occurences
        for row in candidats.to_numpy():
            candidates.setdefault((row[0], row[1]), [0, 0, 0, 0, 0, 0, 0, 0, 0])[0]+=1
    #generate the candidates dataframe
    candidates = pd.DataFrame(data = candidates).transpose()
    candidates.index.names = ['NOUN', 'VERB']
    candidates.set_axis(['N', 'P', 'P_n', 'P_v', 'dist', 'P_obl', 'P_subj', 'P_obj', 'P_other'], axis = 1, inplace= True)
    return candidates
    
#ORDER 1_1_1_2
def compute_features(corpus_dir_path: str, candidates: pd.DataFrame) -> pd.DataFrame:
    """
    FR : Calcul diverses mesures pour chacun des candidats en se basant sur le corpus\n
    EN : Compute various features for each of the candidats by using the corpus \n
    Params
    ------
    corpus_dir_path : str\n
        FR : Emplacement du corpus\n
        EN : The corpus path\n
    couples : DataFrame[('NOUN', 'VERB') : (9,)]\n
        FR : Tableau des candidats, toutes colonnes doivent être à 0,
        sauf la première représentant le nombre d'occurence du candidat\n
        EN : Table of the candidats, all columns should be at 0 
        except the first describing the number of occurences of the candidat.\n
    Returns
    -------
    features : DataFrame[('NOUN', 'VERB') : (15,)]\n
        FR : Tableau des candidats, toutes les colonnes mise à jour\n
        EN : Candidats table, all columns updated\n
    See also
    --------
    find_candidats\n
    """
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

        #### Distance nom, verbe ####
        
        # Pour chaque couple (nom, verbe) unique; fait la somme des distance (nombre de mots) entre le nom est le verbe des instances de (nom, verbe)
        t = s.assign(dist=np.abs(pd.to_numeric(s.Id_n) - pd.to_numeric(s.Id_v))).groupby(['Lemma_n', 'Lemma_v']).sum()
        add_to_col(candidates, t, 'dist')

        #### Fréquence relation ####

        # Pour chaque couple (nom, verbe) ; compte le nombre de fois où le couple est lié par une relation de type
        # obj, obl ou subj
        t = s[['DepRel_n', 'Lemma_n', 'Lemma_v']].assign(N_rel=0).groupby(['DepRel_n', 'Lemma_n', 'Lemma_v']).count()
        add_to_col(candidates, t.loc[('obj')], 'P_obj')
        add_to_col(candidates, t.loc[('obl')], 'P_obl')
        add_to_col(candidates, t.loc[('nsubj')], 'P_subj')

    #### Distance nom, verbe ####
    
    # Pour chaque couple (nom, verbe) ; calcule la distanc moyenne entre le nom et le verbe
    
    candidates.update(pd.DataFrame({'dist' : candidates['dist'] / candidates['N']}))
    
    #### Probablitées ####
    candidates.update(pd.DataFrame({'P' : candidates['N'] / candidates['N'].sum()}))
    
    candidates.update(
        pd.merge(
            candidates,
            pd.DataFrame({'P_n' : candidates['N'].groupby('NOUN').sum() / candidates['N'].sum()}),
            left_on = 'NOUN',
            right_index = True,
            suffixes=['a','']
        )['P_n'])
    
    candidates.update(
        pd.merge(
            candidates,
            pd.DataFrame({'P_v' : candidates['N'].groupby('VERB').sum() / candidates['N'].sum()}),
            left_on = 'VERB',
            right_index = True,
            suffixes=['a','']
        )['P_v'])


    # Pour chaque couple nom, verbe ; calcule la probabilité qu'un nom quelconque soit le nom en question sachant le verbe
    candidates = candidates.assign(P_n_given_v = candidates['P'] / candidates['P_v'])
    # Pour chaque couple nom, verbe ; calcule la probabilité qu'un verbe quelconque soit le verbe en question sachant le nom
    candidates = candidates.assign(P_v_given_n = candidates['P'] / candidates['P_n'])
    # Pour chaque nom ; calcule la variance de la probabilité du nom
    candidates = candidates.assign(V_n = candidates['P_n'] * (1 - candidates['P_n']))
    # Pour chaque verbe ; calcule la variance de la probabilité du verbe
    candidates = candidates.assign(V_v = candidates['P_v'] * (1 - candidates['P_v']))
    # Pour chaque couple (nom, verbe) ; calcule la covariance de la probabilité du nom et la probabilité du verbe
    candidates = candidates.assign(V = candidates['P'] - (candidates['P_n'] * candidates['P_v']))
    # Pour chaque cuple (nom, verbe) ; calcule la corrélation entre la probabilité du nom et la probabilité du verbe
    candidates = candidates.assign(corr = candidates['V'] / (np.sqrt(candidates['V_n']) * np.sqrt(candidates['V_v'])))
    # Pour chaque couple nom, verbe ; calcule le pmi
    candidates = candidates.assign(pmi = np.log(candidates['P_n_given_v'] / candidates['P_n']))

    #### Fréquence relation ####
    
    #Pour chaque couple nom, verbe ; calcule la fréquence à laquelle le nom est objet du verbe
    candidates.update(pd.DataFrame({'P_obj' : candidates['P_obj'] / candidates['N']}))
    #Pour chaque couple nom, verbe ; calcule la fréquence à laquelle le nom est oblique du verbe
    candidates.update(pd.DataFrame({'P_obl' : candidates['P_obl'] / candidates['N']}))
    #Pour chaque couple nom, verbe ; calcule la fréquence à laquelle le nom est sujet du verbe
    candidates.update(pd.DataFrame({'P_subj' : candidates['P_subj'] / candidates['N']}))
    #Pour chaque couple nom, verbe ; calcule la fréquence à laquelle le nom n'est, ni objet, oblique ou sujet du verbe
    candidates.update(pd.DataFrame({'P_other' : 1 - (candidates['P_obj'] + candidates['P_obl'] + candidates['P_subj'])}))
    
    candidates = candidates.drop(['N'], axis=1)
    return candidates.fillna(0)

#ORDER 1_1_2_1
def find_all_patterns(corpus_dir_path : str) -> list:
    """
    FR : Trouve tout les patrons syntaxique du corpus \n
    EN : Find all the syntaxical patterns of the corpus\n
    Params
    ------
    corpus_dir_path : str\n
        FR : Emplacement du corpus\n
            EN : The corpus path\n
    Returns
    -------
    all_patterns : list[(Sid : int, (word1 :str, word2 :str, pattern : list[pos : str,]))]\n
        FR : Liste des patrons syntaxique avec l'id de leur phrase
        et du mot avant et après le patron\n
        EN : Liste of the syntaxical pattern with th id ofthe sentence
        and the word before and after the pattern \n
    """
    d = {}
    for data, sentences in corpus_batcher(corpus_dir_path):
        verbes = data.loc[data.UPosTag == 'VERB'] 
        nouns = data.loc[data.UPosTag == 'NOUN']
        candidates = pd.merge(nouns.reset_index(), verbes.reset_index()
                     , left_on=['SId', 'Head']
                     , right_on=['SId', 'Id']
                     , suffixes = ['_n', '_v']).set_index('SId')
        all_patterns = []
        temp = []
        sentence_sid = -1
        gen = couples_per_sentences(candidates)
        for (Sid, Id), pos in data['UPosTag'].iteritems():
            try :
                while(sentence_sid < Sid):
                    sentence_sid, sentence_couples = next(gen)
                    all_patterns.append((sentence_sid, sentence_couples))
                    #print(si, Sid)
                for curr_couple in sentence_couples:
                    if int(Id) > curr_couple[0] and int(Id) < curr_couple[1]:
                        curr_couple[2].append(pos)
            except :
                pass 
        return all_patterns

# ORDER 1_1_2_1_1
def couples_per_sentences(couples):
    """
    FR : Génère pour chaque phrase une liste des couples (nom,verbe) ou (verbe,nom) 
    trié par ordre d'apparition de l'element du couple apparraisant le premier\n
    EN : Yield for each sentence a list of the couples (noun,verb) or (verb, noun) ordered
    by apparition order of the first element of the couple to appear\n
    Parameters
    ----------
    couples : Dataframe\n
        FR : Tableau des couples (nom,verbe) -unique- du corpus\n
        EN : Table of the -unique- couples (noun,verb) of the corpus\n
    Yields
    ------
    Sid : int
        FR : identifiant de la phrase\n
        EN : Sentence id\n
    patterns : list[(word1 : str, word2 :str, [])]\n
        FR : Identifiant des mots avant et après le patron
        ainsi qu'une liste vide pour y mettre le patron\n
        EN : Ids of the words before and after the pattern, and an empty list to put the pattern\n
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

# ORDER 1_1_2_2
def get_most_frequent_patterns(all_patterns : list)-> list:
    """
    FR : Réduit la liste des patrons aux 20 patrons les plus fréquent 
    et retire les identifiants.
    EN : Filter the list of patterns to the 20 most frequent patterns\n
    and remove the ids.\n
    Params
    ------
    all_patterns : list[(Sid : int, (word1 :str, word2 :str, pattern : list[pos : str,]))]\n
        FR : Liste des patrons syntaxique avec l'id de leur phrase
        et du mot avant et après le patron\n
        EN : Liste of the syntaxical pattern with th id ofthe sentence
        and the word before and after the pattern \n
    Returns
    -------
    frequent_pattern : list[pattern : list[pos : str]]\n
        FR : liste des 20 patrons les plus fréquent\n
        EN : list of the 20 most frequent pattern\n
    """
    dic = defaultdict(int)
    for _,i in all_patterns:
        for _,_,pattern in i:
            dic[tuple(pattern)]+=1
    return [a[0] for a in sorted([(k,v) for k,v in dic.items()], key = lambda x: x[1], reverse=True)[:20]]

#ORDER 1_1_2
def get_candidats_pattern_frequency(corpus_dir_path :str) -> pd.DataFrame:
    """
    FR : Calcule pour chaque candidat du corpus la fréquence des patrons fréquent\n
    EN : Compute for each candidat of the corpus the frequency of the frequent patterns\n
    Params
    ------
    corpus_dir_path : str\n
        FR : Emplacement du corpus\n
        EN : The corpus path\n
    Returns
    -------
    frequent_pattern_frequency : DataFrame\n
        FR : Tableau des candidats et leur fréquence associer à chaque patron fréquent\n
        EN : Table of the candidats and their associated frequence to each frequent pattern.\n
    """
    all_patterns = find_all_patterns(corpus_dir_path)
    frequent_patterns = get_most_frequent_patterns(all_patterns)
    dic = defaultdict(lambda : defaultdict(int))
    n = 0
    for data, sentences in corpus_batcher(corpus_dir_path):
        last = data.iloc[-1].name[0]
        for Sid, patterns in all_patterns[n:]:
            if Sid > last:
                break
            #ONEDAY should add a pattern "other" frequence would be more accurate from it
            for id_w1, id_w2, pattern in patterns:
                if tuple(pattern) in frequent_patterns:
                    lemma_w1 = data.loc[(Sid,str(id_w1))]['Lemma']
                    lemma_w2 = data.loc[(Sid,str(id_w2))]['Lemma']
                    if data.loc[(Sid,str(id_w1))]['UPosTag'] == 'VERB':
                        dic[(lemma_w2, lemma_w1)][tuple(pattern)] +=1
                    else :
                        dic[(lemma_w1, lemma_w2)][tuple(pattern)] +=1
            n+=1
    res = pd.DataFrame.from_dict(dic, orient='index').fillna(0)
    res.columns = frequent_patterns
    res.index.names = ('NOUN', 'VERB')
    res = res.div(res.sum(axis=1), axis=0)
    return res

#ORDER 1_1_1_2_1
def add_to_col(df_to_update : pd.DataFrame, to_add : pd.Series, col_to_update_name):
    """
    FR : Ajoute les valeurs de to_add à la colonne col_to_update_name de la DataFrae df_to_update.
    EN : Add the values of to_add to the col_to_update_name column of the df_to_update DataFrame
    """
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

# ORDER 1_1_1
def get_features(corpus_dir_path : str) -> pd.DataFrame:
    """
    FR : Trouve tout les candidats du corpus, puis calcul leurs mesures\n
    EN : Find all candidats of the corpus then compute their features\n
    Params
    ------
    corpus_dir_path : str\n
        FR : Emplacement du corpus\n
        EN : The corpus path\n
    Returns
    -------
    features : DataFrame\n
        FR : Tableau des candidats et de leurs mesures\n
        EN : Table of the candidats and their features\n
    """
    c = find_candidats(corpus_dir_path)
    features = compute_features(corpus_dir_path, c)
    return features
























