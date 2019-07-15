from conllu_reader import corpus_batcher
from collections import defaultdict
import pickle
from os import path
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from codecs import encode
import pandas as pd
from utilities import drive_cached

SAVE_DIR = 'Save'


def f_for_defaultdict():
    """
    FR : fonction nécessaire afin de créer des defaultdic<defaultdict<int>> pickable.\n
    EN : function necessary in order to generate pickable defaultdic<defaultdict<int>>\n
    """
    return defaultdict(int)

def saved(to_save, file_path):
    """
    FR : Pickle l'objet to_save à l'emplacement spécifier puis retourne l'objet sans le modifier.\n
    EN : Pickle the given object in the specified field then return the object without modifying it.\n
    Parameters
    ----------
    to_save : Pickable object\n
        FR : L'objet à sauvegarder, celui ci doit pouvoir être pickler, sinon il ne sera pas sauvé.\n
        EN : The object to be saved, it has to be Picklable, otherwise it won't be saved.\n
    file_path : str\n
        FR : Emplacement du fichier généré.\n
        EN : Path of the generated file\n
    Returns
    -------
    object\n
        FR : L'objet fourni en entré\n
        EN : The given object\n
    """
    with open(file_path, 'wb') as output:
        pickle.dump(to_save, output)
    return to_save

#ORDER 1_1_4_1
def corpus_cooccurrences(corpus_dir_path):
    """
    FR : Compte les cooccurrences entre paires de mots dans le corpus donné 
    et les retournes sous forme de dict.\n
    EN : Count the cooccurrences between pair of words in the given corpus and return the in a dict.\n
    Parameters
    ----------
    corpus_dir_path : str\n
        FR : Emplacement du corpus\n
        EN : The corpus path\n
    Returns
    -------
    dict<dict<int>>\n
        {word1 : {word1 : nb_cooc_word1_word2}}\n
        FR : Dictionnaire contenant les paire toutes les paires de mots rencontrées dans le corpus
        ainsi que leur nombre d'occurences respectives.\n
        EN : Dictionary with all the pair of word seen in the corpus
        and the number of time each pair has been sighted.\n
    """
    cooc_dic = defaultdict(f_for_defaultdict)
    sid = 0 # Id de la phrase
    sentence_first_word_id = 0 
    for data, sentences in corpus_batcher(corpus_dir_path):
        del sentences
        # Transforme la DataFrame en matrice numpy pour optimiser la lecture iterrative
        data = data[['Lemma']].reset_index().to_numpy()
        for n,row in enumerate(data):
            # parcour tout les mots de la phrase avant le mot actuel (n)
            for a in data[sentence_first_word_id:n]:
                cooc_dic[row[2]][a[2]]+=1
                cooc_dic[a[2]][row[2]]+=1
            # met à jour l'id du premier mot de la phrase lorsque l'on change de phrase
            if sid != row[0] : 
                sid = row[0]
                sentence_first_word_id = n
    return {word1 : {word2 : count for word2, count in dic.items()} for word1, dic in cooc_dic.items()}

# TODO Remove, use utilities
def load_or_compute_dec(func, file_name):
    """
    FR : Décore une fonction afin d'enregistrer son résultat, ou de le charger si celui-ci
    à déjà était enregistré.\n
    EN : Decore a function in order to save its result, or to load the result if it has already been
    saved.\n
    Parameters
    ----------
    func : callable\n
        FR : La fonction que l'on désire décorer.\n
        EN : The function to be decored.\n
    file_name : str\n
        FR : Le nom sous lequel le résultat sera enregistré / chargé.\n
        EN : The name under which the result will be saved / loaded.\n
    Returns
    -------
        callable :\n
            FR : La function décoré.\n
            EN : the decored function\n
    Examples
    --------
    >>> def do_something(...):
    >>>     print('some side effect')
    >>>     return something

    >>> decored_do_something = load_or_compute_dec(do_something, 'a_name') 
    >>> first_call = decored_do_someting(...) # do_something and save it
    some side effect
    # load saved result without calling do_something again
    >>> second_call = decored_do_someting(...) 
    # results of the first and second call are the same
    >>> first_call == second_call 
    True
    """
    def intern(*args, **kwargs):
        file_path = path.join(SAVE_DIR, file_name)
        if path.isfile(file_path) :
            with open(file_path, 'rb') as file:
                return pickle.load(file)
        else :
            res = func(*args, **kwargs)
        return saved(res, file_path)
    return intern
# ORDER 1_1_4_2
def dic_to_mat(cooc_dic):
    """
    FR : Transforme un dictionnaire de cooccurrences en une matrices creuse de coccurrences\n
    EN : Transform a cocccurrences dictionary into a cooccurrences sparse matrix\n
    Parameters
    ----------
    cooc_dic : dic<dic<int>>\n
        {word1 : {word2 : nb_cooc_word1_word2}}\n
        FR : Dictionnaire contenant les paire toutes les paires de mots rencontrées dans le corpus
        ainsi que leur nombre d'occurences respectives.\n
        EN : Dictionary with all the pair of word seen in the corpus
        and the number of time each pair has been sighted.\n
    Returns
    -------
    Sparse Matrix\n
        FR : Matrice de cooccurrence représentant les mêmes données que le dictionnaire fourni
        en entré.\n
        EN : Cooccurrences matrix representing the same data as the dictionnary given.\n
    """
    data, i, j = [] , [], []
    n = 0
    voc = {}
    sums = {k : sum(v.values()) for k,v in cooc_dic.items()}
    for k, v in cooc_dic.items():
        if True:
            if voc.setdefault(k, n) == n: n+=1
            for l, w in v.items():
                if voc.setdefault(l, n) == n : n+=1
                i.append(voc[k])
                j.append(voc[l])
                data.append(w / sums[l])
    return coo_matrix((data, (i, j))).tocsr(), voc, sums
#ORDER 1_1_5_1_1_1
def expressions_cooccurrences(corpus_dir_path, expressions):
    """
    FR : Compte les cooccurrences entre les expressions fournies en entré et les mots du corpus.\n
    EN : Count cooccurrences between the given expressions and corpus' words.\n
    Parameters
    ----------
        corpus_dir_path : str\n
            FR : Emplacement du corpus\n
            EN : The corpus path\n
        expressions : list<(str,str)>\n
            FR : liste des expressions dont on veut les coccurrences\n
            EN : list of the expressions of which we want the coccurrences\n
    Returns
    -------
    dict<dict<int>>\n
        {exp : {word1 : nb_cooc_exp_word2}}\n
        FR : Dictionnaire contenant pour chaque expression les mots avec lequelles celle-ci
        cooccurre ainsi que le nombre de coccurences respective\n
        EN : Dictionary with, for each expression, the word cooccurring with said expression and
        the number of cooccurrence
    """
    cooc_dic = defaultdict(f_for_defaultdict)
    if expressions == []: return cooc_dic
    last_word_sid = 0
    for data, sentences in corpus_batcher(corpus_dir_path):
        data = data.reset_index().to_numpy()
        word_occurrences = defaultdict(int)
        verbs = defaultdict(lambda :-1)
        nouns = defaultdict(lambda :-2)
        for n,end_row in enumerate(data):
            if last_word_sid != end_row[0] :
                if len(verbs) != 0 and len(nouns) != 0:
                    for expression in expressions:
                        if verbs[expression] == nouns[expression]:
                            for word, occurences in word_occurrences.items():
                                if word not in expression:
                                    cooc_dic[expression][word] += occurences#*C[exp]
                word_occurrences = defaultdict(int)
                verbs = defaultdict(lambda :-1)
                nouns = defaultdict(lambda :-2)
                last_word_sid = end_row[0]
            word_occurrences[end_row[3]]+=1
            
            if end_row[4] == 'NOUN':
                for expression in expressions:
                    if end_row[3] == expression[1]:
                        nouns[expression] = end_row[7]
            if end_row[4] == 'VERB':
                for expression in expressions:
                    if end_row[3] == expression[0]:
                        verbs[expression] = end_row[1]
    return cooc_dic

# TODOC
# TODO rename var
# ORDER 1_1_5_1_2
def dic_to_vec(x, voc, sums):
    """
    FR : \n
    EN : \n
    Parameters
    ----------
    
    Returns
    -------
    
    """
    data = []
    i = []
    j = []
    left = dict(voc)
    #here
    for n, (exp, cooc) in enumerate(x.items()):
        for word, cooccurrences in cooc.items():
            if word in left : left.pop(word)
            j.append(voc[word])
            i.append(n)
            if sums[word] == 0 or cooccurrences == 0:
                data.append(0)
            else:
                data.append((cooccurrences / sums[word]))# (np.log(len(sums) / y))) 
        if len(left) != 0:
            j.append(max(left.values()))
            i.append(n)
            data.append(0)
    return coo_matrix((data, (i, j)))

class cached_expressions_cooccurrences:
    """
    FR : Objet appelable (une fois initialisé, l'objet peut être appeler comme une fonction)
    comptant les cooccurrences d'une liste d'expressions avec le reste des mots du corpus\n
    EN : Callable object (once initialized, it can be called the same way a function does)
    counting the cooccurrences between a list of expressions and the words of the corpus.\n
    """
    #ORDER 1_1_4_3
    def __init__(self, corpus_dir_path):
        """
        Params
        ------
            corpus_dir_path : str\n
                FR : Emplacement du corpus\n
                EN : The corpus path\n
        """
        self.cache = {}
        self.corpus_dir_path = corpus_dir_path
    #ORDER 1_1_5_1_1
    def __call__(self, expressions):
        """
        FR : Compte ou retrouve le nombre de cooccurrences entre les expressions et
        les mots du corpus.\n
        EN : Count or retrieve the number of cooccurrences between the expressions 
        and the corpus' words\n
        Params
        ------
            expressions : list<(str,str)>\n
                FR : liste des expressions dont on veut les coccurrences\n
                EN : list of the expressions of which we want the coccurrences\n
        Returns
        -------
        dict<dict<int>>\n
            {exp : {word1 : nb_cooc_exp_word2}}\n
            FR : Dictionnaire contenant pour chaque expression les mots avec lequelles celle-ci
            cooccurre ainsi que le nombre de coccurences respective\n
            EN : Dictionary with, for each expression, the word cooccurring with said expression and
            the number of cooccurrence
        See also
        --------
        expressions_cooccurrences
        """
        # TODO rename
        # TODO replace __call__
        list_exp2 = [ex for ex in expressions if ex not in self.cache]
        e = expressions_cooccurrences(self.corpus_dir_path, list_exp2)
        for ex in list_exp2:
            self.cache[ex] = {k: v for k,v in e[ex].items()}
        return {ex : self.cache[ex] for ex in expressions}
    # ORDER 1_1_5_3_1
    def save_cache(self, name):
        """
        FR : Enregistre le cache dans un fichier.\n
        EN : SAve the cache in a file.\n
        Params
        ------
            name : str\n
                FR : nom sous lequel on désire enregistrer le cache.\n
                EN : name under which the cache is to be saved.\n
        """
        with open(path.join(SAVE_DIR, name), 'wb') as file:
            pickle.dump(self.cache, file)
    # ORDER 1_1_5_0_1
    def load_cache(self, name):
        """
        FR : Charge le cache depuis un fichier.\n
        EN : Load the cache from a file.\n
        Params
        ------
            name : str\n
                FR : nom sous lequel le cache est enregistré.\n
                EN : name under which the cache is saved.\n
        """
        file_path = path.join(SAVE_DIR, name)
        if path.isfile(file_path) :
            with open(file_path, 'rb') as file:
                self.cache = pickle.load(file)

#TODOC
class cached_expressions_vectors:
    """
    FR : Objet appelable (une fois initialisé, l'objet peut être appeler comme une fonction)
    renvoitant la représentation vectoriel des coocurrences d'une liste d'expressions\n
    EN : Callable object (once initialized, it can be called the same way a function does)
    returning the vectorial reprensation of the cooccurrences of the given list of expressions.\n
    """
    #ORDER 1_1_4_4
    def __init__(self, word_id, sums, svd, exps_cooc):
        """
        Params
        ------
            word_id : dic[str, int]
                FR :
                EN :
            sums : 

        """
        self.exps_cooc = exps_cooc
        self.cache = {}
        self.word_id = word_id
        self.sums = sums
        self.svd = svd
    # TODO rename abuchastuff
    # TODO replace __call__
    #ORDER 1_1_5_1
    def __call__(self, list_exp):
        e = self.exps_cooc(list_exp)
        list_exp2 = [ex for ex in list_exp if ex not in self.cache]
        a = dic_to_vec(e, self.word_id, self.sums)
        b = self.svd.transform(a)
        for ex, c in zip(list_exp2, b):
            self.cache[ex] = c
        return [self.cache[ex] for ex in list_exp]
    # ORDER 1_1_5_3_2
    def save_cache(self, name):
        with open(path.join(SAVE_DIR, name), 'wb') as file:
            pickle.dump(self.cache, file)
    # ORDER 1_1_5_0_2
    def load_cache(self, name):
        file_path = path.join(SAVE_DIR, name)
        if path.isfile(file_path) :
            with open(file_path, 'rb') as file:
                self.cache = pickle.load(file)
              
# TODOC 
# TODO expressions
# TODO change cache loading
# TODO rename class
class temp():
    #ORDER 1_1_4
    def __init__(self, corpus_dir_path):
        self.id =encode(str.encode(corpus_dir_path), 'hex').decode()+'.pkl'
        cooc_dic = drive_cached(
            corpus_cooccurrences, 
            'coocs'+self.id
        )(corpus_dir_path)
        self.cooc_mat, self.word_id, self.sums = dic_to_mat(cooc_dic)
        self.id_word = {v : k for k,v in self.word_id.items()}
        self.svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
        self.svd = drive_cached(self.svd.fit,
            'svd'+self.id
        )(self.cooc_mat)
        self.lsa = drive_cached(self.svd.transform,
            'lsa'+self.id
        )(self.cooc_mat)
        self.exps_cooc = cached_expressions_cooccurrences(corpus_dir_path)
        self.exps = cached_expressions_vectors(self.word_id,self.sums,self.svd, self.exps_cooc)
    # ORDER 1_1_5_3
    def save_cache(self):
        print('cooc_cache'+self.id)
        self.exps_cooc.save_cache('cooc_cache'+self.id)
        self.exps.save_cache('vec_cache'+self.id)
    # ORDER 1_1_5_0
    def load_cache(self):
        print('cooc_cache'+self.id)
        self.exps_cooc.load_cache('cooc_cache'+self.id)
        self.exps.load_cache('vec_cache'+self.id)
#TODO put in a func


#TODO rename bunchashit
#TODOC
#ORDER 1_1_5
def sshh(df, t):
    expressions = [(v,n) for n,v in df.index.tolist()]
    t.load_cache()
    t.exps(expressions) #TODO too obscure
    t.save_cache()
    res = pd.DataFrame.from_dict(t.exps.cache, orient='index')
    tmp = pd.DataFrame(res.reset_index()['index'].tolist(), index= res.index, columns=['VERB', 'NOUN'])
    res = pd.merge(res, tmp, left_index=True, right_index=True).set_index(['NOUN', 'VERB'])
    return res
#pd.DataFrame(t.lsa, index= t.word_id).to_csv(path.join(SAVE_DIR, 'lsa.csv'), sep = '\t', quoting = csv.QUOTE_NONE, escapechar = ' ', encoding='utf-8')


#t.exps.cache = tmp
#t.save_cache()

#print(pd.DataFrame({k : t.svd.transform(v) for k,v in t.exps.cache.items()}))
    
