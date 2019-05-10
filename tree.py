from conllu_reader import Conllu_file
from enum import Enum


class FLAG(Enum):
    '''FLAG are used to make evaluation function return result more readable'''
    NO = -1
    UNSURE = 0
    YES = 1


class DecisionTree:
    ''' A Decisiontree of which each node is composed a evaluation function called f,
and a dictionary with FLAGs as keys and child DecisionTree as values. If a FLAG is
not present in the dictionary it is then an exit case.'''
    def __init__(self, f, tree):
        self.f = f
        self.tree = tree
    def evaluate(self, verb, noun, sentence):
        ''' Evaluates whether a candidat is an LVC, using the DecisionTree
evaluation function then move according to the DecisionTree to the next stage'''
        res = self.f(verb, noun, sentence) # evaluate the candidats
        next_step = self.tree.get(res) # find the next step in the DecisionTree
        if next_step is None: return res.name, self.f.__name__ # exit the evaluation if no next step is necessary
        return next_step.evaluate(verb, noun, sentence) # goes to the next step if necessary
        
def find_candidats(conllu, tree):
    '''find all potential verb-noun part of an LVC and evaluate them
for now all combination of verb-noun is candidate. '''
    for sentence in conllu:
        verbs = []
        nouns = []
        for word, cells in sentence.rows.items():    
            if cells['UPosTag'] == 'VERB':
                verbs.append(word)
            if cells['UPosTag'] == 'NOUN':
                nouns.append(word)
        for verb in verbs:
            for noun in nouns:
                print(tree.evaluate(verb, noun, sentence))
        print()


def test_lvc_0(verb, noun, sentence):
    # n est abstrait :
        # web semantique
        # word embedding
    return FLAG.UNSURE

def test_lvc_1(verb, noun, sentence):
    # n a au moins 1 argument semantique, pas nécessairement dans la phrase:
        # dictionnaire
        # web semantique
        # word embedding ? (vecteur argument ?)
    print(verb)
    print(noun)
    for word, cells in sentence.rows.items():
        
    return FLAG.UNSURE

def test_lvc_2(verb, noun, sentence):
    #Le sujet du verbe est l'argument semantique du nom
        # detection sujet
            # UD
        # detection de l'argument semantique
            # UD
    return FLAG.UNSURE

def test_lvc_3(verb, noun, sentence):
    # Le verbe ne porte casi-pas de sense:
        # word embedding
    return FLAG.UNSURE

def test_lvc_4(verb, noun, sentence):
    #Le verbe peut être enlevé
        # word embedding
    return FLAG.UNSURE

def test_lvc_5(verb, noun, sentence):
    #Le sujet du verbe est la cause du nom
        # UD
        # word embedding ?
    return FLAG.UNSURE

step3 = DecisionTree(test_lvc_3,
                     {FLAG.YES : DecisionTree(test_lvc_4, {})})

step2 = DecisionTree(test_lvc_2,
                     {FLAG.YES : step3,
                      FLAG.UNSURE : step3,
                      FLAG.NO : DecisionTree(test_lvc_5, {})})

step1 = DecisionTree(test_lvc_1,
                     {FLAG.YES : step2,
                      FLAG.UNSURE : step2})

tree = DecisionTree(test_lvc_0,
                 {FLAG.YES :step1,
                  FLAG.UNSURE :step1})

conllu = Conllu_file('fr-common_crawl-164.conllu', 1)
find_candidats(conllu, tree)
