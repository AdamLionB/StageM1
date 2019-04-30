from conllu_reader import Conllu_file
from enum import Enum

class FLAG(Enum):
    NO = -1
    UNSURE = 0
    YES = 1

class DecisionTree:
    def __init__(self, f, tree):
        self.f = f
        self.tree = tree
    def evaluate(self, verb, noun, sentence):
        res = self.f(verb, noun, sentence)
        todo = self.tree.get(res)
        if todo is None: return res
        return todo.evaluate(verb, noun, sentence)
        
def find_candidats(conllu):
    for sentence in conllu:
        verbs = []
        nouns = []
        for word, cells in sentence.rows.items():    
            if cells['POS'] == 'VERB':
                verbs.append(word)
            if cells['POS'] == 'NOUN':
                nouns.append(word)
        yield verbs, nouns, sentence

def evaluate_candidats(candidats, tree):
    for candidat in candidats:
        verbs, nouns, sentence = candidat
        for verb in verbs:
            for noun in nouns:
                print(tree.evaluate(verb, noun, sentence))
        print()


def test_lvc_0(verb, noun, sentence):
    return FLAG.UNSURE

def test_lvc_1(verb, noun, sentence):
    return FLAG.NO

tree = DecisionTree(test_lvc_0,
                 {FLAG.YES :
                      DecisionTree(test_lvc_1, {}),
                  FLAG.UNSURE :
                      DecisionTree(test_lvc_1, {})})

conllu = Conllu_file('fr-common_crawl-164.conllu', 10)
evaluate_candidats(find_candidats(conllu), tree)
