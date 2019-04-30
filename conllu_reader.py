from nltk.corpus.reader.conll import ConllCorpusReader


keys = ['word', 'lemma', 'POS', 'A', 'flexion', 'refs', 'B', 'C', 'D']

class Conllu_file:
    def __init__(self, file_path):
        conllu_sentences = []
        with open(file_path, encoding='utf-8') as file:
            for line in file:
                conllu_sentences.append(Conllu_sentence(file))
                if len(conllu_sentences) ==3 : break

class Conllu_sentence:
    def __init__(self, file):
        tokens = {}
        for line in file:
            if line == '\n' or line == '\n\r' : break
            columns = line[:-1].split('\t')
            tokens[columns[0]] = {k:v for k,v in zip(keys,columns[1:])}
        print(tokens)


Conllu_file('fr-common_crawl-164.conllu')
