from nltk.corpus.reader.conll import ConllCorpusReader


keys = ['word', 'lemma', 'POS', 'A', 'flexion', 'refs', 'B', 'C', 'D']
prefix_len = len('# text = ')


class Conllu_file:
    def __init__(self, file_path, limit=0):
        self.sentences = []
        with open(file_path, encoding='utf-8') as file:
            for line in file:
                self.sentences.append(Conllu_sentence(line, file))
                if len(self.sentences) == limit : break
    def __repr__(self):
        st = ''
        for sentence in self.sentences:
            st += str(sentence)
        return st

class Conllu_sentence:
    def __init__(self,sentence, file):
        self.rows = {}
        self.sentence = sentence[prefix_len:]
        for line in file:
            if line == '\n' or line == '\n\r' : break
            cells = line[:-1].split('\t')
            self.rows[cells[0]] = {k:v for k,v in zip(keys,cells[1:])}
    def __repr__(self):
        print(len(self.rows))
        st = self.sentence + '\n'
        for k,row in self.rows.items():
            st += k+'\t|\t' 
            for cell in row.values():
                st += cell + '\t'
            st += '\n'
        return st+'\n'
