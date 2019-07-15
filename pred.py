import pandas as pd
import measurer
from codecs import encode
from utilities import drive_cached
import lsa as LSA # TODO Rename
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import cohen_kappa_score, recall_score, f1_score, precision_score, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt




TRAIN_DIR = 'Train'
TEST_DIR = 'Test'
CORPUS_DIR = 'Corpus'
TEMP_DIR = 'temp'
SAVE_DIR = 'Save'
DEV_DIR = "Dev"



#TODOC
#ORDER 1_1
def fetch_regular_features(corpus_dir_path : str ) -> (
	pd.DataFrame, measurer.patronateur, pd.DataFrame, pd.DataFrame):
	"""
	FR : Récupère les mesures standard tel que les probabilités, distances, patron syntaxique
	pour chaque candidat du corpus\n
	EN : Fetch regular features such as probabilities, distances, syntaxical pattern for
	each candidat of the given corpus \n
	Params
	------
		corpus_dir_path : str\n
			FR : Emplacement du corpus\n
            EN : The corpus path\n
	Returns
	-------
		features : DataFrame\n
			FR : Tableau des candidats et de leur mesures.\n
			EN : Table of the candidats and their features.\n
		patronateur : patronateur\n
			FR :\n
			EN :\n
		lsa_noun : DataFrame\n
			FR :\n
			EN :\n
		lsa_verb : DataFrame :
			FR :\n
			EN :\n

	"""
	
	corpus_id = encode(str.encode(corpus_dir_path), 'hex').decode()+'.pkl'
	
	get_features = drive_cached(measurer.get_features, 'features'+corpus_id)
	features = get_features(corpus_dir_path)

	patronateur = measurer.patronateur()
	patronateur.fit(corpus_dir_path)
	patron = patronateur.to_rename(corpus_dir_path)
	patron = patron.reset_index().set_index(['NOUN', 'VERB'])
	features = pd.merge(features, patron, how = 'left', left_index=True, right_index=True).fillna(0)
	
	tmp = LSA.temp(corpus_dir_path)
	lsa = pd.DataFrame(tmp.lsa, index= tmp.word_id)
	lsa.columns.name = 'WORD'

	exps_lsa = drive_cached(LSA.sshh,'exps'+corpus_id)(features, tmp)

	lsa_noun = pd.merge(features, lsa, how='left', left_on='NOUN', right_index=True).iloc[:,-100:].fillna(0)
	lsa_verb = pd.merge(features, lsa, how='left', left_on='VERB', right_index=True).iloc[:,-100:].fillna(0)
	len_v = pd.DataFrame(lsa_verb.abs().sum(axis=1))
	len_v.columns = ['len_v']
	len_n = pd.DataFrame(lsa_noun.abs().sum(axis=1))
	len_n.columns = ['len_n']
	
	features = pd.merge(features, len_v, left_index=True, right_index=True)
	features = pd.merge(features, len_n, left_index=True, right_index=True)
	dist_noun = pd.DataFrame(cosine_similarity(
			exps_lsa.loc(axis=0)[lsa_noun.index].fillna(0).sort_index(),
			lsa_noun.sort_index()
			).diagonal(),
		index=lsa_noun.sort_index().index)
	dist_noun.columns = ['dist_noun']
	features = pd.merge(features, dist_noun, left_index=True, right_index=True)
	# TODO replace lox with reindex
	dist_verb = pd.DataFrame(cosine_similarity(
			exps_lsa.loc(axis=0)[lsa_verb.index].fillna(0).sort_index(),
			lsa_verb.sort_index()
			).diagonal(),
		 index=lsa_verb.sort_index().index)
	dist_verb.columns = ['dist_verb']
	features = pd.merge(features, dist_verb, left_index=True, right_index=True)
	
	features = features.assign(
			dist_relative= features['dist_noun'] / (features['dist_noun'] + features['dist_verb'])
		).fillna(0.5)
	return features, patronateur, lsa_noun, lsa_verb

#ORDER 1_2
#ORDER 1_5
def load_shat(corpus_dir_path, truth_dir_path, features):
	c = features.loc[measurer.find_candidats(corpus_dir_path)[2].index].dropna()

	#TODO automate truth
	truth = pd.read_csv(truth_dir_path, encoding='utf-8')
	truth = truth.set_index(['NOUN', 'VERB']).assign(isLVC='YES')
	features = pd.merge(c, truth, how='left', left_index= True, right_index= True).fillna('NO')
	y = features[['isLVC']]
	X = features.drop(['isLVC'], 1)
	return X,y





# TODO custom KFold in order to skew the data
# TODOC
class custom_KFold():
	def __init__(n_splits=2, ratio= 0.5):
		self.n_splits=n_splits
		self.ratio=ratio
	def split(X, y):
		return 

class custom_classifier():
	#ORDER 1_3
	def __init__(self, lsa_noun, lsa_verb):
		self.lsa_noun = lsa_noun
		self.lsa_verb = lsa_verb
	#ORDER 1_4
	def fit(self, X, y):
		vecs = self.lsa_noun
		vecs = pd.merge(vecs, y , left_index= True, right_index= True)[lambda x: x.isLVC=='YES']
		self.noun_vec = vecs.mean(axis=0).to_frame().transpose()
		tmp = pd.DataFrame(
		cosine_similarity(
				self.noun_vec
				, self.lsa_noun.fillna(0)
		)[0], index=self.lsa_noun.index)
		
		X_tmp = pd.merge(X, tmp, left_index=True, right_index=True)
		
		vecs = self.lsa_verb
		vecs = pd.merge(vecs, y , left_index= True, right_index= True)[lambda x: x.isLVC=='YES']
		self.verb_vec = vecs.mean(axis=0).to_frame().transpose()
		tmp = pd.DataFrame(
		cosine_similarity(
				self.verb_vec
				, self.lsa_verb.fillna(0)
		)[0], index=self.lsa_verb.index)
		X_tmp = pd.merge(X_tmp, tmp, left_index=True, right_index=True)
		
		X_tmp = X_tmp.drop(['N', 'T', 'supp', 'N_n', 'T_n', 'supp_n'
				, 'N_v', 'T_v', 'supp_v'
				,"conf_v_n", 'conf_n_v'], 
				axis=1)

		self.classifier = GradientBoostingClassifier(n_estimators=500,
		learning_rate=0.15, min_samples_split=3, max_leaf_nodes=15
		, loss='exponential')
		shufling = pd.merge(X_tmp, y, left_index=True, right_index=True)
		shufling.sample(frac=1)
		y_tmp = shufling[['isLVC']]
		X_tmp = shufling.drop(['isLVC'], 1)
		self.classifier.fit(X_tmp, y_tmp['isLVC'])
	#ORDER 1_6_1
	def pred(self, X):
		tmp = pd.DataFrame(
		cosine_similarity(
				self.noun_vec
				, self.lsa_noun.fillna(0)
		)[0], index=self.lsa_noun.index)
		X_tmp = pd.merge(X, tmp, left_index=True, right_index=True)
		
		tmp = pd.DataFrame(
		cosine_similarity(
				self.verb_vec
				, self.lsa_verb.fillna(0)
		)[0], index=self.lsa_verb.index)
		X_tmp = pd.merge(X_tmp, tmp, left_index=True, right_index=True)
		
		X_tmp = X_tmp.drop(['N', 'T', 'supp', 'N_n', 'T_n', 'supp_n'
				, 'N_v', 'T_v', 'supp_v'
				,"conf_v_n", 'conf_n_v'], 
				axis=1)
		return X_tmp
	#ORDER 1_6
	def predict(self, X):
		X = self.pred(X)
		return pd.DataFrame(self.classifier.predict(X), index=X.index)
	def predict_proba(self, X):
		X = self.pred(X)
		return pd.DataFrame(self.classifier.predict_proba(X), index=X.index, columns=self.classifier.classes_)
	def predict_threshold(self, X, threshold=0.5):
		res = self.predict_proba(X)
		#TODO finish

def main2():
	#ORDER 1
	features, patronateur, lsa_noun, lsa_verb = fetch_regular_features(DEV_DIR)
	
	X_train, y_train = load_shat(TRAIN_DIR, 'truth.csv',features)
	classifier = custom_classifier(lsa_noun, lsa_verb)
	classifier.fit(X_train, y_train)

	
	X_test, y_test = load_shat(TEST_DIR, 'test_truth.csv', features)

	res = classifier.predict(X_test)
	
	print(res.loc[res[0] == 'YES'])
	print(res.assign(count=1).groupby(0).count())
	print('kappa', cohen_kappa_score(y_test.sort_index(), res.sort_index()))
	print('f1', f1_score(y_test.sort_index(),res.sort_index(),  pos_label='YES'))
	print('precision', precision_score(y_test.sort_index(), res.sort_index(), pos_label='YES'))
	print('recall', recall_score(y_test.sort_index(),res.sort_index(), pos_label='YES'))
	
	precision, recall, threshold = precision_recall_curve(
	  		y_test.sort_index(), classifier.predict_proba(X_test)['YES'].sort_index(), pos_label='YES')
	f1 = [(2 * p * r) / (p + r) if p+r > 0 else 0 for p, r in zip(precision, recall)]
	sns.scatterplot(x="x", y="y", data=pd.DataFrame(zip(precision, threshold), columns=['y','x'])
	, marker='+', alpha=0.5, label='precision')
	sns.scatterplot(x="x", y="y", data=pd.DataFrame(zip(recall,threshold), columns=['y','x'])
	, marker='+', alpha=0.5, label='recall')
	sns.scatterplot(x="x", y="y", data=pd.DataFrame(zip(f1, threshold), columns=['y','x'])
	, marker='+', alpha=0.5, label='f1')
	plt.show()


		

main2()