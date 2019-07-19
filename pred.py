import pandas as pd
import measurer
from codecs import encode
import utilities
from lsa import LSA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import cohen_kappa_score, recall_score, f1_score, precision_score, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import re
import argparse
import cupter




TRAIN_DIR = 'Train'
TEST_DIR = 'Test'
CORPUS_DIR = 'Corpus'
TEMP_DIR = 'temp'
SAVE_DIR = utilities.SAVE_DIR()
DEV_DIR = "Dev"


#TODOC
def fetch_regular_features(corpus_dir_path : str, reset=False ) -> (
	pd.DataFrame, pd.DataFrame, pd.DataFrame):
	"""
	FR : Récupère les mesures standards tel que les probabilités, distances, patron syntaxique
	pour chaque candidat du corpus\n
	EN : Fetch regular features such as probabilities, distances, syntaxical pattern for
	each candidat of the given corpus \n
	Params
	------
	corpus_dir_path : str\n
		FR : Emplacement du corpus\n
		EN : The corpus path\n
	reset : bool\n
		#TODOC
	Returns
	-------
	features : DataFrame\n
		FR : Tableau des candidats et de leur mesures.\n
		EN : Table of the candidats and their features.\n
	lsa_noun : DataFrame\n
		FR : Tableau des candidats et du vecteur de leur nom\n
		EN : Table of the candidats and their noun vector\n
	lsa_verb : DataFrame\n
		FR : Tableau des candidats et du vecteur de leur verbe\n
		EN : Table of the candidats and their verb vector\n
	exps_lsa : DataFrae\n
		#TODO
	"""
	corpus_id = encode(str.encode(corpus_dir_path), 'hex').decode()+'.pkl'
	
	get_features = utilities.drive_cached_func(measurer.get_features, 'features'+corpus_id, reset)
	features = get_features(corpus_dir_path)

	get_pattern_frequency = utilities.drive_cached_func(
		measurer.get_candidats_pattern_frequency, 'patterns'+corpus_id, reset)
	patterns = get_pattern_frequency(corpus_dir_path)
	features = pd.merge(features, patterns, how = 'left', left_index=True, right_index=True).fillna(0)
	
	tmp = LSA(corpus_dir_path)
	lsa = pd.DataFrame(tmp.lsa, index= tmp.word_id)
	lsa.columns.name = 'WORD'

	exps_lsa = utilities.drive_cached_func(tmp,'exps'+corpus_id, reset)(features)

	lsa_noun = pd.merge(features, lsa, how='left', left_on='NOUN', right_index=True).iloc[:,-100:].fillna(0)
	lsa_verb = pd.merge(features, lsa, how='left', left_on='VERB', right_index=True).iloc[:,-100:].fillna(0)
	len_v = pd.DataFrame(lsa_verb.abs().sum(axis=1))
	len_v.columns = ['len_v']
	len_n = pd.DataFrame(lsa_noun.abs().sum(axis=1))
	len_n.columns = ['len_n']
	
	features = pd.merge(features, len_v, left_index=True, right_index=True)
	features = pd.merge(features, len_n, left_index=True, right_index=True)
	dist_noun = pd.DataFrame(utilities.cos_similarities(
		exps_lsa.loc(axis=0)[lsa_noun.index].fillna(0).sort_index().values,
		lsa_noun.sort_index().values
	), index=lsa_noun.sort_index().index)

	dist_noun.columns = ['dist_noun']
	features = pd.merge(features, dist_noun, left_index=True, right_index=True)
	# TODO replace loc with reindex
	dist_verb = pd.DataFrame(utilities.cos_similarities(
		exps_lsa.loc(axis=0)[lsa_verb.index].fillna(0).sort_index().values,
		lsa_verb.sort_index().values
	), index=lsa_verb.sort_index().index)

	dist_verb.columns = ['dist_verb']
	features = pd.merge(features, dist_verb, left_index=True, right_index=True)
	
	features = features.assign(
			dist_relative= features['dist_noun'] / (features['dist_noun'] + features['dist_verb'])
		).fillna(0.5)
	return features, lsa_noun, lsa_verb, exps_lsa


#TODOC
def fetch_predictable_candidates(corpus_dir_path : str, features : pd.DataFrame):
	return features.reindex(measurer.find_candidats(corpus_dir_path).index).dropna()
	

def fetch_annotated_candidates(annotated_corpus_dir_path :str, features : pd.DataFrame
	) -> (pd.DataFrame, pd.DataFrame):
	"""
	FR : Recupère parmis les candidats ceux qui apparraisent dans corpus annoté\n
	EN : Fetch from the candidats those which appear in the annotated corpus\n
	Params
	------
	annotated_corpus_dir_path : str\n
		FR : Emplacement du corpus annoté\n
		EN : The annotated corpus path\n
	features : str\n
		FR : Tableau des candidats et leurs mesures\n
		EN : Table of the candidtas and their features\n
	Returns
	-------
	X : DataFrame\n
		FR : Tableau des candidats restant et leur mesure\n
		EN : Table of the selected candidats and their features\n
	y : DataFrame\n
		FR : Tableau des candidats restant et leur annotation\n
		EN : Table of the selected candidats and their annotation\n
	"""
	predictable_candidates = fetch_predictable_candidates(annotated_corpus_dir_path, features)

	truth = cupter.get_LVCs(annotated_corpus_dir_path).assign(isLVC='YES')
	features = pd.merge(predictable_candidates, truth, how='left', left_index= True, right_index= True).fillna('NO')
	y = features[['isLVC']]
	X = features.drop(['isLVC'], axis=1)
	return X,y



#TODO Find a better name
class custom_classifier():
	def __init__(self, lsa_noun : pd.DataFrame, lsa_verb : pd.DataFrame, exps_lsa : pd.DataFrame):
		"""
		Params
		------
		lsa_noun : DataFrame\n
			FR : Tableau des candidats et du vecteur de leur nom\n
			EN : Table of the candidats and their noun vector\n
		lsa_verb : DataFrame\n
			FR : Tableau des candidats et du vecteur de leur verbe\n
			EN : Table of the candidats and their verb vector\n
		exps_lsa : DataFrame\n
			#TODOC
		"""
		self.lsa_noun = lsa_noun
		self.lsa_verb = lsa_verb
		self.exps_lsa = exps_lsa
	def fit(self, X : pd.DataFrame, y : pd.DataFrame):
		"""
		FR : Entraine le modèle de prédiction\n
		EN : Fit the prediction model\n
		Params
		------
		X : DataFrame\n
			FR : Tableau des candidats et leurs features\n
			EN : Table of the candidats and their features\n
		y : DataFrame\n
			FR : Tableau des candidats et leur annotation\n
			EN : Table of the candidats and their annotation\n
		"""
		vecs = self.lsa_noun
		vecs = pd.merge(vecs, y , left_index= True, right_index= True)[lambda x: x.isLVC=='YES']
		self.noun_vec = vecs.mean(axis=0).to_frame().transpose()
		vecs = self.lsa_verb
		vecs = pd.merge(vecs, y , left_index= True, right_index= True)[lambda x: x.isLVC=='YES']
		self.verb_vec = vecs.mean(axis=0).to_frame().transpose()
		vecs = self.lsa_verb
		vecs = pd.merge(vecs, y , left_index= True, right_index= True)[lambda x: x.isLVC=='YES']
		self.exps_vec = vecs.mean(axis=0).to_frame().transpose()
		
		X_tmp = self.compute_average_vector_features(X)
		

		self.classifier = GradientBoostingClassifier(n_estimators=500,
		learning_rate=0.15, min_samples_split=3, max_leaf_nodes=15
		, loss='exponential')

		#shuffle
		shufling = pd.merge(X_tmp, y, left_index=True, right_index=True)
		shufling.sample(frac=1)
		y_tmp = shufling[['isLVC']]
		X_tmp = shufling.drop(['isLVC'], 1)

		self.classifier.fit(X_tmp, y_tmp['isLVC'])
		return self
	def compute_average_vector_features(self, X : pd.DataFrame) -> pd.DataFrame:
		"""
		FR : Calcul pour chaque candidats des mesures en rapport à la distance entre les
		vecteurs du candidats et le vecteur moyen des candidats positif\n
		EN : Compute for each candidats features regarding the distance between the vector of the
		candidats and the average vector of postitive candidats\n
		Params
		------
		X : DataFrame\n
			FR : Tableau des candidats et leurs features\n
			EN : Table of the candidats and their features\n
		Returns
		-------
		new_X : DataFrame\n
			FR : Tableau X auquel on a ajouté quelques mesures\n
			EN : Table X with a few more features\n
		"""
		tmp = pd.DataFrame(
		cosine_similarity(
				self.noun_vec
				, self.lsa_noun.fillna(0)
		)[0], index=self.lsa_noun.index)
		tmp.columns= ['noun']
		X_tmp = pd.merge(X, tmp, left_index=True, right_index=True)

		tmp = pd.DataFrame(
		cosine_similarity(
				self.verb_vec
				, self.lsa_verb.fillna(0)
		)[0], index=self.lsa_verb.index)
		tmp.columns= ['verb']
		X_tmp = pd.merge(X_tmp, tmp, left_index=True, right_index=True)

		tmp = pd.DataFrame(
		cosine_similarity(
				self.exps_vec
				, self.exps_lsa.fillna(0)
		)[0], index=self.exps_lsa.index)
		tmp.columns= ['exp']
		X_tmp = pd.merge(X_tmp, tmp, left_index=True, right_index=True)
		return X_tmp
	def predict(self, X : pd.DataFrame) -> pd.DataFrame:
		"""
		FR : Prédit la classe des candidats fournie en X\n
		EN : Predict X's candidats' classe\n
		Params
		------
		X : DataFrame\n
			FR : Tableau des candidats et leurs features\n
			EN : Table of the candidats and their features\n
		Returns
		-------
		y_predicted : DataFrame\n
			FR : Tableau des candidats et leur classe prédite\n
			EN : Table of the candidats and their predicted class\n
		"""
		X = self.compute_average_vector_features(X)
		return pd.DataFrame(self.classifier.predict(X), index=X.index)
	def predict_proba(self, X):
		#TODOC
		X = self.compute_average_vector_features(X)
		return pd.DataFrame(self.classifier.predict_proba(X), index=X.index, columns=self.classifier.classes_)

class Manager():
	#TODOC
	def set_dir(self, corpus_dir, train_dir, test_dir, save_dir):
		self.CORPUS_DIR = corpus_dir
		self.TRAIN_DIR = train_dir
		self.TEST_DIR = test_dir
		self.SAVE_DIR = save_dir
		utilities.SAVE_DIR.set(save_dir)
	def train(self, reset_features = False, reset_classifier= False):
		self.compute_features(reset_features)
		self.fit(reset_classifier)
	def compute_features(self, reset = False):
		self.features, self.lsa_noun, self.lsa_verb, self.exps_lsa = fetch_regular_features(
			self.CORPUS_DIR, reset)
	def fit(self, reset=False):
		def intern():
			X_train, y_train = fetch_annotated_candidates(self.TRAIN_DIR, self.features)
			classifier = custom_classifier(self.lsa_noun, self.lsa_verb, self.exps_lsa)
			return classifier.fit(X_train, y_train), y_train
		corpus_id = encode(str.encode(self.CORPUS_DIR+self.TRAIN_DIR), 'hex').decode()+'.pkl'
		self.classifier, self.y_train = utilities.drive_cached_func(intern, 'classifier'+corpus_id, reset)()
	def predict(self):
		X_test = fetch_predictable_candidates(self.CORPUS_DIR, self.features)
		return self.classifier.predict(X_test).drop(self.y_train.index).loc[lambda df : df[0] == 'YES']	
	def evaluate(self):
		X_test, y_test = fetch_annotated_candidates(self.TEST_DIR, self.features)
		corpus_id = encode(str.encode(self.TEST_DIR), 'hex').decode()+'.pkl'
		nb_tot = len(utilities.drive_cached_func(cupter.get_LVCs, 'truth'+corpus_id)(self.TEST_DIR))
		nb_inter = len(y_test.loc[y_test['isLVC'] == 'YES'])
		
		res = self.classifier.predict(X_test)
		a = pd.merge(y_test,res
			, how='outer', left_index=True, right_index=True).loc[
				lambda df : (df['isLVC'] == 'YES') | (df[0] == 'YES')
			]
		to_predict = pd.merge(self.y_train, y_test,
				how='outer', left_index=True, right_index=True).drop('isLVC_x', axis= 1).drop(self.y_train.index).fillna('~')
		a = pd.merge(res, to_predict
			, how='inner', left_index=True, right_index=True)
		print(a.loc[a['isLVC_y']== 'YES'])
		conf_mat = a.assign(count=1).groupby([0, 'isLVC_y']).count()
		print(conf_mat)
		try :
			print('r', conf_mat.loc[('YES','YES')] / (conf_mat.loc[('YES','YES')] + conf_mat.loc[('NO','YES')]))
		except :
			print('r', 0)
		try :
			print('p', conf_mat.loc[('YES','YES')] / (conf_mat.loc[('YES','YES')] + conf_mat.loc[('YES','NO')]))
		except :
			print('p', 0)
		print('kappa', cohen_kappa_score(y_test.sort_index(), res.sort_index()))
		r = recall_score(y_test.sort_index(),res.sort_index(), pos_label='YES')
		p = precision_score(y_test.sort_index(), res.sort_index(), pos_label='YES')
		print('precision', p)
		print('recall',r)
		print('f1', (2 * p * r) / (p + r))
		true_r = r  * nb_inter / nb_tot
		print('true recall', true_r)
		print('true f1', (2 * p * true_r) / (p + true_r))

		precision, recall, threshold = precision_recall_curve(
				y_test.sort_index(), self.classifier.predict_proba(X_test)['YES'].sort_index(), pos_label='YES')
		f1 = [(2 * p * r) / (p + r) if p+r > 0 else 0 for p, r in zip(precision, recall)]
		true_recall = [r * nb_inter / nb_tot for r in recall]
		true_f1 = [(2 * p * r) / (p + r) if p+r > 0 else 0 for p, r in zip(precision, true_recall)]
		plt.subplot(1,1,1)
		sns.set_style('whitegrid')
		sns.lineplot(x="x", y="y", data=pd.DataFrame(zip(precision, threshold), columns=['y','x'])
		, label='precision')
		sns.lineplot(x="x", y="y", data=pd.DataFrame(zip(recall,threshold), columns=['y','x'])
		, label='recall')
		sns.lineplot(x="x", y="y", data=pd.DataFrame(zip(f1, threshold), columns=['y','x'])
		, label='f1')
		sns.lineplot(x="x", y="y", data=pd.DataFrame(zip(true_recall,threshold), columns=['y','x'])
		, label='true recall')
		sns.lineplot(x="x", y="y", data=pd.DataFrame(zip(true_f1, threshold), columns=['y','x'])
		, label='true f1')
		try : 
			pd.DataFrame(zip(
				self.classifier.compute_average_vector_features(X_test).columns,
				self.classifier.classifier.feature_importances_)
			).set_index(0).sort_values(1).plot.pie(y=1, legend=False)
		except :
			pass
		plt.show()


def nothing(*args, **kwargs):
	pass

def something():
	print('a')





if __name__ == '__main__' :		
	manager = Manager()
	parser = argparse.ArgumentParser()
	parser.add_argument('-reset', dest='reset', nargs='+',
		choices=['features', 'classifier', 'full'], default=[])
	# parser.add_argument('-train', dest='train',
	# 	action='store_const', const=manager.train, default=nothing)
	parser.add_argument('-predict', dest='predict',
		action='store_const', const=manager.predict, default=nothing)
	parser.add_argument('-eval', dest='evaluate',
		action='store_const', const=manager.evaluate, default=nothing)
	parser.add_argument('-corpus_dir', default=DEV_DIR)
	parser.add_argument('-train_dir', default=TRAIN_DIR)
	parser.add_argument('-test_dir', default=TEST_DIR)
	parser.add_argument('-save_dir', default=SAVE_DIR)
	
	args = parser.parse_args()
	if 'full' in args.reset:
		utilities.clear()
	manager.set_dir(args.corpus_dir, args.train_dir, args.test_dir, args.save_dir)
	manager.train('features' in args.reset, 'classifier' in args.reset)
	pred = args.predict()
	if pred is not None: print(pred)
	args.evaluate()
	


