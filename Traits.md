# Liste des traits utilisé par le classifieur
* Mesure probabilistes

	Ces traits on pour objectif de détectées les anomalies statistiques, cas particuliers ou tendances.
	Par exemple, l'information mutuel entre un nom et un verbe pouvant former une CVS pourrait être 
	différente le l'information mutuel attendu entre un nom et un verbe ne pouvant former de CVS.
	* P

		probabilité jointe du nom et du verbe

		La probabilité qu'un candidat selectionné au hasard ai pour nom et verbe 
		le nom et le verbe du candidat en question.
	* P_n et P_v

		probabilité du nom/verbe

		La probabilité qu'un candidat selectionné au hasard ai pour nom/verbe
		le nom/verbe du candidat en question.
	* P_n_given_v et P_v_given_n

		probabilité conditionnel du nom et du verbe

		La probabilité qu'un candidat selectionné au hasard ai pour nom/verbe 
		le nom/verbe du candidat en question, sachant qu'il a pour le verbe/nom
		du candidat en question pour verbe/nom
	* V_n et V_v

		variance du nom/verbe (calculer grâce à P_n et P_v de manière non-linéaire)
	* V

		covariance du nom et du verbe
	* corr

		corrélation entre le nom et le verbe
	* pmi

		information mutuel entre le nom et le verbe
* Mesures syntaxique

	Ces mesures ont pour objectif de détecter d'éventuels paritcularité syntaxique des CVS.
	Par exemple les nom et verbe pouvant former des CVS sont peut-être souvent séparer par un
	déterminant.
	* dist

		Distance dans la phrase moyenne entre le nom et le verbe du candidat en question
	* P_obj, P_obl, P_subj et P_other

		Probabilité que le nom du candidat en question soit 
		objet/oblique/sujet/autre du verbe du candidat en question
	* patrons syntaxiques fréquents

		Probabilité que le nom et le verbe du candidat en question soit séparer dans la phrase par
		une sequence de POS fréquente. (les 20 séquences les plus fréquente sont utilisé)
		Par exemple "(Det,)" représente la probabilité d'être séparer par un déterminant.
* Mesures sémantique

	Ces mesures ont pour objectif de détecter d'éventuels particularité chez les réprésentation 
	sémantique des nom et verbe pouvant former des CVS.
	Par exemple, chez les CVS le verbe n'apporte que peu de sens à l'expression, peut-être cette
	particularité peut se retrouvé dans un espace sémantique.

	* noun / verb

		Similarité entre le vecteur du nom/verbe du candidat en question et le vecteur moyen 
		des noms/verbes annotés comme CVS dans le corpus 'Train'
	* exp

		Similarité entre l'estimateur du vecteur du candidat en question et le vecteur moyen 
		des estimateurs de vecteur de candidats annotés comme CVS dans le corpus 'Train'
	* len_n / len_v

		norm du vecteur nom/verbe
	* dist_n / dist_v

		distance entre le vecteur du nom/verbe du candidat en question et l'estimateur du
		vecteur du candidat en question
	* dist_relative

		ratio de dist_n et dist_v



