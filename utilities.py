from os import path
import pickle
from time import time

SAVE_DIR = 'Save'

# TODOC
def timed(func):
	def intern(*args, **kwargs):
		start = time()
		res = func(*args, **kwargs)
		print(time() - start)
		return res
	return intern

def saved(to_save, file_name):
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
    file_path = path.join(SAVE_DIR, file_name)
    with open(file_path, 'wb') as output:
        pickle.dump(to_save, output)
    return to_save

def drive_cached(func, file_name):
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
        return saved(res, file_name)
    return intern
