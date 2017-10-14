import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    allwords = test_set.wordlist
    # Implement the recognizer
    for i, word in enumerate(allwords):
      currprob = {}; bestguess = ''; bestscore = -float('inf')
      currX, currlen = test_set.get_item_Xlengths(i)
      for m in models:
        try:
          currscore = models[m].score(currX, currlen)
          currprob[m] = currscore
          if currscore > bestscore: bestguess = m; bestscore = currscore
        except: continue
      probabilities.append(currprob); guesses.append(bestguess)
    return probabilities, guesses
