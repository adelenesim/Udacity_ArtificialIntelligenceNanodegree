import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        # Implement model selection based on BIC scores
        
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # build first model for initialization
        bestmodel = None; bestbic = float('inf')
        for i in range(self.min_n_components,self.max_n_components+1):
            try:
                currmodel = self.base_model(i)
                # from forum discussion https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/17
                p = i*i+2*i*len(self.X[0])-1
                currbic = -2*currmodel.score(self.X, self.lengths)+p*np.log(len(self.X))
                if currbic < bestbic: bestbic = currbic; bestmodel = currmodel
            except: continue
        return bestmodel


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def caldic(self,currmodel):
        leftsum = currmodel.score(self.X,self.lengths)
        M = len(self.words.keys())-1
        rightsum = 0
        for w in self.words.keys():
            if w != self.this_word:
                newx, newlengths = self.hwords[w]
                rightsum += currmodel.score(newx, newlengths)
        return leftsum-rightsum/(M-1)

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # Implement model selection based on DIC scores
        # build first model for initialization
        bestmodel = None; bestdic = -float('inf')
        for i in range(self.min_n_components,self.max_n_components+1):
            try:
                currmodel = self.base_model(i)
                currdic = self.caldic(currmodel)
                if currdic > bestdic: bestdic = currdic; bestmodel = currmodel
            except: continue
        return bestmodel


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection using CV
        # generate the CV, as demo-ed in the notebook
        # we keep these the same for consistency across the models
        alltrain = []; alltest = []
        # see discussion here: https://discussions.udacity.com/t/fish-word-with-selectorcv-problem/233475/5
        try:
            # with default 3 splits
            for cv_train_idx, cv_test_idx in KFold().split(self.sequences):
                trainX,trainlen = combine_sequences(cv_train_idx, self.sequences)
                testX,testlen = combine_sequences(cv_test_idx, self.sequences)
                alltrain.append((trainX,trainlen)); alltest.append((testX,testlen))
        except:
            # not enough for a meaningful split, so we do not score on the cross validation set
            # just score on the training data
            trainX,trainlen = combine_sequences(np.arange(len(self.sequences)), self.sequences)
            testX,testlen = combine_sequences(np.arange(len(self.sequences)), self.sequences)
            alltrain.append((trainX,trainlen)); alltest.append((testX,testlen))

        bestL = -float('inf'); bestmodel = None
        for i in range(self.min_n_components,self.max_n_components+1):
            currscore = 0
            try:
                for j, (self.X, self.lengths) in enumerate(alltrain):
                    currmodel = self.base_model(i)
                    currX, currlen = alltest[j]
                    currscore += currmodel.score(currX, currlen)
                # take average
                currscore /= len(alltrain)
                if currscore>bestL: bestL = currscore; bestmodel = currmodel
            except: continue
        return bestmodel
