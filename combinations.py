from sklearn.base import BaseEstimator, TransformerMixin


class Combinator(BaseEstimator, TransformerMixin):
    """ Static A posteriori Combinator of predictions.

        Args:
            - scheme: String flag. Can be one of the following:
                - 'majority': Simple Hard Majority Voting
                - 'weights': Weighted Voting, with weights
                             passed by user in the weights
                             arg
                - 'accuracy': Weights are calculated according
                              to prediction accuracy over the
                              meta train set
                - 'optimal': The optimal weights are found, this
                             is done by optimizing over the classification
                             error
            - weights: list or numpy.array(not sure?) containing as many
                         weights as the models in the ensemble
        Returns:
            - The  ensemble Model. Needs to be fitted for the encoding part
    """

    def __init__(self, scheme='majority', weights=None):

        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        import numpy

        self.scheme = scheme
        self.weights = weights
        self.transformation = self.voting
        self.num_labels = 0
        self.num_models = 0
        self.lab = LabelEncoder()
        self.ohe = OneHotEncoder()

        if self.scheme == 'majority':
            print "Hurray! Equality for all!"
            self.weights = None
        else:
            print "Not so much Vox Populi, Vox Dei, huh?"
            if self.scheme == 'weights':
                if type(self.weights) in (numpy.array, numpy.ndarray):
                    pass  # It is from the optimization part
                else:
                    if not(self.weights):
                        print "Need weights for this scheme!"
                self.weights = weights
                weights_string = " %.2f |" * len(self.weights) % tuple(self.weights)
                print "Using given weights: | %s" % weights_string
            else:
                # print "Will find the weights after fitting"
                pass

    def fit(self, X, y, weights=None):

        if y is None:
            raise ValueError('We need y labels to supervise-fit!')
        X_tr, y_tr = self.fit_encoders(X, y)
        if not(self.scheme in ['majority', 'weights']):
            self.find_weights(X, y, X_tr, y_tr)
            weights_string = " %.2f |" * len(self.weights) % tuple(self.weights)
            print "Using found weights: | %s" % weights_string
        return self

    def transform(self, X):

        import numpy

        if type(X[0]) is numpy.array:
            N_samples = X[0].shape[0]
        else:
            N_samples = len(X[0])
        X = self.lab.transform(X)
        X = self.ohe.transform(X.reshape(-1, 1)).todense().reshape(N_samples, -1)
        prediction = self.transformation(X)
        prediction = self.lab.inverse_transform(prediction.argmax(axis=1))
        return prediction

    def predict(self, X):
        return self.transform(X)

    def score(self, X, y_true):

        from sklearn.metrics import accuracy_score

        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred, normalize=True)

    def fit_encoders(self, X, y):

        import numpy

        self.num_labels = len(set(y))
        N_samples = len(y)
        # print "N_smaples"
        # print N_samples
        if type(X) is numpy.array:
            y = y.reshape(-1, 1)
        else:
            y = numpy.array(y).reshape(-1, 1)
        # print y
        y = self.lab.fit_transform(y).reshape(-1, 1)
        # print 'label'
        # print y
        y = self.ohe.fit_transform(y).todense()
        # print 'ohe'
        # print y
        X = self.lab.transform(X)
        # reshape(N_samples, N_samples*self.num_labels)
        X = self.ohe.transform(X.T.reshape(-1, 1)).todense().reshape(N_samples, -1)
        # print 'ohe'
        # print X.shape
        self.num_models = int(X.shape[1] / self.num_labels)
        return X, y

    def voting(self, X):

        import numpy

        predictions = numpy.zeros([X.shape[0], self.num_labels])
        if type(self.weights) in (numpy.array, numpy.ndarray):
            pass
        else:
            if not(self.weights):
                self.weights = [1 for i in xrange(self.num_models)]
        for i in xrange(X.shape[0]):
            # print X.shape
            # print X
            subarrays = numpy.split(X[i, :], self.num_models, axis=1)
            # print "subarrays"
            # print subarrays
            votes = numpy.zeros([1, self.num_labels])
            for model_index, subar in enumerate(subarrays):
                # print subar
                votes = numpy.vstack((votes, subar*self.weights[model_index]))
            # print votes
            pred_ = votes.sum(axis=0).argmax()
            pred_ = self.ohe.transform(pred_).todense()
            predictions[i, :] = pred_
        return predictions

    def find_weights(self, X, y, X_tr, y_tr):

        import numpy

        weights = [0 for i in xrange(self.num_models)]
        if self.scheme == 'accuracy':
            for i in xrange(X_tr.shape[0]):
                subarrays = numpy.split(X_tr[i, :], self.num_models, axis=1)
                for model_index, subar in enumerate(subarrays):
                    if (subar == y_tr[i, :]).all():
                        weights[model_index] += 1
            self.weights = weights
        if self.scheme == 'optimal':
            from scipy.optimize import minimize

            w = [1 for i in xrange(self.num_models)]
            bnds = tuple([(0, None) for i in xrange(self.num_models)])
            a = minimize(f, w, args=(Combinator, X, y), method='SLSQP', bounds=bnds)
            self.weights = list(a.x)
        return


#  Example Usage
# def f(w, Combinator, x, y):
#     gg = Combinator(scheme='weights', weights=w)
#     gg.fit(x, y)
#     score = 1- gg.score(x, y)
#     #print 'Weights'
#     #print w
#     #print 'Score: ' + str(score)
#     return score    

# w = [1 for i in xrange(6)]
# bnds = tuple([(0, None) for i in xrange(6)])
# a = minimize(f, w,  args=(Combinator, predictions_meta, y_meta), method='SLSQP', bounds=bnds)
