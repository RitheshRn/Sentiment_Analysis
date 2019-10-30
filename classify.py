#!/bin/python

def train_classifier(X, y, regularization_constant=10):
	"""Train a classifier using the given training data.

	Trains logistic regression on the input data with default parameters.
	"""
	from sklearn.linear_model import LogisticRegression
	cls = LogisticRegression(random_state=0, solver='lbfgs', max_iter=10000, C=regularization_constant)
	cls.fit(X, y)
	return cls

def evaluate(X, yt, cls, name='data'):
    """Evaluated a classifier on the given labeled data using accuracy."""
    from sklearn import metrics
    yp = cls.predict(X)
    acc = metrics.accuracy_score(yt, yp)
    #print("  Accuracy on %s  is: %s" % (name, acc))
    return acc

def evaluate_predict(X, cls):
    """Evaluated a classifier on the given labeled data using accuracy."""
    from sklearn import metrics
    yp = cls.predict(X)
    return yp

def evaluate_predict_prob(X, cls):
    """Evaluated a classifier on the given labeled data using accuracy."""
    from sklearn import metrics
    yp = cls.predict_proba(X)
    return yp
