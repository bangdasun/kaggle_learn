
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_X_y, check_is_fitted
from scipy import sparse


class NBLRClassifier(BaseEstimator, ClassifierMixin):
    """
    LogisticRegression classifier with naive bayes features

    see https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf
    refer from https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline/notebook
    see also https://bangdasun.github.io/2018/11/22/29-nbsvm-a-strong-classification-baseline/

    Examples
    --------
    >>> clf = NBLRClassifier(C=4, dual=True, n_jobs=-1)
    >>> clf.fit(X_train, y_train)
    >>> y_test_pred = clf.predict(X_test)
    """

    def __init__(self, C=1.0, dual=False, n_jobs=1):
        """

        Parameters
        ----------
        C     : float, Inverse of regularization strength; must be a positive float.
                       Like in support vector machines, smaller values specify stronger regularization.
        dual  : boolean, Dual or primal formulation. Dual formulation is only implemented for
                       l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.
        n_jobs: int or None, Number of CPU cores used when parallelizing over classes if multi_class=’ovr’”.
                       This parameter is ignored when the solver is set to ‘liblinear’ regardless of whether
                       ‘multi_class’ is specified or not. None means 1 unless in a joblib.parallel_backend context.
                       -1 means using all processors.
        """
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def fit(self, X, y):
        # check that X and y have correct shape
        X, y = check_X_y(X, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y == y_i].sum(0)
            return (p + 1) / ((y == y_i).sum() + 1)

        # calculate log-count ratio
        self._log_count_ratio = sparse.csr_matrix(np.log(pr(X, 1, y) / pr(X, 0, y)))
        # naive bayes features
        X_nb = X.multiply(self._log_count_ratio)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(X_nb, y)
        return self

    def predict(self, X):
        # verify that model has been fit
        check_is_fitted(self, ["_log_count_ratio", "_clf"])
        return self._clf.predict(X.multiply(self._log_count_ratio))

    def predict_proba(self, X):
        # verify that model has been fit
        check_is_fitted(self, ["_log_count_ratio", "_clf"])
        return self._clf.predict_proba(X.multiply(self._log_count_ratio))
