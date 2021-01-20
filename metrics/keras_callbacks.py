
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from keras.callbacks import Callback


class RocAucEvaluation(Callback):
    """ Calculate ROC AUC """
    def __init__(self, validation_data, interval=1):
        """

        Parameters
        ----------
        validation_data : tuple, validation data (X, y)
        interval        : int, number of epochs between checkpoints
        """
        super().__init__()
        self.X_valid, self.y_valid = validation_data
        self.interval = interval
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_valid, verbose=0)
            score = roc_auc_score(self.y_valid, y_pred)
            print("ROC AUC - epoch: {:d} - score: {:.5f}".format(epoch + 1, score))

            
class PRAucEvaluation(Callback):
    """ Calculate Precision - Recall AUC """
    def __init__(self, validation_data, interval=1):
        """

        Parameters
        ----------
        validation_data : tuple, validation data (X, y)
        interval        : int, number of epochs between checkpoints
        """
        super().__init__()
        self.X_valid, self.y_valid = validation_data
        self.interval = interval
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_valid, verbose=0)
            score = average_precision_score(self.y_valid, y_pred)
            print("PR AUC - epoch: {:d} - score: {:.5f}".format(epoch + 1, score))

            
class F1Evaluation(Callback):
    """ Calculate F1 score """
    def __init__(self, validation_data, interval=1):
        """

        Parameters
        ----------
        validation_data : tuple, validation data (X, y)
        interval        : int, number of epochs between checkpoints
        """
        super().__init__()
        self.X_valid, self.y_valid = validation_data
        self.interval = interval
    
    def on_epoch_end(self, epoch, threshold=0.5, logs=None):
        if epoch % self.interval == 0:
            y_pred_proba = self.model.predict(self.X_valid, verbose=0)
            y_pred = (y_pred_proba > threshold).astype(np.int)
            score = f1_score(self.y_valid, y_pred)
            print("F1 - epoch: {:d} - score: {:.5f} threshold = {:.3f}".format(epoch + 1, score, threshold))
