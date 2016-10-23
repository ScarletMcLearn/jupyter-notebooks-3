import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve

def load_smes_main_bank():
    return SME().load_main_bank()

def load_smes_num_bank():
    return SME().load_num_bank()

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap,vmin=0.0,vmax=1.0)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

class SME:

    _file =  "~/Dropbox/MowersWanzenried_Paper/kmuifzdata_workingdata_reducedsample.dta"

    _columms = [
          'anz_bankbezieh'
        , 'dum_hauptbankkb'
        , 'dum_hauptbankregiob'
        , 'dum_hauptbankraiffeisen'
        , 'dum_hauptbankgrossbank'
        , 'dum_andere'
        , 'ln_size'
        , 'av_ta_inklstillres'
        , 'gewinnmarge'
        , 'familyfirm'
        , 'dum_zufrieden_bankbez'
        , 'dum_keine_zukinv'
        , 'dumind_bau'
        , 'dumind_handel'
        , 'dumind_verkehrlagerei'
        , 'dumind_gastgewerbe'
        , 'dumind_infokomm'
        , 'dumind_fbwisstechndl'
        , 'dumind_sonstwirtdl'
        , 'dumind_erzunterricht'
        , 'dumind_gesundsoz'
        , 'dumind_kunstunterherhol'
        , 'dumind_andere'
        , 'dumind_na'
    ]
    frame = None
    data = None
    target = None
    targetCats = None

    def load_num_bank(self):
        self.frame = pd.read_stata(self._file)
        lhv = [
            'anz_bankbezieh'
        ]
        exclude = [
            'dum_hauptbankkb'
            , 'dum_hauptbankregiob'
            , 'dum_hauptbankraiffeisen'
            , 'dum_hauptbankgrossbank'
            , 'dum_andere'
        ]  + lhv
        rhv = [x for x in self._columms if (x not in exclude)]
        df = self.frame[lhv + rhv].dropna()
        self.data = df[rhv]
        self.target = df[lhv[0]]
        return self

    def load_main_bank(self):
        self.frame = pd.read_stata(self._file)
        lhv = [
            'dum_hauptbankkb'
            , 'dum_hauptbankregiob'
            , 'dum_hauptbankraiffeisen'
            , 'dum_hauptbankgrossbank'
            , 'dum_andere'
        ]
        rhv = [x for x in self._columms if x not in lhv]
        df = self.frame[lhv + rhv]
        df = df.dropna()

        self.data =  df[rhv]
        self.target =  (df[lhv] * np.array([1,2,3,4,5])).sum(axis=1)
        self.target.name = 'main_bank'
        return self





