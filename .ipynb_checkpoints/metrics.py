from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from subprocess import call
import numpy as np
import pandas as pd
#-----------------------------------------------------------------------------------------


def mse(y_predicted, y_true):  
    mse = ((y_predicted - y_true) ** 2).mean()
    return mse
#-----------------------------------------------------------------------------------------

def rmse(y_predicted, y_true):  
    rmse = np.sqrt(mse(y_predicted, y_true))
    return rmse
#-----------------------------------------------------------------------------------------
def rsq(y_predicted, y_true):  
    u = ((y_predicted - y_true) ** 2).mean()
    v = ((y_true.mean() - y_true) ** 2).mean()
    return 1 - u / (v + 1e-9)

#-----------------------------------------------------------------------------------------
def print_scores(rf, _X_train, _X_test, _y_train, _y_test):

    print([
        rmse(rf.predict(_X_train), _y_train),
        rmse(rf.predict(_X_test), _y_test),
        rf.score(_X_train, _y_train),
        rf.score(_X_test, _y_test),
        rf.oob_score_
        if hasattr(rf, "oob_score_")
        else ''
    ])
    return rf
#-----------------------------------------------------------------------------------------

def visualize_tree(dt, figsize=(20, 20), feature_names=None):
    export_graphviz(dt,
                    out_file="tree.dot",
                    feature_names=feature_names,
                    rounded=True,
                    filled=True)
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

    plt.figure(figsize=figsize)
    plt.imshow(plt.imread('tree.png'))


def accuracy_score(y_pred, y_actual):
    pass
