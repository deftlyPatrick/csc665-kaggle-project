import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from . import features as fet
import subprocess
from sklearn.tree import export_graphviz
from sklearn.metrics import r2_score

def mse(y_predicted, y_true):
		# return Mean-Squared Error
	# ((y_hat - y_test) ** 2).mean() -> using numpy
	return np.square(np.subtract(y_predicted, y_true)).mean()

def rmse(y_predicted, y_true):
	# return Root Mean-Squared Error
	return np.sqrt(mse(y_predicted, y_true))


# def rsq(y_p, y_actual):
# 	# return R^2
# 	u = ((y_predicted-y_true) ** 2).mean()
# 	v = ((y_true.mean() - y_true) ** 2).mean()
# 	return (1-u)/(v+1e-9)
#
# 	# v = np.square(np.subtract(y_true.mean(), y_predicted).mean())
# 	# rsq = 1 - (mse/v)
# 	# return rsq

def accuracy_score(y_pred, y_actual):
	y_pred = np.array([y_pred])
	y_actual = np.array([y_actual])
	accuracy = np.mean(y_pred == y_actual)
	return accuracy


def rsq(y_pred, y_actual):
	y_pred = np.asarray(y_pred)
	y_actual = np.asarray(y_actual)
	u = ((y_pred - y_actual) ** 2).mean()
	v = ((y_actual.mean() - y_actual) ** 2).mean()
	return 1 - u / (v + 1e-9)

def print_scores(rf, _X_train, _X_valid, _y_train, _y_valid):
	print([rmse(rf.predict(_X_train), _y_train), rmse(rf.predict(_X_valid), _y_valid), rf.score(_X_train, _y_train), rf.score(_X_valid, _y_valid), rf.oob_score_ if hasattr(rf, "oob_score_") else ''])
	return rf

def visualize_tree(dt, figsize=None, feature_names=None):
	export_graphviz(dt, out_file="tree.dot", feature_names=feature_names, rounded=True, filled=True)

	subprocess.call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])
	plt.figure(figsize=figsize)
	plt.imshow(plt.imread('tree.png'))


# y_true = [3, -0.5, 2, 7]
# y_pred = [2.5, 0.0, 2,8]
# a = r2_score(y_true, y_pred)
# b = rsq(y_true, y_pred)
#
# print(a)
# print(b)
