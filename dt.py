__author__ = 'rusyida'

from sklearn import tree
from sklearn.datasets import load_iris
import nltk
import sklearn
import graphviz
import pandas as pd
import numpy as np

print('The nltk version is {}.'.format(nltk.__version__))
print('The scikit-learn version is {}.'.format(sklearn.__version__))

df = pd.read_excel("/Users/ahmadauliawiguna/Documents/WORK/Riset/pemilu_as_int.xls", header=0)
dataset = df

# get class label
class_label = list(set(dataset.target.astype(str).values.flat))
header = list(dataset.columns.astype(str).values)
header = header[:16]
datasetValues = dataset.values

data = datasetValues[:datasetValues.shape[0] , :datasetValues.shape[1]-1]
target = datasetValues[:datasetValues.shape[0] , datasetValues.shape[1]-1]


clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=25, max_depth=20, min_samples_leaf=2)
clf = clf.fit(data, target)
dot_data = tree.export_graphviz(clf, out_file=None) 
dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=header,  
                     class_names=np.asarray(['democrat', 'republic']),  
                    #  feature_names=iris.feature_names,  
                    #  class_names=iris.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render("president")