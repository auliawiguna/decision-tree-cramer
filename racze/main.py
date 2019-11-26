from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from c45 import C45
import pandas as pd 
import numpy as np

iris = load_breast_cancer()

dataset = np.genfromtxt('breast-cancer.csv', dtype=None, delimiter=',') 
header = dataset[0, 0:-1]
X = dataset[1:,0:dataset.shape[1]-1] #ambil kolom dari kolom ke 0 sampai ke kolom 2 dari kanan
y = dataset[1:,dataset.shape[1] - 1] #ambil kolom terakhir

# clf = C45(attrNames=iris.feature_names)
clf = C45(attrNames=header)
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
clf.fit(X_train, y_train)
clf.printTree()
print 'Akurasi ', clf.score(X_test, y_test)
print clf.predict(X_test)
