from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from c45 import C45
import pandas as pd 
import numpy as np

iris = load_breast_cancer()

dataset = np.genfromtxt('car.csv', dtype=None, delimiter=',') 
header = dataset[0, 0:-1]
allDataset = dataset[1:, : ]
X = dataset[1:,0:dataset.shape[1]-1] #ambil kolom dari kolom ke 0 sampai ke kolom 2 dari kanan
y = dataset[1:,dataset.shape[1] - 1] #ambil kolom terakhir

kf = KFold(n_splits = 10, random_state=10, shuffle=True)
incrementTest=1
akurasiTotal = float(0)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = C45(attrNames=header, method='gain')
    clf.fit(X_train, y_train)
    clf.printTree()
    akurasi = float(clf.score(X_test, y_test))
    print 'Akurasi = ', akurasi
    akurasiTotal = float(akurasiTotal) + float(akurasi)
    # print clf.predict(X_test)

print 'Akurasi Rata-Rata : ', float(akurasiTotal)/float(10) * 100
