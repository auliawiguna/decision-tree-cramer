from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from c45 import C45
import pandas as pd 
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

iris = load_breast_cancer()

dataset = np.genfromtxt('abalone.csv', dtype=None, delimiter=',') 
header = dataset[0, 0:-1]
allDataset = dataset[1:, : ]
X = dataset[1:,0:dataset.shape[1]-1] #ambil kolom dari kolom ke 0 sampai ke kolom 2 dari kanan
y = dataset[1:,dataset.shape[1] - 1] #ambil kolom terakhir

kf = KFold(n_splits = 10, random_state=40, shuffle=True)
incrementTest=1
akurasiTotal = float(0)
precisionTotal = float(0)
recallTotal = float(0)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = C45(attrNames=header, method='cramer')
    clf.fit(X_train, y_train)
    clf.printTree()

    y_pred = clf.predict(X_test)
    akurasi = accuracy_score(y_test, y_pred)
    presisi = precision_score(y_test,y_pred, average='macro')
    recall = recall_score(y_test,y_pred, average='macro')

    print 'Akurasi = ', akurasi
    print 'Presisi = ', presisi
    print 'Recall = ', recall

    akurasiTotal = float(akurasiTotal) + float(akurasi)
    recallTotal = float(recallTotal) + float(recall)
    precisionTotal = float(precisionTotal) + float(presisi)

print 'Akurasi Rata-Rata : ', float(akurasiTotal)/float(10) * 100
print 'Precision Rata-Rata : ', float(precisionTotal)/float(10) * 100
print 'Recall Rata-Rata : ', float(recallTotal)/float(10) * 100
