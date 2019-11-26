from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from c45 import C45

iris = load_breast_cancer()
clf = C45(attrNames=iris.feature_names)
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.5)
print clf.fit(X_train, y_train)
print 'Akurasi ', clf.score(X_test, y_test)
# print(f'Accuracy: {clf.score(X_test, y_test)}')
clf.printTree()