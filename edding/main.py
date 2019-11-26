from decision_tree_classifier import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

if __name__ == '__main__':
    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target,
        test_size=0.2,
        random_state=1234,
        stratify=iris.target
    )

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    clf.describe_tree()

    y_pred = clf.predict(X_test)

    print(classification_report(y_true=y_test, y_pred=y_pred, target_names=iris.target_names))