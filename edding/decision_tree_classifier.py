from decision_tree import *


class DecisionTreeClassifier:

    def __init__(self, max_depth=999, min_records=1):
        self.max_depth = max_depth
        self.min_records = min_records

        self.tree = DecisionTree()

    def fit(self, X, y):
        self.tree.build(X, y)

    def predict(self, X):
        return [self.tree.find(x) for x in X]

    # Describe the classifier attributes
    def describe(self):
        print("Decision Tree Classifier:\nmax_depth = %d, min_records = %d" % (self.max_depth, self.min_records))

    def describe_tree(self):
        self.tree.describe()