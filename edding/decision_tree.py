from utils import *
import pptree as pp


class DecisionTree:
    def __init__(self, root=None, max_depth=999, min_records=1):
        self.root = root
        self.max_depth = max_depth
        self.min_records = min_records

    def build(self, X, y):
        self.root = Node(X, y, n_class=max(np.unique(y)) + 1)
        self._split_node(self.root)

    def _split_node(self, node):
        if node is None:
            return
        if node.depth > self.max_depth:
            return
        if node.min_records() < self.min_records:
            return

        node.split()
        self._split_node(node.left)
        self._split_node(node.right)

    def describe(self):
            pp.print_tree(self.root)

    def find(self, x):
        return self._find(x, self.root)

    def _find(self, x, node):
        if node.condition is None:
            return node.cls
        if node.condition.evaluate(x):
            return self._find(x, node.left)
        else:
            return self._find(x, node.right)


class Node:
    def __init__(self, X, y, n_class=0, gini=-1.0, depth=0, parent=None):
        self.gini = gini

        self.X = X
        self.y = y
        self.n_class = n_class
        self.depth = depth

        self.value = cnt_value(y, self.n_class)
        self.cls = -1

        self.condition = None

        self.left = None
        self.right = None

        # For pptree printing
        self.children = []
        if parent:
            parent.children.append(self)

    def __str__(self):
        if self.condition is None:
            return "g: %.2f (%s) cls: %d" % (self.gini, self.value, self.cls)
        else:
            return "g: %.2f (%s) %s" % (self.gini, self.value, self.condition)

    # TODO: Think of a good way to avoid repeating calculations here
    # Right not it is always true
    def min_records(self):
        return min_records_cnt(self.y)

    def split(self):
        # Calculate & set gini if root
        if self.gini < 0.0:
            self.gini = calculate_gini(self.y)

        # Stop split if all records lie in the same class (gini=0.0)
        if self.gini == 0.0:
            self.cls = [i for i in range(len(self.value)) if self.value[i] > 0][0]
            return

        # Find best split and split node
        best_split = find_best_split(self.X, self.y, self.gini)
        if best_split is None:
            self.cls = [i for i in range(len(self.value)) if self.value[i] == max(self.value)][0]
            return

        idx, val, g_l, g_r = best_split
        X_l, y_l, X_r, y_r = split_data(self.X, self.y, idx, val)

        self.condition = Condition(idx, val)

        # Set left & right child
        self.left = Node(X_l, y_l, n_class=self.n_class, gini=g_l, depth=self.depth + 1, parent=self)
        self.right = Node(X_r, y_r, n_class=self.n_class, gini=g_r, depth=self.depth + 1, parent=self)


class Condition:
    def __init__(self, idx, val):
        self.idx = idx
        self.val = val

    def evaluate(self, x):
        if x[self.idx] <= self.val:
            return True
        return False

    def __str__(self):
        return "X[" + str(self.idx) + "] <= " + str(self.val)