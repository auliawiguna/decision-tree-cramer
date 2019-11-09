import pandas as pd
import math
import operator
import numpy as np

class DecisionTreeClassifier(object):
    def __init__(self, max_depth, features):
        self.depth = 0
        self.max_depth = max_depth
        self.features = features

    def entropy_func(self, c, n):
        """
        The math formula
        """
        return -(c*1.0/n)*math.log(c*1.0/n, 2)

    def entropy_cal(self, c1, c2):
        """
        Returns entropy of a group of data
        c1: count of one class
        c2: count of another class
        """
        if c1== 0 or c2 == 0:  # when there is only one class in the group, entropy is 0
            return 0
        return self.entropy_func(c1, c1+c2) + self.entropy_func(c2, c1+c2)

    # get the entropy of one big circle showing above
    def entropy_of_one_division(self, division): 
        """
        Returns entropy of a divided group of data
        Data may have multiple classes
        """
        s = 0
        n = len(division)
        classes = set(division)
        for c in classes:   # for each class, get entropy
            n_c = sum(division==c)
            e = n_c*1.0/n * self.entropy_cal(sum(division==c), sum(division!=c)) # weighted avg
            s += e
        return s, n

    # The whole entropy of two big circles combined
    def get_entropy(self, y_predict, y_real):
        """
        Returns entropy of a split
        y_predict is the split decision, True/False, and y_true can be multi class
        """
        if len(y_predict) != len(y_real):
            print('They have to be the same length')
            return None
        n = len(y_real)
        s_true, n_true = self.entropy_of_one_division(y_real[y_predict]) # left hand side entropy
        s_false, n_false = self.entropy_of_one_division(y_real[~y_predict]) # right hand side entropy
        s = n_true*1.0/n * s_true + n_false*1.0/n * s_false # overall entropy, again weighted average
        # print(y_real)
        # print(y_predict)
        # print(~y_predict)
        # print(s_true)
        # print(n_true)
        # print(s_false)
        # print(n_false)
        # print('------------------------------------------------')
        return s

    def get_split_info(self, y_predict, y_real):
        """
        Returns entropy of a split
        y_predict is the split decision, True/False, and y_true can be multi class
        """
        if len(y_predict) != len(y_real):
            print('[2] They have to be the same length')
            return None
        n = len(y_real)
        s_true, n_true = self.entropy_of_one_division(y_real[y_predict]) # left hand side entropy
        s_false, n_false = self.entropy_of_one_division(y_real[~y_predict]) # right hand side entropy
        s = n_true + n_false
        if n_true>0 and n_false>0:
            s = ((float(-1)) * float(n_true) / (float(n_true)+float(n_false)) * math.log(float(n_true) / (float(n_true)+float(n_false)), 2)) - (float(n_false) / (float(n_true)+float(n_false)) * math.log(float(n_false) / (float(n_true)+float(n_false)), 2))
            return s
        else:
            return 0

    def fit(self, x, y, par_node={}, depth=0):
        if par_node is None: 
            return None
        elif len(y) == 0:
            return None
        elif self.all_same(y):
            return {'val':y[0]}
        elif depth >= self.max_depth:
            return None
        else: 
            col, cutoff, entropy = self.find_best_split_of_all(x, y)    # find one split given an information gain 
            split_info = self.find_best_split_of_all_split_info(x, y)    # find one split info 
            y_left = y[x[:, col] < cutoff]
            y_right = y[x[:, col] >= cutoff]
            if split_info==0:
                gain_ratio = 0
            else:
                gain_ratio = float(entropy)/float(split_info)
            par_node = {'col': header[col], 
                        'information gain' : entropy,
                        'split info' : split_info,
                        'gain ratio' : gain_ratio,
                        'index_col':col,
                        'cutoff':cutoff,
                       'val': np.round(np.mean(y))
                       }
            par_node['left'] = self.fit(x[x[:, col] < cutoff], y_left, {}, depth+1)
            par_node['right'] = self.fit(x[x[:, col] >= cutoff], y_right, {}, depth+1)
            self.depth += 1 
            self.trees = par_node
            return par_node
    
    def find_best_split_of_all(self, x, y):
        col = None
        min_entropy = 1
        cutoff = None
        for i, c in enumerate(x.T):
            entropy, cur_cutoff = self.find_best_split(c, y)
            if entropy == 0:    # find the first perfect cutoff. Stop Iterating
                return i, cur_cutoff, entropy
            elif entropy <= min_entropy:
                min_entropy = entropy
                col = i
                cutoff = cur_cutoff
        return col, cutoff, min_entropy

    def find_best_split_of_all_split_info(self, x, y):
        col = None
        min_entropy = 1
        cutoff = None
        for i, c in enumerate(x.T):
            entropy, cur_cutoff = self.find_best_split(c, y)
            split_info = self.find_best_split_info(c, y)
        return split_info
    
    def find_best_split(self, col, y):
        min_entropy = 10
        min_gain_ratio = 0
        n = len(y)
        for value in set(col):
            y_predict = col < value
            my_entropy = self.get_entropy(y_predict, y)
            my_split_info = self.get_split_info(y_predict, y)
            if my_split_info==0:
                gain_ratio = 0
            else:
                gain_ratio = float(my_entropy)/float(my_split_info)

            if gain_ratio >= min_gain_ratio:
                min_gain_ratio = gain_ratio
                cutoff = value
        return min_gain_ratio, cutoff

    def find_best_split_info(self, col, y):
        min_entropy = 10
        n = len(y)
        for value in set(col):
            y_predict = col < value
            my_entropy = self.get_entropy(y_predict, y)
            my_split_info = self.get_split_info(y_predict, y)
            if my_entropy <= min_entropy:
                min_entropy = my_entropy
                cutoff = value
        return my_split_info
    
    def all_same(self, items):
        return all(x == items[0] for x in items)
                                           
    def predict(self, x):
        tree = self.trees
        results = np.array([0]*len(x))
        for i, c in enumerate(x):
            results[i] = self._get_prediction(c)
        return results
    
    def _get_prediction(self, row):
        cur_layer = self.trees
        while cur_layer.get('cutoff'):
            if row[cur_layer['index_col']] < cur_layer['cutoff']:
                cur_layer = cur_layer['left']
            else:
                cur_layer = cur_layer['right']
        else:
            return cur_layer.get('val')

from sklearn.datasets import load_iris
from pprint import pprint

iris = load_iris()

df = pd.read_excel("/Users/ahmadauliawiguna/Documents/WORK/Riset/pemilu_as_int.xls", header=0)
dataset = df

# get class label
class_label = list(set(dataset.target.astype(str).values.flat))
header = list(dataset.columns.astype(str).values)
header = header[:16]
datasetValues = dataset.values

data = datasetValues[:datasetValues.shape[0] , :datasetValues.shape[1]-1]
target = datasetValues[:datasetValues.shape[0] , datasetValues.shape[1]-1]

x = data
y = target
# x = iris.data
# y = iris.target

clf = DecisionTreeClassifier(max_depth=20, features=header)
m = clf.fit(x, y)

pprint(m)
