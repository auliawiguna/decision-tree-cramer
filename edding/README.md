# py-decision-tree

A simple decision tree implementation in Python as a practice
during my study on machine learning algorithms.

Gini Impurity is used in the code to decide how to split the node.
Split with the largest Gini gain is considered as the best.
This is done greedily on each node until any of the following conditions is true.

1. gini impurity = 0
2. number of records of a class in a node is smaller than `min_records`.
(to avoid overfitting)
3. number of depth is larger than `max_depth`.

Then the splitting stops and the node becomes a leaf.

Code is tested on Iris data set embeded in sklearn.
I did a random 80:20 split and the result metrics is shown as below:

```plain
                                   ┌g: 0.00 ([40, 0, 0]) cls: 0
 g: 0.67 ([40, 40, 40]) X[2] <= 1.9┤
                                   │                                 ┌g: 0.00 ([0, 0, 36]) cls: 2
                                   └g: 0.50 ([0, 40, 40]) X[3] <= 1.7┤
                                                                     │                                ┌g: 0.09 ([0, 40, 2]) cls: 1
                                                                     └g: 0.17 ([0, 40, 4]) X[2] <= 5.1┤
                                                                                                      └g: 0.00 ([0, 0, 2]) cls: 2
             precision    recall  f1-score   support

     setosa       1.00      1.00      1.00        10
 versicolor       0.90      0.90      0.90        10
  virginica       0.90      0.90      0.90        10

avg / total       0.93      0.93      0.93        30
```