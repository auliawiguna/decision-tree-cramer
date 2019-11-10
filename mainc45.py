import pdb
from c45 import C45

#1st parameter = main dataset
#2nd parameter = dataset labels
#3rd parameter = method, information_gain or gain_ratio or cramer
c1 = C45(
    "/Users/ahmadauliawiguna/PycharmProjects/Pelatihan/cramer/data/breast-cancer-wisconsin/breast-cancer-wisconsin.data", 
    "/Users/ahmadauliawiguna/PycharmProjects/Pelatihan/cramer/data/breast-cancer-wisconsin/breast-cancer-wisconsin.names", 
    # "/Users/ahmadauliawiguna/PycharmProjects/Pelatihan/cramer/data/iris/iris.data", 
    # "/Users/ahmadauliawiguna/PycharmProjects/Pelatihan/cramer/data/iris/iris.names", 
    # "/Users/ahmadauliawiguna/PycharmProjects/Pelatihan/cramer/data/iris/iris2.data", 
    # "/Users/ahmadauliawiguna/PycharmProjects/Pelatihan/cramer/data/iris/iris2.names", 
    "information_gain")


c1.fetchData()
c1.preprocessData()
c1.setThreshold(2)

#set 10 fold cross validation
#make model based on 10 fold cross validation
c1.makeModelXValidation(10)

# c1.generateTree()
# c1.printTree()