import pdb
from c45 import C45

#1st parameter = main dataset
#2nd parameter = dataset labels
#3rd parameter = method, information_gain or gain_ratio or cramer
c1 = C45(
    "/Users/ahmadauliawiguna/PycharmProjects/Pelatihan/cramer/data/iris/iris.data", 
    "/Users/ahmadauliawiguna/PycharmProjects/Pelatihan/cramer/data/iris/iris.names", 
    "cramer")

c1.fetchData()
c1.preprocessData()

#set 10 fold cross validation
#make model based on 10 fold cross validation
c1.makeModelXValidation(10)

# c1.preprocessData()
# c1.generateTree()
# c1.printTree()