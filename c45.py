import math
from sklearn.model_selection import KFold
from operator import itemgetter
from sklearn import metrics
class C45:

	"""Creates a decision tree with C4.5 algorithm"""
	def __init__(self, pathToData,pathToNames,criterion):
		self.filePathToData = pathToData
		self.filePathToNames = pathToNames
		self.criterion = criterion
		self.data = []
		self.classes = []
		self.numAttributes = -1 
		self.attrValues = {}
		self.attributes = []
		self.tree = None

	def fetchData(self):
		with open(self.filePathToNames, "r") as file:
			classes = file.readline()
			self.classes = [x.strip() for x in classes.split(",")]
			#add attributes
			for line in file:
				[attribute, values] = [x.strip() for x in line.split(":")]
				values = [x.strip() for x in values.split(",")]
				self.attrValues[attribute] = values
		self.numAttributes = len(self.attrValues.keys())
		self.attributes = list(self.attrValues.keys())
		with open(self.filePathToData, "r") as file:
			for line in file:
				row = [x.strip() for x in line.split(",")]
				if row != [] or row != [""]:
					self.data.append(row)

	def preprocessData(self):
		for index,row in enumerate(self.data):
			for attr_index in range(self.numAttributes):
				if(not self.isAttrDiscrete(self.attributes[attr_index])):
					self.data[index][attr_index] = float(self.data[index][attr_index])

	def printTree(self):
		self.printNode(self.tree)

	def printNode(self, node, indent=""):
		if not node.isLeaf:
			if node.threshold is None:
				#discrete
				for index,child in enumerate(node.children):
					if child.isLeaf:
						print(indent + node.label + " = " + self.attributes[index] + " : " + child.label)
					else:
						print(indent + node.label + " = " + self.attributes[index] + " : ")
						self.printNode(child, indent + "	")
			else:
				#numerical
				leftChild = node.children[0]
				rightChild = node.children[1]
				if leftChild.isLeaf:
					print(indent + node.label + " <= " + str(node.threshold) + " : " + leftChild.label)
				else:
					print(indent + node.label + " <= " + str(node.threshold)+" : ")
					self.printNode(leftChild, indent + "	")

				if rightChild.isLeaf:
					print(indent + node.label + " > " + str(node.threshold) + " : " + rightChild.label)
				else:
					print(indent + node.label + " > " + str(node.threshold) + " : ")
					self.printNode(rightChild , indent + "	")

	def predict(self, dataTest):
		arrResult = []
		for index, record in enumerate(dataTest):
			prediction = self._predict(self.tree, record, index)
			arrResult.append(prediction)
		return arrResult

	def _predict(self, node, dataTest, indexTest=0):
		for indexAttribute, attribute in enumerate(self.attributes):
			if not node.isLeaf:
				if node.threshold is None:
					#discrete
					for index,child in enumerate(node.children):
						if child.isLeaf:
							if dataTest[indexAttribute]==self.attributes[index]:
								return child.label
							# print(indent + node.label + " = " + self.attributes[index] + " : " + child.label)
						else:
							return self._predict(child, dataTest, indexTest)
				else:
					#numerical
					leftChild = node.children[0]
					rightChild = node.children[1]
					if leftChild.isLeaf:
						if attribute==node.label:
							# print('[1][<=] attribute ', attribute ,' dataTest[indexAttribute] ', dataTest[indexAttribute], ' node.threshold ', node.threshold)
							if float(dataTest[indexAttribute]) <= float(node.threshold) :
								return leftChild.label
						# print(indent + node.label + " <= " + str(node.threshold) + " : " + leftChild.label)
					else:
						# print(indent + node.label + " <= " + str(node.threshold)+" : ")
						return self._predict(leftChild, dataTest, indexTest)

					if rightChild.isLeaf:
						if attribute==node.label:
							return rightChild.label
							# print('[2][>] attribute ', attribute ,' dataTest[indexAttribute] ', dataTest[indexAttribute], ' node.threshold ', node.threshold)
							if float(dataTest[indexAttribute]) > float(node.threshold):
								return rightChild.label
						# print(indent + node.label + " > " + str(node.threshold) + " : " + rightChild.label)
					else:
						# print(indent + node.label + " > " + str(node.threshold) + " : ")
						return self._predict(rightChild, dataTest, indexTest)

	def generateTree(self):
		self.tree = self.recursiveGenerateTree(self.data, self.attributes)

	def generateTreeByGivenList(self, data):
		self.tree = self.recursiveGenerateTree(data, self.attributes)

	def recursiveGenerateTree(self, curData, curAttributes):
		allSame = self.allSameClass(curData)

		if len(curData) == 0:
			#Fail
			return Node(True, "Fail", None)
		elif allSame is not False:
			#return a node with that class
			return Node(True, allSame, None)
		elif len(curAttributes) == 0:
			#return a node with the majority class
			majClass = self.getMajClass(curData)
			return Node(True, majClass, None)
		else:
			(best,best_threshold,splitted) = self.splitAttribute(curData, curAttributes)
			remainingAttributes = curAttributes[:]
			remainingAttributes.remove(best)
			node = Node(False, best, best_threshold)
			node.children = [self.recursiveGenerateTree(subset, remainingAttributes) for subset in splitted]
			return node

	def getMajClass(self, curData):
		freq = [0]*len(self.classes)
		for row in curData:
			index = self.classes.index(row[-1])
			freq[index] += 1
		maxInd = freq.index(max(freq))
		return self.classes[maxInd]


	def allSameClass(self, data):
		for row in data:
			if row[-1] != data[0][-1]:
				return False
		return data[0][-1]

	def isAttrDiscrete(self, attribute):
		if attribute not in self.attributes:
			raise ValueError("Attribute not listed")
		elif len(self.attrValues[attribute]) == 1 and self.attrValues[attribute][0] == "continuous":
			#numerik
			return False
		else:
			#kategorikal
			return True

	def splitAttribute(self, curData, curAttributes):
		splitted = []
		maxEnt = -1*float("inf")
		best_attribute = -1
		#None for discrete attributes, threshold value for continuous attributes
		best_threshold = None
		for attribute in curAttributes:
			indexOfAttribute = self.attributes.index(attribute)
			if self.isAttrDiscrete(attribute):
				#split curData into n-subsets, where n is the number of 
				#different values of attribute i. Choose the attribute with
				#the max gain
				valuesForAttribute = self.attrValues[attribute]
				subsets = [[] for a in valuesForAttribute]
				for row in curData:
					for index in range(len(valuesForAttribute)):
						if row[i] == valuesForAttribute[index]:
							subsets[index].append(row)
							break
				e = self.gain(curData, subsets)
				if e > maxEnt:
					maxEnt = e
					splitted = subsets
					best_attribute = attribute
					best_threshold = None
			else:
				#sort the data according to the column.Then try all 
				#possible adjacent pairs. Choose the one that 
				#yields maximum gain
				curData.sort(key = lambda x: x[indexOfAttribute])
				for j in range(0, len(curData) - 1):
					if curData[j][indexOfAttribute] != curData[j+1][indexOfAttribute]:
						threshold = (curData[j][indexOfAttribute] + curData[j+1][indexOfAttribute]) / 2
						less = []
						greater = []
						for row in curData:
							if(row[indexOfAttribute] > threshold):
								greater.append(row)
							else:
								less.append(row)
						e = self.gain(curData, [less, greater])
						if e >= maxEnt:
							splitted = [less, greater]
							maxEnt = e
							best_attribute = attribute
							best_threshold = threshold
		return (best_attribute,best_threshold,splitted)

	def gain(self,unionSet, subsets):
		#input : data and disjoint subsets of it
		#output : information gain
		#S adalah jumlah seluruh data dalam dataset
		S = len(unionSet)
		#calculate impurity before split
		impurityBeforeSplit = self.entropy(unionSet)
		#calculate impurity after split
		weights = []

		#calculate split info value
		split_info_value=0
		for i, subset in enumerate(subsets):
			#len(subset) adalah jumlah kasus per kelas
			weights.append(float(float(len(subset))/float(S)))
			split_info_value -= float(float(len(subset))/float(S)) * math.log(float(float(len(subset))/float(S)), 2)
		impurityAfterSplit = 0.0

		chiValue = 0
		for i in range(len(subsets)):
			#weights[i] adalah jumlah kasus per kelas / jumlah seluruh kasus
			#jumlah kasus per kelas / jumlah seluruh kasus * entropi
			impurityAfterSplit += float(weights[i]) * self.entropy(subsets[i], i)
			#chiValue adalah kolom W di excel
			chiValue += self.chisquare(subsets[i], i)

		#calculate total gain
		totalGain = impurityBeforeSplit - impurityAfterSplit
		gainRatio = float(totalGain)/float(split_info_value)
		cramerValue = self.cramer(chiValue, float(S))
		gainRatioPowCramerValue = math.pow(gainRatio, cramerValue)
		# print('Gain = ', totalGain , ', Split Info = ', split_info_value,', gainRatio = ', gainRatio)


		if self.criterion=='information_gain':
			return totalGain
		elif self.criterion=='gain_ratio':
			return gainRatio
		elif self.criterion=='cramer':
			return gainRatioPowCramerValue

		return totalGain

	def chisquare(self, dataSet, main_index=0):
		S = len(dataSet)
		if S == 0:
			return 0
		# num_classes = [0 for i in self.classes]
		num_classes = []
		for index, i in enumerate(self.classes):
			num_classes.append(0)

		for row in dataSet:
			classIndex = list(self.classes).index(row[-1])
			num_classes[classIndex] += 1
			
		num_classes_temp = []
		num_classes_total_per_class = {}
		for index, x in enumerate(num_classes):
			num_classes_total_per_class[index]=0

		for index, x in enumerate(num_classes):
			#x = jumlah kasus per kelas (kelas A + kelas n) -- di excel adalah P6
			#S = jumlah kasus seluruh kelas -- di excel adalah O6
			num_classes_temp.append(float(x)/float(S))
			#num_classes_total_per_class di excel adalah P9
			num_classes_total_per_class[index]+=float(x)

		num_classes = num_classes_temp

		#start menghitung chi
		#array_chi di excel adalah kolom S
		array_chi = {}
		sum_chi = 0
		for index, x in enumerate(num_classes):
			array_chi[index]=0

		for index, x in enumerate(num_classes):
			chi = float(float(x) * float(num_classes_total_per_class[index])) / float(S)
			#((P6-S6)^2)/S6
			if chi==0.0:
				array_chi[index] += 0
				sum_chi += 0
			else:
				array_chi[index] += (math.pow(float(x) - float(chi), 2)) / chi
				sum_chi += (math.pow(float(x) - float(chi), 2)) / chi
		#sum_chi adalah nilai chi per kelas
		#end menghitung chi
		return sum_chi

	def entropy(self, dataSet, main_index=0):
		S = len(dataSet)
		if S == 0:
			return 0
		# num_classes = [0 for i in self.classes]
		num_classes = []
		for index, i in enumerate(self.classes):
			num_classes.append(0)

		class_temp = []
		for row in dataSet:
			classIndex = list(self.classes).index(row[-1])
			num_classes[classIndex] += 1
			
		num_classes_temp = []

		for index, x in enumerate(num_classes):
			#x = jumlah kasus per kelas (kelas A + kelas n) -- di excel adalah P6
			#S = jumlah kasus seluruh kelas -- di excel adalah O6
			num_classes_temp.append(float(x)/float(S))

		num_classes = num_classes_temp

		ent = 0
		for num in num_classes:
			#num adalah target label / jumlah kasus per record
			ent += num * self.log(num)
		return ent*-1

	def cramer(self, chiValue=0, totalChase=0):
		#=W6/(14*(2-1))
		return math.sqrt(float(chiValue) / float(totalChase))

	def log(self, x):
		if x == 0:
			return 0
		else:
			return math.log(x,2)
	
	def setValidationValue(self, x):
		self.x_validation = x

	def makeModelXValidation(self, x_validation):
		cv = KFold(n_splits = x_validation, random_state=42, shuffle=False)
		# print(self.data[1])
		incrementTest=1
		
		akurasi = 0;
		for train_index, test_index in cv.split(self.data):
			# print("Train Index: ", train_index, "\n")
			# print("Test Index: ", test_index)
			# X_train, X_test, y_train, y_test = self.data[train_index], self.data[test_index], self.data[train_index], self.data[test_index]
			X_train = map(self.data.__getitem__, train_index)
			X_test = map(self.data.__getitem__, test_index)
			Y_train = self._getTargetLabel(X_train)
			Y_test = self._getTargetLabel(X_test)

			#generate model berdasarkan dataset yang sudah di split
			self.generateTreeByGivenList(X_train)

			#tampung hasil klasifikasi
			Y_predict = self.predict(X_test)

			#akurasi
			akurasi += metrics.accuracy_score(Y_test, Y_predict)

			print '-----------------------------------------------------Tree ke ',incrementTest,'----------------------------------------------------------------'
			self.printTree()
			incrementTest += 1
		print 'Akurasi : ', float(akurasi)/float(x_validation) * 100

	def _getTargetLabel(self, lst):
		return list( map(itemgetter(-1), lst )) 

class Node:
	def __init__(self,isLeaf, label, threshold):
		self.label = label
		self.threshold = threshold
		self.isLeaf = isLeaf
		self.children = []