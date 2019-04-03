#from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import math
import operator
from sklearn import tree


def rm_main(dataset):

	### fungsi untuk menghitung root entropy ###
	def root_entropy_func(class_label, new_root):
		root = {}
		if not new_root:

			jml_kasus = len(df)
			entropy = 0.0
			log2 = math.log(2)
			simpul = 'akar'
			labels = {}
			for label in class_label:
				column_class = df.loc[:,['label']]
				column_class = column_class.values[column_class == label]
				# print len(column_class),'xx'
				sum_class = float(len(column_class))
				root.update({label :sum_class})
				sum_class = sum_class / jml_kasus
				entropy += -sum_class * (math.log(sum_class)/log2)
			root.update({'simpul': simpul, 'jml_kasus': jml_kasus,'entropy': entropy})
		else:
			root = new_root[-1]
		return root

	### fungsi untuk menghitung entropy ###
	def entropy_func(atribut, new_root):
		# hapus simpul yg sudah menjadi root
		atribut = list(atribut)
		if new_root:
			for lst in new_root:
				simpul = lst.get('simpul')
				atribut.remove(simpul)
	
		log2 = math.log(2)
		tampung = []
		for attr in atribut:
			df_column_attr = df.loc[:,[attr]]
			column_attr = list(set([cols[0] for cols in df_column_attr.values]))
			for vals in column_attr:
				entropy = 0.0
				query_ex = '('+ str(attr) + ' == ' + "'" + str(vals) + "'" + ')'
				if new_root:
					for lst in new_root:
						simpul = lst.get('simpul')
						new_root_val = lst.get('val')
						query_ex_plus = ' & ('+ str(simpul) + ' == ' + "'" + str(new_root_val) + "'" + ')'
						query_ex += query_ex_plus
				indexs_list = df.query(query_ex, engine='python').index.tolist()
				jml_kasus = len(indexs_list)
				data_ids = df.ix[indexs_list]
				data_ids_val = data_ids['label'].values
				label_val = {}
				for (index, labels) in enumerate(class_label):
					sum_class = int(len(data_ids_val[data_ids['label'].astype(str).values == labels]))
					label_val.update({labels: sum_class})
					# print '1.Sum Class ', labels, ' ', sum_class
					if jml_kasus==0:
						sum_class = 0.0
					else :
						# print 'Not Null ', sum_class, '/', jml_kasus
						sum_class = float(sum_class)/float(jml_kasus)
					# print '2.Sum Class ', labels, ' ',sum_class, ' Jumlah kasus ' , jml_kasus
					entropy += -sum_class * ((math.log(sum_class) if sum_class else 0.0)/log2)
					# print '3.Entropy ', labels, ' ',entropy, ' Jumlah kasus ' , jml_kasus
				values = {'simpul': attr, 'jml_kasus': jml_kasus, 'entropy': entropy, 'val': vals}
				for k, v in label_val.iteritems():
					values.update({k: v})
				tampung.append(values)
		return tampung

	### fungsi untuk split atribut ###
	def gain_func(t=False):
		root_entropy = root_entropy_func(class_label, new_root=t)
		entropy = entropy_func(atribut, new_root=t)
		print entropy
		df_entropy = pd.DataFrame(entropy)
		root = root_entropy
		root_simpul = root.get('simpul')
		root_jml_kasus = float(root.get('jml_kasus'))
		root_entropy = root.get('entropy')
		branchs = entropy
		tampung_gain = {}
		tampung_split_info = {}

		# hapus simpul yg sudah menjadi rootn
		atributs = list(atribut)
		if t:
			for lst in t:
				simpul = lst.get('simpul')
				atributs.remove(simpul)

		# purpose
		sum_gains = 0.0
		len_gains = 0
		tampung_for_zscore = {}

		for attr in atributs:
			count_gain = 0.0
			split_info = 0.0
			probabilitas = [] # purpose

			for branch in branchs:
				branch_simpul = branch.get('simpul')
				branch_jml_kasus = float(branch.get('jml_kasus'))
				branch_entropy = branch.get('entropy')
				for labels in class_label:
					branch_label = branch.get(labels)
				if attr == branch_simpul:
					# print attr, branch_simpul, branch_entropy
					count_gain += (branch_jml_kasus / root_jml_kasus) * branch_entropy
					log2 = math.log(2)
					count_split_info = branch_jml_kasus / root_jml_kasus
					split_info += -(count_split_info * (math.log(count_split_info) / log2)) if count_split_info else 0.0
					probabilitas.append(count_split_info)
					# print "Here ", branch_entropy

			gain = root_entropy - count_gain # info gain
			if gain>0:
				gain_ratio = gain / split_info
			else :
				gain_ratio = 0
			# if you want to change the split function, change in below
			tampung_gain.update({attr: gain_ratio})
			# purpose
			sum_gains += gain_ratio
			len_gains += 1
			redu_mc = 1 - max(probabilitas)
			tampung_for_zscore.update({attr: [gain_ratio, redu_mc]})

		# purpose
		avg_gain = sum_gains / len_gains
		# stdev calculation
		sum_exponent = 0.0
		for k, v in tampung_gain.iteritems():
			gain_min_avg = v - avg_gain
			exponent_gain_min_avg = gain_min_avg ** 2
			sum_exponent += exponent_gain_min_avg
		variant = sum_exponent / (16-1) if sum_exponent else 0.0
		stdev = math.sqrt(variant)
		tampung_performance = {}
		for k, v in tampung_for_zscore.iteritems():
			gain = v[0]
			gain_ratio = v[0]
			redu_mc = v[1]
			z_score = (gain_ratio - avg_gain) / stdev if (gain_ratio - avg_gain) else 0.0
			nilai_eksponensial = 2.718281828
			sigmoidal = (1 - (nilai_eksponensial ** (-1 * z_score))) / (1 + (nilai_eksponensial ** (-1 * z_score)))
			test_cost = 600.0/800.0
			performance = (((2 ** sigmoidal) - 1) * redu_mc) / (test_cost+1)
			tampung_performance.update({k: performance})
	
		max_gain = 0.0
		for attr in atributs:
			for gain in tampung_performance:
				if attr == gain:
					if max_gain is 0.0 or max_gain < tampung_performance[gain]:
						max_gain = tampung_performance[gain]
						max_key = gain

		if not max_gain:
				return []
		new_root = df_entropy[df_entropy.loc[:,['simpul']].values == max_key]

		term_new_root = []
		# jika entropy != 0 maka node dilanjutkan
		for i, v in new_root.iterrows():
			next_entropy = v['entropy']
			term_new_root.append(v.to_dict())
		return term_new_root

	# main data
	df =  pd.read_excel("/Users/ahmadauliawiguna/Documents/WORK/Riset/hepatitis_missing.xlsx")
	tot_jml_kasus = len(dataset)
	#class_label = df.loc[:, ['label']].values
	#class_label = list(set([labels[0] for labels in class_label]))
	# set manual label
	class_label = list(set(df.label.astype(str).values.flat))
	#atribut = df.columns.values[df.columns.values != 'label']
	# set manual attribute
	atribut = list(df.columns.astype(str).values)

	# Build Model
	node_ids = []
	tampung_prediksi = []
	if len(node_ids) < 1:
	   # menentukan node paling awal
	   nodes = gain_func(t=node_ids)
	   for node in nodes:
		   if node.get('entropy') != 0:
			   node_ids.append([node])
		   else:
			   tampung_prediksi.append([node])
	
	if len(node_ids) >= 1:
	   start_len = len(node_ids)
	   i = 1
	   for root_node in node_ids:
		   if len(root_node) == 1:
			   if root_node[0].get('entropy') != 0:
				   nodes = gain_func(t=root_node)
				   next_node_ids = []

				   for node in nodes:
					   next_node = [node]
					   next_node.insert(-1, root_node[0])
					   if node.get('entropy') != 0:
						   node_ids.append(next_node)
					   else:
						   tampung_prediksi.append(next_node)
			   else:
				   tampung_prediksi.append(root_node)
	
	   # check recursively
	   no = 0
	   for next_node in node_ids:
		   if no >= start_len:
			   if next_node[len(next_node)-1].get('entropy') != 0.0:
				   nodes = gain_func(t=next_node)
				   for node_id in nodes:
					   next_node_ids = next_node
					   continue_node = [node_id]
					   for loop_node in next_node_ids:
						   continue_node.insert(-1, loop_node)
					   if node_id.get('entropy') != 0.0:
						   node_ids.append(continue_node)
					   else:
						   tampung_prediksi.append(continue_node)
			   else:
				   tampung_prediksi.append(next_node)
		   no += 1
	   i += 1

	new_prediksi = []
	for pred_ids in tampung_prediksi:
	   if pred_ids[len(pred_ids)-1].get('jml_kasus') != 0:
		   new_prediksi.append(pred_ids)

	# Pohon
#	data_prediksi = pd.DataFrame(data=tampung_prediksi))
	return new_prediksi

print rm_main("/Users/ahmadauliawiguna/Documents/WORK/Riset/hepatitis_missing.xlsx")