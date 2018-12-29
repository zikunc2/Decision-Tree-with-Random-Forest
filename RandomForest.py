'''
Random Forest Ensembled Version Decision Tree Classifier with gini-index

Python version: Python 3.6.5
Command line: python3 RandomForest.py somedataname.train somedataname.tests
'''

from itertools import compress
import sys
import copy
import random
# import time

'''
Take in a file path, and parse the file
Return a list of tuples(result, value of attirbute1, value of attribute2 ...)
'''
def readDataFromFile(inputFilePath):
	file = open(inputFilePath, 'r')
	line = file.readline()
	dec = []
	data = []
	while line:
		tup = []
		temp = line.split(' ')
		# get decision value
		dec.append(temp[0])
		# get attribute values
		for i in range(1, len(temp)):
			temp_att = temp[i].split(':')
			temp_num = temp_att[1]
			tup.append(int(temp_num))
		data.append(tup)
		line = file.readline()
	file.close()
	return (dec, data)

'''
find the smallest gini index of given att
'''
def smallest_gini(att_val, y):
	min_gini = 0
	att_val_set = list(set(att_val))
	for a in att_val_set:
		# create selector to compress a given value of an att
		selector = []
		for j in att_val:
			if j == a:
				selector.append(1)
			else:
				selector.append(0)
		a_y = list(compress(y,selector))
		# calculate the gini index of a given value of att
		curr_gini = 1
		counter = {}
		for b in a_y:
			if b not in counter:
				counter[b] = 1
			else:
				counter[b] += 1
		for count in counter.values():
			curr_gini -= (count/len(a_y))**2
		min_gini += len(a_y) * curr_gini / len(y)
	# print (min_gini)
	return min_gini

'''
find the attribute to be partition
'''
def part_att(x, y):
	num_att = len(x[0])
	can_att = num_att + 1
	temp = 2
	for i in range(num_att):
		att_val = list(zip(*x))[i]
		# find the smallest gini
		min_gini = smallest_gini(att_val, y)
		# find the attribute with the smallest index
		if min_gini < temp:
			# print ("here")
			temp = min_gini
			can_att = i
	return can_att


'''
train : recursively build up the decision tree by gini index
'''
def training(x, y, count, k, fbagging_size):
	if count == k:
		# print ("---------------------------------------")
		return max(set(y), key = y.count)
	else:
		x_copy = copy.deepcopy(x)
		sub_tree = {}
		x_temp = []
		if(k - count <= fbagging_size):
			x_temp = x.copy()
		else:
			fselector = [0] * k
			for j in range (fbagging_size):
				rda = random.randint(0, k - 1)
				while (fselector[rda] == 1):
					rda = random.randint(0, k - 1)
				fselector[rda] = 1
			for elem in x_copy:
				for sel in fselector:
					new_elem = list(compress(elem, fselector))
				x_temp.append(new_elem)

			# print(fselector)


		gini_att = part_att(x_temp,y)
		gini_att_vals = list(zip(*x))[gini_att]
		# count occurence of each value of a given att
		child = list(set(gini_att_vals))
		# for every value in the choosen att
		for i in child:
			# create selector to compress a given value of an att
			selector = []
			for j in gini_att_vals:
				if j == i:
					selector.append(1)
				else:
					selector.append(0)
			sub_x = x.copy()
			sub_y = y.copy()
			sub_data = list(compress(sub_x, selector))
			sub_dec = list(compress(sub_y, selector))
			for data in sub_data:
				del data[gini_att]
			sub_tree[(gini_att, i)] = training(sub_data, sub_dec, count+1,k, fbagging_size)
		return sub_tree

'''
run test on given data
'''
def testing(x, y, k, B, forest):
	# initialize confusion matrix
	conf_mat = []
	mat_size = int(max(y))
	for a in range(mat_size):
		row = []
		for b in range(mat_size):
			row.append(0)
		conf_mat.append(row)

	# guessing based on decision trees
	accuracy = 0
	index = 0
	correct = 0
	# guess = []
	for i in x:
		# ------------- for every line of data x -------------
		# initialize vote list (index - dec, val - number of votes on given dec)
		vote = []
		for j in range(mat_size):
			vote.append(0)
		# go thru each tree in forest
		for tree_index in range(B):
			temp_tree = forest[tree_index]
			temp_i = i.copy()
			for j in range(k):
				att = random.choice(list(temp_tree.keys()))[0]
				if (att, temp_i[att]) not in temp_tree.keys():
					dec = temp_tree[random.choice(list(temp_tree.keys()))]
				else:
					dec = temp_tree[(att, temp_i[att])]
				temp_tree = dec
				del temp_i[att]
			if int(dec) > int(max(y)):
				dec = int(max(y))
			vote[int(dec)-1] += 1
		max_val = max(vote)
		vote_dec = vote.index(max_val) + 1
		# guess.append(vote_dec)
		conf_mat[int(y[index])-1][int(vote_dec)-1] += 1
		index += 1

	# print results to screen
	# print ("Confusion Matrix:")
	for i in range(mat_size):
		line = ""
		for j in range(mat_size):
			line = line + str(conf_mat[i][j])
			if i == j:
				correct += conf_mat[i][j]
			if j != mat_size - 1:
				line = line + " "
		print(line)
	# print ("correct", correct)
	acc = correct/len(y)
	# print ("Overall accuarcy:", acc)

	tp = []
	fp = []
	fn = []
	col_totals = [sum(a) for a in zip(*conf_mat)]
	# print ("col_totals", col_totals)
	for row in range(mat_size):
		fn_temp = 0
		for col in range(mat_size):
			if row == col:
				tp.append(conf_mat[row][col])
			else:
				fn_temp += conf_mat[row][col]
		fn.append(fn_temp)
	for b in range(len(col_totals)):
		fp.append(col_totals[b] - tp[b])
	# print ("tp", tp)
	# print ("fn", fn)
	# print ("fp", fp)
	# print ("Overall accuarcy:", acc)

	# f1 = []
	for a in range(mat_size):
		temp = 2*tp[a] / (2*tp[a] + fp[a] + fn[a])
		# print ("f1 score for class", a+1, "is", temp)
		# f1.append(2*tp[a] / (2*tp[a] + fp[a] + fn[a]))
	# return acc


if __name__ == '__main__':
	# start = time.time()
	train_file = sys.argv[1]
	test_file = sys.argv[2]

	train = readDataFromFile(train_file)
	train_x = train[1]
	train_y = train[0]
	test = readDataFromFile(test_file)
	test_x = test[1]
	test_y = test[0]

	num_att = len(train_x[0])

	'''
	If you do not have any concern regarding the computation
	times, the more trees you have, the better (reliable) estimates
	you get from out-of-bag predictions
	'''

	# data bagging:
	# use subset of traning data by sampling with replacemnt for each tree
	# free parameter B, number of samples/trees 10, 30, 100
	B = 30
	forest = {}
	# print ("number of trees:", B)
	for i in range(B):
		smaller = max(1, int(len(train_x)/10))
		larger = int(len(train_x)/5)
		dbagging_size = random.randint(smaller, larger)
		# dbagging_size = int(len(train_x)/3)
		index_list = random.sample(range(len(train_x)), dbagging_size)
		selector = []
		for j in range(len(train_x)):
			if j in index_list:
				selector.append(1)
			else:
				selector.append(0)
		# print (selector)
		# print ("----------------------------------------------------")
		# print ("length of selector", len(selector))
		# print ("expected length", len(train_x))
		fbagging_size = min(num_att, int((num_att)**(1/2))+1)
		dbagging_train_x = list(compress(train_x, selector))
		dbagging_train_y = list(compress(train_y, selector))
		# print (dbagging_train_x)
		# print ("----------------------------------------------------")
		# print ("length of train x", len(dbagging_train_x))
		# print ("expected length", len(train_x))

		# build decision tree for each tree
		in_train_x = copy.deepcopy(dbagging_train_x)
		in_train_y = copy.deepcopy(dbagging_train_y)
		k = min(9, num_att)
		tree = training(in_train_x, in_train_y, 1, k, fbagging_size)
		forest[i] = tree

	# train and test
	testing(test_x, test_y, k-1, B, forest)
	# end = time.time()
	# print ("time consumed:", end-start)
