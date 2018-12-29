'''
Decision Tree Classifier with gini-index as the attribute selection measure

Python version: Python 3.6.5
Command line: python3 DecisionTree.py somedataname.train somedataname.tests
'''

from itertools import compress
import sys
import copy
import random

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
def training(x, y, count, k):
	if count == k:
		return max(set(y), key = y.count)
	else:
		sub_tree = {} 
		gini_att = part_att(x,y)
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
			sub_tree[(gini_att, i)] = training(sub_data, sub_dec, count+1,k)
		return sub_tree

'''
run test on given data
'''
def testing(x, y, k, tree):
	# initialize confusion matrix
	conf_mat = []
	# mat_size = int(max(max(y),max(gue)))
	mat_size = int(max(y))
	for a in range(mat_size):
		row = []
		for b in range(mat_size):
			row.append(0)
		conf_mat.append(row)

	# guessing based on decision tree
	accuracy = 0
	index = 0
	correct = 0
	# guess = []
	for i in x:
		temp_tree = tree
		for j in range(k):
			att = list(temp_tree.keys())[0][0]
			if (att, i[att]) not in temp_tree.keys():
				dec = temp_tree[random.choice(list(temp_tree.keys()))]
			else:
				dec = temp_tree[(att, i[att])]
			temp_tree = dec
			del i[att]
		# print ("mat_size", mat_size)
		# print ("correct dec", int(y[index]))
		# print ("my dec", int(dec))
		if int(dec) > int(max(y)):
			dec = int(max(y))
		conf_mat[int(y[index])-1][int(dec)-1] += 1
		if dec == y[index]:
			correct += 1
		index += 1
	# 	guess.append(dec)
	# print ("guess: ", guess)
	acc = correct/len(y)

	# print results to screen
	# print ("Confusion Matrix:")
	for i in range(mat_size):
		line = ""
		for j in range(mat_size):
			line = line + str(conf_mat[i][j])
			if j != mat_size - 1:
				line = line + " "
		print(line)
	
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


if __name__ == '__main__':	
	train_file = sys.argv[1]
	test_file = sys.argv[2]

	train = readDataFromFile(train_file)
	train_x = train[1]
	train_y = train[0]
	test = readDataFromFile(test_file)
	test_x = test[1]
	test_y = test[0]

	# copy to memory for executing recursion for training
	in_train_x = copy.deepcopy(train_x)
	in_train_y = copy.deepcopy(train_y)

	# train
	k = min(9, len(train_x[0]))
	tree = training(in_train_x, in_train_y, 1, k)
	# print (tree)

	testing(test_x, test_y, k-1, tree)
