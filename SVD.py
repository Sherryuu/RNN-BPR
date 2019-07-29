import numpy as np 
import math 

train_file = open('train.txt','r')
test_file = open('test.txt','r')

# item = []

# for line in test_file:
# 	line = line.split('\t')
# 	item_id = line[1]
# 	item.append(int(item_id))

# item.sort()
# print item[-1]

train_data = np.zeros((943,1679))
test_data = np.zeros((943,1682))
pu = np.random.randn(943,10) * 0.5
qi = np.random.randn(1682,10) * 0.5
learning_rate = 0.005
lamda = 0.02

for file in train_file:
	file = file.split('\t')
	user = int(file[0])
	item = int(file[1])
	score = int(file[2])
	train_data[user-1][item-1] = score

for file in test_file:
	file = file.split('\t')
	user = int(file[0])
	item = int(file[1])
	score = int(file[2])
	test_data[user-1][item-1] = score

def avg(matrix):
	count = 0
	total = 0
	for i in range(len(matrix)):
		for j in range(len(matrix[0])):
			if matrix[i][j] != 0:
				count += 1
				total += matrix[i][j]
	total_avg = total / count
	return total_avg

def biases_user(matrix,avg):
	biases = []
	for i in range(len(matrix)):
		count = 0
		total = 0
		for j in range(len(matrix[0])):
			if matrix[i][j] != 0:
				count += 1
				total += matrix[i][j] - avg
			if count == 0:
				biases.append(count)
			else:
				biases.append(total/count)
	return biases 

def biases_item(matrix,avg):
	biases = []
	for i in range(len(matrix[0])):
		count = 0
		total = 0
		for j in range(len(matrix)):
			if matrix[j][i] != 0:
				count += 1
				total += matrix[j][i] - avg
			if count == 0:
				biases.append(count)
			else:
				biases.append(total/count)
	return biases

avg_train = avg(train_data)
avg_test = avg(test_data)

def train():
	global pu
	global qi 
	biases_u1 = biases_user(train_data, avg_train)
	biases_i1 = biases_item(train_data,avg_train)
	for i in range(len(train_data)):
		for j in range(len(train_data[0])):
			if train_data[i][j] != 0:
				rui = avg_train + biases_u1[i] + biases_i1[j] + np.dot(pu[i],qi[j].T)
				eui = train_data[i][j] - rui
				biases_u1[i] += learning_rate * (eui - lamda * biases_u1[i])
				biases_i1[j] += learning_rate * (eui - lamda * biases_i1[j])
				temp = qi[j]
				qi[j] += learning_rate * (eui * pu[i] - lamda * qi[j])
				pu[i] += learning_rate * (eui * temp - lamda * pu[i])


def test():
	count = 0
	total = 0
	biases_u2 = biases_user(test_data,avg_test)
	biases_i2 = biases_item(test_data,avg_test)
	for i in range(len(test_data)):
		for j in range(len(test_data[0])):
			if test_data[i][j] != 0:
				rui = avg_test + biases_u2[i] + biases_i2[j] + np.dot(pu[i],qi[j].T)
				eui = test_data[i][j] - rui
				count += 1
				total += eui * eui 
	loss = math.sqrt(total / count)
	return loss 

for i in range(100):
	print i 
	train()
	final = test()
	print final




























































































