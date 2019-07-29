import numpy as np
import random 
from sklearn.metrics import roc_auc_score

train_file = open('train.txt','r')
test_file = open('test.txt','r')

train_data = np.zeros((943,1682))
test_data = np.zeros((943,1682))
pu = np.random.randn(943,10) * 0.5
qi = np.random.randn(1682,10) * 0.5
learning_rate = 0.01
lamda = 0.01 

def sigmoid(x):
	output = 1 / (1 + np.exp(-x))
	return output

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

def train():
	global pu
	global qi 
	for u in range(943):
		for i in range(1682):
			if train_data[u][i] > 3:
				j = random.randint(0,1681)
				while train_data[u][j] > 3:
					j = random.randint(0,1681)
				xuij = np.dot(pu[u],qi[i].T) - np.dot(pu[u],qi[j].T)
				delta = 1.0 / (np.exp(-xuij) + 1)
				temp = pu[u]
				pu[u] += -learning_rate * ((1-delta)*(qi[j]-qi[i]) + lamda*pu[u])
				qi[i] += -learning_rate * ((1-delta)*(-temp) + lamda*qi[i])
				qi[j] += -learning_rate * ((1-delta) * temp + lamda*qi[j])

def predict(res_user, res_item):
	res_predict = np.zeros(1586126)
	for i in range(943):
		for j in range(1682):
			res_predict[i*1682+j] = np.dot(res_user[i], res_item[j].T)
	return res_predict

for i in range(200):
	print "iter %i"%i
	train()

res_predict = predict(pu,qi)
test = np.zeros(1586126)
for i in range(943):
	for j in range(1682):
		if test_data[i][j] > 3:
			test[i*1682+j] = 1
		else:
			test[i*1682+j] = 0

res = roc_auc_score(test,res_predict)
print res 










































