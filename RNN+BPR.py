import numpy as np 
import random
import json
import sys 

USER_SIZE = 1904
ITEM_SIZE = 1157
HIDDEN_SIZE = 10 
LEARNING_RATE = 0.1
LAMBDA = 0.001
TOP = 20

U = np.random.randn(HIDDEN_SIZE,HIDDEN_SIZE) * 0.5
W = np.random.randn(HIDDEN_SIZE,HIDDEN_SIZE) * 0.5
X = np.random.randn(ITEM_SIZE,HIDDEN_SIZE) * 0.5
H_ZERO = np.zeros((1, HIDDEN_SIZE))

DATAFILE = 'user_cart_basic.json'

ITEM_TRAIN = {}
ITEM_TEST = {}
SPLIT = 0.9

def sigmoid(x):
	output = 1.0/(1.0+np.exp(-x))
	return output

def pre_data():
	global ITEM_TRAIN
	global ITEM_TEST
	global SPLIT 
	global DATAFILE

	all_cart = []
	data = open(DATAFILE,'r')
	files = data.readlines()
	for file in files:
		line = json.loads(file)
		all_cart.append(line)
	# print all_cart[0]
	for i in xrange(len(all_cart)):
		item_train = []
		item_test = []
		behavior_list = all_cart[i]
		behavior_train = behavior_list[0:int(SPLIT*len(behavior_list))]
		behavior_test = behavior_list[int(SPLIT*len(behavior_list)):]
		for behavior in behavior_train:
			item_train.append(behavior[0])
		for behavior in behavior_test:
			item_test.append(behavior[0])
		ITEM_TRAIN[i] = item_train
		ITEM_TEST[i] = item_test 

def train(user_cart):
	global U,X,W
	hiddenlist = []
	dhlist = []
	midlist = []
	hl = np.copy(H_ZERO)
	du = 0
	dw = 0
	loss = 0
	dhl = np.copy(H_ZERO)
	# BPR
	for i in xrange(len(user_cart)-1):
		neg = random.randint(1,ITEM_SIZE)
		while user_cart[i+1] == neg:
			neg = random.randint(1,ITEM_SIZE)
		item_pos = X[user_cart[i+1]-1, :].reshape(1, HIDDEN_SIZE)		# positive sample's vector
		item_cur = X[user_cart[i]-1, :].reshape(1, HIDDEN_SIZE)		# current input vector
		item_neg = X[neg-1, :].reshape(1, HIDDEN_SIZE)
		b = np.dot(item_cur,U) + np.dot(hl,W)
		h = sigmoid(b)
		xi_j = item_pos.T - item_neg.T
		xij = np.dot(h,xi_j)
		loss += xij
		tmp = -(1-sigmoid(xij))

		hiddenlist.append(h)
		mid = h * (1-h)
		midlist.append(mid)
		dhlist.append(tmp*(item_pos - item_neg))

		dpos = tmp*h + LAMBDA*item_pos 
		X[user_cart[i+1]-1,:] += -LEARNING_RATE*(dpos.reshape(HIDDEN_SIZE,))
		dneg = -tmp*h + LAMBDA*item_neg
		X[neg-1,:] += -LEARNING_RATE*(dneg.reshape(HIDDEN_SIZE,))

		hl = h

	# BPTT
	for i in range(len(user_cart)-1)[::-1]:
		item = X[user_cart[i]-1,:].reshape(1,HIDDEN_SIZE)
		h_op = hiddenlist[i]
		dh = dhlist[i] + dhl

		du += np.dot(item.T,dh*midlist[i])
		dw += np.dot(h_op.T,dh*midlist[i])

		dx = np.dot(dh*midlist[i],U.T)
		X[user_cart[i]-1,:] += -LEARNING_RATE * (dx.reshape(HIDDEN_SIZE,) + LAMBDA*X[user_cart[i]-1,:])
		dhl = np.dot(dh*midlist[i],W.T)
	U += -LEARNING_RATE*(du+LAMBDA*U)
	W += -LEARNING_RATE*(dw+LAMBDA*W)
	return loss

def predict():
	relavant = 0.0
	hit = {}
	recall = {}
	recallres = {}

	for i in range(TOP):
		hit[i+1] = 0
		recall[i+1] = 0
	for n in ITEM_TEST.keys():
		train = ITEM_TRAIN[n]
		test = ITEM_TEST[n]
		hl = np.copy(H_ZERO)
		h = np.copy(H_ZERO)

		for item_id in train:
			item = X[item_id-1]
			b = np.dot(item,U) + np.dot(hl,W)
			h = sigmoid(b)
			hl = h

		for j in range(len(test)):
			relavant += 1
			predict_matrix = np.dot(h,X.T)
			rank = np.argpartition(predict_matrix[0],-TOP)[-TOP:]
			rank = rank[np.argsort(predict_matrix[0][rank])]
			rank_index_list = list(reversed(list(rank)))
			if test[j]-1 in rank_index_list:
				index = rank_index_list.index(test[j]-1)
				hit[index+1] += 1
			item = X[test[j]-1]
			b = np.dot(item,U) + np.dot(h,W)
			h = sigmoid(b)
	for i in range(20):
		for j in range(20-i):
			recall[20-j] += hit[i+1]
	for i in range(20):
		recallres[i+1] = recall[i+1]/relavant

	print relavant
	print recall
	print recallres

	# for i in range(20):
	# 	for j in range(20-i):

def learn():
	global X
	iter = 0
	while iter < 200:
		print "Iter %d" % iter
		sumloss = 0
		for i in ITEM_TRAIN.keys():
			user_cart = ITEM_TRAIN[i]
		# print X[user_cart[1]-1,:]
		# print len(X[0]) 
			loss = train(user_cart)
			sumloss += loss
		print sumloss

		predict()

		iter += 1
	


def main():
	pre_data()
	learn()

if __name__ == "__main__":
	main()


