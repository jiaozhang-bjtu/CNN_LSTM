import scipy.io as sio
import numpy as np

with open("H:/SpaceWork/EEG_Work/lable.txt") as file_object:
	lines = file_object.readlines()  # 从文件中读取每一行，将获取的内容放到list里
label = []
for line in lines:
	label.append(line.strip())  # 将每行地址追加到一个数组里

def one_hot(labels, n_class = 7):
	""" One-hot encoding """
	expansion = np.eye(n_class)
	y = expansion[:, labels-1].T
	assert y.shape[1] == n_class, "Wrong number of labels!"
	return y

def label_conv1(data, num, batchsize):
	'''
    :param data:     得到的预测数据
    :param num:      第几个人的数据，主要得到对应的标签
    :param batchsize:这个人总共有几个样本
    :return:         对于每个试验和每个类，如果试验是给定的类，我们构造了值为1的向量，如果是另一个类，我们构造了值0
    raw_data 5000 return 5000
    '''
	tem = []
	for i in range(batchsize):
		if data[i] == int(label[num])-1 :
			for t in range(5000):
				tem.append(1)
		else:
			for t in range(5000):
				tem.append(0)

	res = np.array(tem)
	res = np.reshape(res,(-1))
	return res

def label_pool1(data, num, batchsize):
	'''
    :param data:     得到的预测数据
    :param num:      第几个人的数据，主要得到对应的标签
    :param batchsize:这个人总共有几个样本
    :return:         对于每个试验和每个类，如果试验是给定的类，我们构造了值为1的向量，如果是另一个类，我们构造了值0
    raw_data 5000 return 2500
    '''
	tem = []
	for i in range(batchsize):
		if data[i] == int(label[num])-1 :
			for t in range(2500):
				tem.append(1)
		else:
			for f in range(2500):
				tem.append(0)

	res = np.array(tem)
	res = np.reshape(res,(-1))
	return res
def label_pool2(data, num, batchsize):
	'''
    :param data:     得到的预测数据
    :param num:      第几个人的数据，主要得到对应的标签
    :param batchsize:这个人总共有几个样本
    :return:         对于每个试验和每个类，如果试验是给定的类，我们构造了值为1的向量，如果是另一个类，我们构造了值0
      1250
    '''
	tem = []
	for i in range(batchsize):
		if data[i] == int(label[num])-1:
			for t in range(1250):
				tem.append(1)
		else:
			for f in range(1250):
				tem.append(0)

	res = np.array(tem)
	res = np.reshape(res,(-1))
	return res

def label_pool3(data, num, batchsize):
	'''
    :param data:     得到的预测数据
    :param num:      第几个人的数据，主要得到对应的标签
    :param batchsize:这个人总共有几个样本
    :return:         对于每个试验和每个类，如果试验是给定的类，我们构造了值为1的向量，如果是另一个类，我们构造了值0
     data shape 1250
    '''
	tem = []
	for i in range(batchsize):
		if data[i] == int(label[num])-1 :
			for t in range(625):
				tem.append(1)
		else:
			for f in range(625):
				tem.append(0)

	res = np.array(tem)
	res = np.reshape(res,(-1))
	return res

def label_pool4(data, num, batchsize):
	'''
    :param data:     得到的预测数据
    :param num:      第几个人的数据，主要得到对应的标签
    :param batchsize:这个人总共有几个样本
    :return:         对于每个试验和每个类，如果试验是给定的类，我们构造了值为1的向量，如果是另一个类，我们构造了值0
    data shape 313
    '''
	tem = []
	for i in range(batchsize):
		if data[i] == int(label[num])-1 :
			for t in range(313):
				tem.append(1)
		else:
			for f in range(313):
				tem.append(0)

	res = np.array(tem)
	res = np.reshape(res,(-1))
	return res
def label_fc1(data, num, batchsize):
	'''
    :param data:     得到的预测数据
    :param num:      第几个人的数据，主要得到对应的标签
    :param batchsize:这个人总共有几个样本
    :return:         对于每个试验和每个类，如果试验是给定的类，我们构造了值为1的向量，如果是另一个类，我们构造了值0
    data shape 512
    '''
	tem = []
	for i in range(batchsize):
		if data[i] == int(label[num])-1 and i !=10:
			for t in range(512):
				tem.append(1)
		else:
			for f in range(512):
				tem.append(0)

	res = np.array(tem)
	res = np.reshape(res,(-1))
	return res
def label_fc2(data, num, batchsize):
	'''
    :param data:     得到的预测数据
    :param num:      第几个人的数据，主要得到对应的标签
    :param batchsize:这个人总共有几个样本
    :return:         对于每个试验和每个类，如果试验是给定的类，我们构造了值为1的向量，如果是另一个类，我们构造了值0
    独热编码

    '''
	tem = []
	for i in range(batchsize):
		t = one_hot(data[i])
		tem.append(t)
	res = np.array(tem)
	res = np.reshape(res,(-1))
	return res


