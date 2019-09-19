import scipy.io as sio
import numpy as np


CNN =['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'pool3', 'conv4','pool4']
path = 'D:/eliminateBP/MRS/delta_max/eliminateBP'
num4 = [0,5,6,7,8]
num5 = [1,2,3,4,11,13]
num1 = [9,12]
num6 = [10]
num0 = [14]
def delt():

	path1 = path + str(num4[0] + 1) + '.mat'
	load_matrix1 = sio.loadmat(path1)
	load_matrix1 = load_matrix1['data']
	batch = load_matrix1
	for i in range(1,5):
		path1 = path + str(num4[i]+1) + '.mat'
		load_matrix1 = sio.loadmat(path1)
		load_matrix1 = load_matrix1['data']
		batch += load_matrix1
	batch = np.array(batch)
	batch = batch / 5
	sio.savemat('D:/eliminateBP/MRS/deltaClassMean/class4.mat', {'data': batch})





	# for i in range(0,15):
	#
	# 	newpath = path + str(i)+ '.mat'
	# 	load_data = sio.loadmat(newpath)
	# 	list = []
	# 	for channel in range(8):
	# 		load_matrix = load_data[CNN[channel]] # 取不同隐藏层后的相关系数
	# 		channel0 =load_matrix[0:9] #
	# 		array0 = np.array(channel0)
	# 		array0 = np.reshape(array0, (-1,9)) #
	# 		temp = []
	# 		for j in range(9):
	# 			emp1 = array0[:, j] # 取第j列的所有数据
	# 			emp1 = np.reshape(emp1, -1)
	#
	# 			max1 = max(emp1, key=abs)
	# 			temp.append(max1)
	#
	# 		list.append(temp)
	# 	list = np.array(list)
	# 	list = np.reshape(list,(-1,9))
	#
	# 	sio.savemat('D://max_delta/delta/'+str(i)+'.mat', {'data':list })
delt()