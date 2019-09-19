import scipy.io as sio
import numpy as np
'''
取每个通道在每个隐藏层绝对值最大的相关性
'''

CNN =['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'pool3', 'conv4','pool4']
path = 'D:/CNN_LSTM/label/temp/'
def delt():
	for i in range(0,15):

		newpath = path + str(i)+ '.mat'
		load_data = sio.loadmat(newpath)
		list = []
		for channel in range(8):
			load_matrix = load_data[CNN[channel]] # 取不同隐藏层后的相关系数
			channel0 =load_matrix[0:9] #
			array0 = np.array(channel0)
			array0 = np.reshape(array0, (-1,9)) #
			temp = []
			for j in range(9):
				emp1 = array0[:, j] # 取第j列的所有数据
				emp1 = np.reshape(emp1, -1)

				max1 = max(emp1, key=abs) #取绝对值最大然后返回原来值
				temp.append(max1)

			list.append(temp)
		list = np.array(list)
		list = np.reshape(list,(-1,9))

		sio.savemat('D:/CNN_LSTM/label/temp/'+str(i)+'.mat', {'data':list })
delt()
# newpath = path + str(0)+ '.mat'
# load_data = sio.loadmat(newpath)
# load_matrix = load_data[CNN[0]]
# channel0 =load_matrix[:]
# array0 = np.array(channel0)
# array0 = np.reshape(array0, (-1,9))
# temp = []
# for i in range(9):
# 	emp1  = array0[:,i]
# 	emp1 = np.reshape(emp1,-1)
# 	print(emp1)
# 	max1 = max(emp1, key=abs)
# 	temp.append(max1)
# temp =np.array(temp)
# print(temp)





