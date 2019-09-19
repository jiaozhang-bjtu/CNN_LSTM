import scipy.io as sio
import numpy as np
'''
无关的数据相关度那一个值填充为0
按照感受野对应相关值
'''

CNN =['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'pool3', 'conv4', 'pool4']
path = 'D:/CNN_LSTM/label/'
def delt():
	for i in range(0,15):

		newpath = path + str(i)+ '.mat'
		load_data = sio.loadmat(newpath)
		load_matrix = load_data[CNN[0]]
		channel0 =load_matrix[0:9]
		array0 = np.array(channel0)
		array0 = np.reshape(array0, (-1,9))
		array0[0,3:9] = 0
		array0[1, 0] = 0
		array0[1,4:9] = 0
		array0[2, 0:2] = 0
		array0[2, 5:9] = 0
		array0[3, 0:3] = 0
		array0[3, 6:9] = 0
		array0[4, 0:4] = 0
		array0[4, 7:9] = 0
		array0[5, 0:5] = 0
		array0[5, 8] = 0
		array0[6, 0:6] = 0
		array0[7, 0:7] = 0
		array0[8, 0:8] = 0


		load_matrix1 = load_data[CNN[1]]
		channel1 = load_matrix1[0:9]
		array1 = np.array(channel1)
		array1 = np.reshape(array1, (-1, 9))
		array1[0, 5:9] = 0
		array1[1, 0:2] = 0
		array1[1, 7:9] = 0
		array1[2, 0:4] = 0
		array1[3, 0:6] = 0
		array1[4, 0:8] = 0

		load_matrix2 = load_data[CNN[2]]
		channel2 = load_matrix2[0:9]
		array2 = np.array(channel2)
		array2 = np.reshape(array2, (-1, 9))
		array2[1, 0:2] = 0
		array2[2, 0:4] = 0
		array2[3, 0:6] = 0
		array2[4, 0:8] = 0

		load_matrix3 = load_data[CNN[3]]
		channel3 = load_matrix3[0:9]
		array3 = np.array(channel3)
		array3 = np.reshape(array3, (-1, 9))
		array3[1, 0:4] = 0
		array3[2, 0:8] =  0

		load_matrix4 = load_data[CNN[4]]
		channel4 = load_matrix4[0:9]
		array4 = np.array(channel4)
		array4 = np.reshape(array4, (-1, 9))
		array4[1, 0:4] = 0
		array4[2, 0:8] = 0

		load_matrix5 = load_data[CNN[5]]
		channel5 = load_matrix5[0:9]
		array5 = np.array(channel5)
		array5 = np.reshape(array5, (-1, 9))
		array5[1, 0:8] = 0

		load_matrix6 = load_data[CNN[6]]
		channel6 = load_matrix6[0:9]
		array6 = np.array(channel6)
		array6 = np.reshape(array6, (-1, 9))
		array6[1, 0:8] = 0
		load_matrix7 = load_data[CNN[7]]
		channel7 = load_matrix7[0:9]
		array7 = np.array(channel7)
		array7 = np.reshape(array7, (-1, 9))
		sio.savemat('D:/CNN_LSTM/label/temp/'+str(i)+'.mat', {CNN[0]:array0, CNN[1]:array1, CNN[2]:array2, CNN[3]:array3, CNN[4]:array4, CNN[5]:array5, CNN[6]:array6, CNN[7]:array7 })


delt()




