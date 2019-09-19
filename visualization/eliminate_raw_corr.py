import scipy.io as sio
import numpy as np
'''
排除架构带来的差异，故将训练模型的数据的相关性减去未经训练的模型相关度
'''

CNN =['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'pool3', 'conv4', 'pool4']
path = 'D:/CNN_LSTM/alpha/'
path1 = 'D:/CNN_LSTM/raw/'
def eliminate_corr():
	for i in range(0,15):

		newpath = path + str(i) + '.mat'
		rawpath = path1 + str(i) + '.mat'
		load_data = sio.loadmat(newpath)
		load_raw_corr = sio.loadmat(rawpath)

		load_matrix = load_data[CNN[0]]
		load_raw_matrix = load_raw_corr[CNN[0]]
		channel0 = load_matrix[0:9]
		array0 = np.array(channel0)
		array0 = np.reshape(array0, (-1,9))
		raw_data0 = load_raw_matrix[0:9]
		raw_array0 = np.array(raw_data0)
		raw_array0 = np.reshape(raw_array0, (-1,9))
		array0 = array0 - raw_array0

		load_matrix1 = load_data[CNN[1]]
		load_raw_matrix1 = load_raw_corr[CNN[1]]
		channel1 = load_matrix1[0:9]
		array1 = np.array(channel1)
		array1 = np.reshape(array1, (-1, 9))
		raw_data1 = load_raw_matrix1[0:9]
		raw_array1 = np.array(raw_data1)
		raw_array1 = np.reshape(raw_array1, (-1, 9))
		array1 = array1 - raw_array1

		load_matrix2 = load_data[CNN[2]]
		load_raw_matrix2 = load_raw_corr[CNN[2]]
		channel2 = load_matrix2[0:9]
		array2 = np.array(channel2)
		array2 = np.reshape(array2, (-1, 9))
		raw_data2 = load_raw_matrix2[0:9]
		raw_array2 = np.array(raw_data2)
		raw_array2 = np.reshape(raw_array2, (-1, 9))
		array2 = array2 - raw_array2

		load_matrix3 = load_data[CNN[3]]
		load_raw_matrix3 = load_raw_corr[CNN[3]]
		channel3 = load_matrix3[0:9]
		array3 = np.array(channel3)
		array3 = np.reshape(array3, (-1, 9))
		raw_data3 = load_raw_matrix3[0:9]
		raw_array3 = np.array(raw_data3)
		raw_array3 = np.reshape(raw_array3, (-1, 9))
		array3 = array3 - raw_array3

		load_matrix4 = load_data[CNN[4]]
		load_raw_matrix4 = load_raw_corr[CNN[4]]
		channel4 = load_matrix4[0:9]
		array4 = np.array(channel4)
		array4 = np.reshape(array4, (-1, 9))
		raw_data4 = load_raw_matrix4[0:9]
		raw_array4 = np.array(raw_data4)
		raw_array4 = np.reshape(raw_array4, (-1, 9))
		array4 = array4 - raw_array4

		load_matrix5 = load_data[CNN[5]]
		load_raw_matrix5 = load_raw_corr[CNN[5]]
		channel5 = load_matrix5[0:9]
		array5 = np.array(channel5)
		array5 = np.reshape(array5, (-1, 9))
		raw_data5 = load_raw_matrix5[0:9]
		raw_array5 = np.array(raw_data5)
		raw_array5 = np.reshape(raw_array5, (-1, 9))
		array5 = array5 - raw_array5

		load_matrix6 = load_data[CNN[6]]
		load_raw_matrix6 = load_raw_corr[CNN[6]]
		channel6 = load_matrix6[0:9]
		array6 = np.array(channel6)
		array6 = np.reshape(array6, (-1, 9))
		raw_data6 = load_raw_matrix6[0:9]
		raw_array6 = np.array(raw_data6)
		raw_array6 = np.reshape(raw_array6, (-1, 9))
		array6 = array6 - raw_array6

		load_matrix7 = load_data[CNN[7]]
		load_raw_matrix7 = load_raw_corr[CNN[7]]
		channel7 = load_matrix7[0:9]
		array7 = np.array(channel7)
		array7 = np.reshape(array7, (-1, 9))
		raw_data7 = load_raw_matrix7[0:9]
		raw_array7 = np.array(raw_data7)
		raw_array7 = np.reshape(raw_array7, (-1, 9))
		array7 = array7 - raw_array7

		sio.savemat('D:/CNN_LSTM/alpha/eliminate/'+str(i)+'.mat', {CNN[0]:array0, CNN[1]:array1, CNN[2]:array2, CNN[3]:array3, CNN[4]:array4, CNN[5]:array5, CNN[6]:array6, CNN[7]:array7 })


eliminate_corr()




