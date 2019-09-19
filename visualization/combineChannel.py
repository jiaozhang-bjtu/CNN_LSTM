import scipy.io as sio
import numpy as np


CNN =['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'pool3', 'conv4','pool4']
path = 'D://even/delta/'
pool4 = 'D://pool4/delta/'
def alpha_data(num):
    ph = path + str(num) + '.mat'
    pool = pool4 +str(num) +'.mat'
    load_data0 = sio.loadmat(ph)  #
    load_data1 = sio.loadmat(pool)
    load_matrix0 = load_data0[CNN[0]]  #
    load_matrix1 = load_data0[CNN[1]]  #
    load_matrix2 = load_data0[CNN[2]]  #
    load_matrix3 = load_data0[CNN[3]]  #
    load_matrix4 = load_data0[CNN[4]]  #
    load_matrix5 = load_data0[CNN[5]]  #
    load_matrix6 = load_data0[CNN[6]]  #
    x = load_data1['pool4']
    sio.savemat('D:/even/delta/'+str(num)+'.mat', {CNN[0]:load_matrix0, CNN[1]:load_matrix1, CNN[2]:load_matrix2, CNN[3]:load_matrix3,CNN[4]:load_matrix4, CNN[5]:load_matrix5, CNN[6]:load_matrix6, CNN[7]:x})
for i in range(1,15):
	alpha_data(i)