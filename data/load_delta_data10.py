import scipy.io as sio
import numpy as np
from scipy import fftpack

"""
alpha_data(num)表示去第num个经过alpha滤波的数据
return 对应mat数据集的batch
"""
## 本来是想将每个文件的路径放到一个txt文件中，通过列表提取每行赋的路径，但一直报错，'' 与""的问题--已解决
with open("H:/SpaceWork/CNN-LSTM/Theta_path10") as file_object: #在这里选择什么频带数据
    lines = file_object.readlines()  # 从文件中读取每一行，将获取的内容放到list里
delta_path = []
for line in lines:
    delta_path.append(line.strip())  # 将每行地址追加到一个数组里
#

###怎么将mat数据分成单个500*9的数据矩阵，将128*9矩阵放到一个batch里##
def delta_data(num):
    test_batch = []
    load_data0 = sio.loadmat(delta_path[num])  # 鍔犺浇mat鏂囦欢
    load_matrix = load_data0['data2']  # 鎻愬彇鍑鸿鏁版嵁
    shape = load_matrix.shape[0]
    for i in range(0, int(shape/5000)):#存储第一个人的数据，将其作为测试集
        batch = load_matrix[i * 5000:(i + 1) * 5000]  # 取500*9的数据矩阵
        test_batch.append(batch)  # 取得的矩阵追加到list里
    test_batch = np.array(test_batch)
    return test_batch
def build_single_delta(num,channel):
    '''
    :param num : 要提取哪一个人
    :param channel: 要提取的通道数据
    :return: 将该通道的数据转变为频谱的包络输出
    '''
    load_data = sio.loadmat(delta_path[num])
    load_maxtrix = load_data['data2']
    shape = load_maxtrix.shape
    len = shape[0]
    pre_train = load_maxtrix[0:len,channel]
    pre_train = np.array(pre_train)
    pre_train = np.reshape(pre_train,(-1))
    hx = fftpack.hilbert(pre_train)
    envelop = pre_train ** 2 + hx ** 2
    # envelop = np.sqrt(pre_train ** 2 + hx ** 2)
    return envelop
def load_delta_data_conv1(num,channel):
    #path 表示第几个实验者，count表示要处理的通道
    delta_data = build_single_delta(num,channel)
    shape = delta_data.size
    temp = []
    flg = int(shape / 5000)
    for j in range(flg):
        for i in range(0, 4998):
            tem = (delta_data[j * 5000 +i]+delta_data[j * 5000 +i+1]+delta_data[j * 5000 +i+2])/ 3
            temp.append(tem)
        temp.append((delta_data[j * 5000 + 4998]+delta_data[j * 5000 + 4999])/2)# 添加倒数第二个数据
        temp.append(delta_data[j * 5000 + 4999])
    data = np.array(temp)
    return data

def load_delta_data_pool1(num,channel,batchsize):

    #num 表示第几个实验者，channel表示要处理的通道
    # raw_data 5000 return 2500
    delta_data = load_delta_data_conv1(num, channel)

    temp = []
    flg = batchsize
    for j in range(0, flg):
        for i in range(2499):
            tem = (delta_data[j*5000+i*2]+delta_data[j*5000 +i*2+1]+delta_data[j*5000 +i*2+2])/3
            temp.append(tem)
        temp.append((delta_data[j * 5000+4998]+delta_data[j * 5000+4999])/2)
    data = np.array(temp)
    return data
def load_delta_data_conv2(num,channel,batchsize):
    # raw_data 2500 return 2500
    temp = []
    delta_data = load_delta_data_pool1(num,channel,batchsize)
    flg = batchsize
    for j in range(0, flg):
        for i in range(0, 2498):
            tem = (delta_data[j * 2500+i] + delta_data[j * 2500  +i+ 1] + delta_data[j * 2500 +i + 2]) / 3
            temp.append(tem)
        temp.append((delta_data[j * 2500 + 2498] + delta_data[j * 2500 + 2499]) / 2)
        temp.append(delta_data[2499])
    data = np.array(temp)
    return data

def load_delta_data_pool2(num,channel,batchsize):
    #path 表示第几个实验者，count表示要处理的通道
    # raw_data 2500 return 1250
    temp = []
    delta_data = load_delta_data_conv2(num,channel,batchsize)
    flg = batchsize
    for j in range(0, flg):
        for i in range(0, 1249):
            tem = (delta_data[j * 2500 + i * 2] + delta_data[j * 2500 + i * 2 + 1] + delta_data[j * 2500 + i * 2 + 2]) / 3
            temp.append(tem)
        temp.append((delta_data[j * 2500 + 2498] + delta_data[j * 2500 + 2499]) / 2)
    data = np.array(temp)
    return data
def load_delta_data_conv3(num,channel,batchsize):
    #path 表示第几个实验者，count表示要处理的通道
    # raw_data 1250 return 1250
    temp = []
    delta_data = load_delta_data_pool2(num,channel,batchsize)
    flg = batchsize
    for j in range(0, flg):
        for i in range(0,1248):
            tem = (delta_data[j * 1250 +i] + delta_data[j * 1250 + i  + 1] + delta_data[j * 1250 + i  + 2]) / 3
            temp.append(tem)
        temp.append((delta_data[j * 1250 + 1248] + delta_data[j * 1250 + 1249]) / 2)
        temp.append(delta_data[j * 1250 + 1249])
    data = np.array(temp)
    return data
def load_delta_data_pool3(num,channel,batchsize):
    # path 表示第几个实验者，count表示要处理的通道
    # raw_data 1250 return 625
    temp = []
    delta_data = load_delta_data_conv3(num,channel,batchsize)
    flg = batchsize
    for j in range(0, flg):
        for i in range(0, 624):
            tem = (delta_data[j * 1250 + i*2] + delta_data[j * 1250+ i*2 + 1] + delta_data[j * 1250+ i*2 + 2]) / 3
            temp.append(tem)
        temp.append((delta_data[j * 1250 + 1248] + delta_data[j * 1250 + 1249]) / 2)
    data = np.array(temp)
    return data


def load_delta_data_conv4(num,channel,batchsize):
    # raw_data 625 return 625
    temp = []
    delta_data = load_delta_data_pool3(num,channel,batchsize)
    flg = batchsize
    for j in range(0, flg):
        for i in range(0, 623):
            tem = (delta_data[j * 625 + i ] + delta_data[j * 625 + i + 1] + delta_data[j * 625 + i + 2]) / 3
            temp.append(tem)
        temp.append((delta_data[j * 625+623]+delta_data[j * 625+624] )/2)
        temp.append(delta_data[j*625+624])
    data = np.array(temp)
    return data


def load_delta_data_pool4(num,channel,batchsize):
    # raw_data 625 return 313
    temp = []
    delta_data = load_delta_data_conv4(num,channel,batchsize)
    flg = batchsize
    for j in range(0, flg):
        for i in range(0, 312):
            tem = (delta_data[j * 625 + i*2] + delta_data[j * 625 + i*2 + 1] + delta_data[j * 625 + i*2 + 2]) / 3
            temp.append(tem)
        temp.append(delta_data[j * 625 + 624])
    data = np.array(temp)
    return data

# load_delta_data(0,0)




