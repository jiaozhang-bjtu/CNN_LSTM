import scipy.io as sio
import numpy as np

## 本来是想将每个文件的路径放到一个txt文件中，通过列表提取每行赋的路径，但一直报错，'' 与""的问题--已解决
from scipy import fftpack

with open("H:/SpaceWork/EEG_Work/path.txt") as file_object:
    lines = file_object.readlines()  # 从文件中读取每一行，将获取的内容放到list里
mat_path = []
for line in lines:
    mat_path.append(line.strip())  # 将每行地址追加到一个数组里
# print("ok")
with open("H:/SpaceWork/EEG_Work/raw_path10.txt") as file_object:
    line_alpha = file_object.readlines()  # 从文件中读取每一行，将获取的内容放到list里
raw_path=[]
for alpha in line_alpha:
    raw_path.append(alpha.strip())
with open("H:/SpaceWork/EEG_Work/lable.txt") as file_object:
    lines_lable = file_object.readlines()  # 从文件中读取每一行，将获取的内容放到list里
lable_value = []
for line in lines_lable:
    lable_value.append(int(line.strip()))  # 将每行标签值追加到一个数组里

mat_dictionary = {}  # 构造字典
for i in range(0, 15):
    mat_dictionary[mat_path[i]] = lable_value[i]  # 存入每个对应的值

def get_lable(load_path):  # 根据路径得到标签
    return mat_dictionary[load_path]

###怎么将mat数据分成单个500*9的数据矩阵，将128*9矩阵放到一个batch里##
def raw_test_batch10(num):
    #取num位实验者数据
    test_batch = []
    test_label = []
    load_data0 = sio.loadmat(mat_path[num])  # 鍔犺浇mat鏂囦欢
    load_matrix = load_data0['data2']  # 鎻愬彇鍑鸿鏁版嵁
    shape = load_matrix.shape[0]
    for i in range(0, int(shape/5000)):#存储第一个人的数据，将其作为测试集
        batch = load_matrix[i * 5000:(i + 1) * 5000]  # 取500*9的数据矩阵
        label = get_lable(mat_path[num])
        test_batch.append(batch)  # 取得的矩阵追加到list里
        test_label.append(label)
    test_batch = np.array(test_batch)
    test_label = np.array(test_label)
    return test_batch,test_label

def build_single_raw_data(num, channel):
    '''
    :param num : 要提取哪一个人
    :param channel: 要提取的通道数据
    :return: 将该通道的数据转变为频谱的包络输出
    '''
    load_data = sio.loadmat(raw_path[num])
    load_maxtrix = load_data['data']
    shape = load_maxtrix.shape
    len = shape[0]
    pre_train = load_maxtrix[0:len,channel]
    pre_train = np.array(pre_train)
    pre_train = np.reshape(pre_train,(-1))
    hx = fftpack.hilbert(pre_train)
    envelop = np.sqrt(pre_train ** 2 + hx ** 2)
    return envelop
def load_raw_data_conv1(path,count):
    #path 表示第几个实验者，count表示要处理的通道
    raw_data = build_single_raw_data(path, count)
    shape = raw_data.size
    temp = []
    flg = int(shape / 5000)
    for j in range(flg):
        for i in range(0, 4998):
            tem = (raw_data[j * 5000 +i]+raw_data[j * 5000 +i+1]+raw_data[j * 5000 +i+2])/ 3
            temp.append(tem)
        temp.append((raw_data[j * 5000 + 4998]+raw_data[j * 5000 + 4999])/2)# 添加倒数第二个数据
        temp.append(raw_data[j * 5000 + 4999])
    data = np.array(temp)
    return data

def load_raw_data_pool1(path,count):
    #conv2也可以调用这个框架
    #path 表示第几个实验者，count表示要处理的通道
    temp = []
    raw_data = build_single_raw_data(path, count)
    shape = raw_data.size
    flg = int(shape / 5000)
    for j in range(0, flg):
        for i in range(0, 2498):
            tem = (raw_data[j * 5000 +i*2]+raw_data[j * 5000 +i*2+1]+raw_data[j * 5000 +i*2+2]+raw_data[j * 5000 +i*2+3]+raw_data[j * 5000 +i*2+4])/5
            temp.append(tem)
        temp.append((raw_data[j * 5000+4996]+raw_data[j * 5000+4997]+raw_data[j * 5000+4998]+raw_data[j * 5000+4999])/4)
        temp.append((raw_data[j * 5000 + 4998] +raw_data[j * 5000 + 4999]) / 2)
    data =np.array(temp)
    return data
def load_raw_data_conv2(path,count):
    #path 表示第几个实验者，count表示要处理的通道
    temp = []
    raw_data = build_single_raw_data(path, count)
    shape = raw_data.size
    flg = int(shape / 5000)
    for j in range(0, flg):
        for i in range(0, 2496):
            tem = (raw_data[j * 5000 +i*2]+raw_data[j * 5000 +i*2+1]+raw_data[j * 5000 +i*2+2]+raw_data[j * 5000 +i*2+3]+raw_data[j * 5000 +i*2+4]+
                   raw_data[j * 5000 +i*2+5]+raw_data[j * 5000 +i*2+6]+raw_data[j * 5000 +i*2+7]+raw_data[j * 5000 +i*2+8])/9
            temp.append(tem)
        temp.append((raw_data[j * 5000 + 4992] + raw_data[j * 5000 + 4993] + raw_data[j * 5000 + 4994] +raw_data[j * 5000 + 4995]+
                     raw_data[j * 5000 + 4996] + raw_data[j * 5000 + 4997] + raw_data[j * 5000 + 4998] +raw_data[j * 5000 + 4999]) / 8)
        temp.append((raw_data[j * 5000 + 4994] +raw_data[j * 5000 + 4995]+raw_data[j * 5000 + 4996] + raw_data[j * 5000 + 4997] + raw_data[j * 5000 + 4998] +raw_data[j * 5000 + 4999]) / 6)
        temp.append((raw_data[j * 5000 + 4996] +raw_data[j * 5000 + 4997] + raw_data[j * 5000 + 4998] + raw_data[j * 5000 + 4999]) / 4)
        temp.append((raw_data[j * 5000 + 4998] +raw_data[j * 5000 + 4999]) / 2)
    data =np.array(temp)
    return data
def load_raw_data_pool2(path,count):
    #path 表示第几个实验者，count表示要处理的通道
    temp = []
    raw_data = build_single_raw_data(path, count)
    shape = raw_data.size
    flg = int(shape / 5000)
    for j in range(0, flg):
        for i in range(0, 1247):
            tem = (raw_data[j * 5000 + i * 4] + raw_data[j * 5000 + i * 4 + 1] + raw_data[j * 5000 + i * 4 + 2] + raw_data[j * 5000 + i * 4 + 3] + raw_data[j * 5000 + i * 4 + 4] +
                       raw_data[j * 5000 + i * 4 + 5] + raw_data[j * 5000 + i * 4 + 6] + raw_data[j * 5000 + i * 4 + 7] + raw_data[j * 5000 + i * 4 + 8]+raw_data[j * 5000 + i * 4+9] +
                       raw_data[j * 5000 + i * 4 + 10] + raw_data[j * 5000 + i * 4 + 11] + raw_data[j * 5000 + i * 4 + 12]) / 13
            temp.append(tem)
        temp.append((raw_data[j * 5000 + 4988] + raw_data[j * 5000 + 4989] + raw_data[j * 5000 + 4990] +
                         raw_data[j * 5000 + 4991] +raw_data[j * 5000 + 4992] + raw_data[j * 5000 + 4993] + raw_data[j * 5000 + 4994] +
                         raw_data[j * 5000 + 4995]+ raw_data[j * 5000 + 4996] +raw_data[j * 5000 + 4997] + raw_data[j * 5000 + 4998] + raw_data[j * 5000 + 4999]) / 12)
        temp.append((raw_data[j * 5000 + 4992] + raw_data[j * 5000 + 4993] +raw_data[j * 5000 + 4994] + raw_data[j * 5000 + 4995] + raw_data[j * 5000 + 4996] +
                         raw_data[j * 5000 + 4997] + raw_data[j * 5000 + 4998] + raw_data[j * 5000 + 4999]) / 8)
        temp.append((raw_data[j * 5000 + 4996] + raw_data[j * 5000 + 4997] + raw_data[j * 5000 + 4998] +raw_data[j * 5000 + 4999]) / 4)
    data = np.array(temp)
    return data
def load_raw_data_conv3(path,count):
    #path 表示第几个实验者，count表示要处理的通道
    temp = []
    raw_data = build_single_raw_data(path, count)
    shape = raw_data.size
    flg = int(shape / 5000)
    for j in range(0, flg):
        for i in range(0, 1245):
            tem = 0
            for k in range(21):
                tem += raw_data[j*5000+i*4+k]
            tem = tem/21
            temp.append(tem)
        temp.append((raw_data[j * 5000 + 4980] + raw_data[j * 5000 + 4981] + raw_data[j * 5000 + 4982] +raw_data[j * 5000 + 4983] + raw_data[j * 5000 + 4984] + raw_data[j * 5000 + 4985] + raw_data[j * 5000 + 4986] +
                     raw_data[j * 5000 + 4987] +raw_data[j * 5000 + 4988] + raw_data[j * 5000 + 4989] + raw_data[j * 5000 + 4990] + raw_data[j * 5000 + 4991] + raw_data[j * 5000 + 4992] + raw_data[j * 5000 + 4993] +
                    raw_data[j * 5000 + 4994] +raw_data[j * 5000 + 4995] + raw_data[j * 5000 + 4996] + raw_data[j * 5000 + 4997] +
                     raw_data[j * 5000 + 4998] + raw_data[j * 5000 + 4999]) / 20)
        temp.append((raw_data[j * 5000 + 4984] + raw_data[j * 5000 + 4985] + raw_data[j * 5000 + 4986] +
                     raw_data[j * 5000 + 4987] +raw_data[j * 5000 + 4988] + raw_data[j * 5000 + 4989] + raw_data[j * 5000 + 4990] + raw_data[j * 5000 + 4991] + raw_data[j * 5000 + 4992] + raw_data[j * 5000 + 4993] +
                    raw_data[j * 5000 + 4994] +raw_data[j * 5000 + 4995] + raw_data[j * 5000 + 4996] + raw_data[j * 5000 + 4997] +
                     raw_data[j * 5000 + 4998] + raw_data[j * 5000 + 4999]) / 16)
        temp.append((raw_data[j * 5000 + 4988] + raw_data[j * 5000 + 4989] +raw_data[j * 5000 + 4990] + raw_data[j * 5000 + 4991] + raw_data[j * 5000 + 4992] +
                     raw_data[j * 5000 + 4993] + raw_data[j * 5000 + 4994] + raw_data[j * 5000 + 4995] + raw_data[j * 5000 + 4996] +
                     raw_data[j * 5000 + 4997] +raw_data[j * 5000 + 4998] + raw_data[j * 5000 + 4999]) / 12)
        temp.append((raw_data[j * 5000 + 4992] + raw_data[j * 5000 + 4993] + raw_data[j * 5000 + 4994] +
                     raw_data[j * 5000 + 4995] + raw_data[j * 5000 + 4996] + raw_data[j * 5000 + 4997] + raw_data[j * 5000 + 4998] + raw_data[j * 5000 + 4999]) / 8)
        temp.append((raw_data[j * 5000 + 4996] + raw_data[j * 5000 + 4997] + raw_data[j * 5000 + 4998] +
                     raw_data[j * 5000 + 4999]) / 4)
    data = np.array(temp)
    return data
def load_raw_data_pool3(path,count):
    # path 表示第几个实验者，count表示要处理的通道
    temp = []
    raw_data = build_single_raw_data(path, count)
    shape = raw_data.size
    flg = int(shape / 5000)
    for j in range(0, flg):
        for i in range(0, 622):
            tem = 0
            for k in range(29):
                tem += raw_data[j * 5000 + i * 8 + k]
            tem = tem / 29
            temp.append(tem)
        temp.append((raw_data[j * 5000 + 4976] + raw_data[j * 5000 + 4977] + raw_data[j * 5000 + 4978] +
                     raw_data[j * 5000 + 4979] +raw_data[j * 5000 + 4980] + raw_data[j * 5000 + 4981] + raw_data[j * 5000 + 4982] +
                     raw_data[j * 5000 + 4983] + raw_data[j * 5000 + 4984] + raw_data[j * 5000 + 4985] +
                     raw_data[j * 5000 + 4986] + raw_data[j * 5000 + 4987] + raw_data[j * 5000 + 4988] + raw_data[j * 5000 + 4989] +
                     raw_data[j * 5000 + 4990] + raw_data[j * 5000 + 4991] + raw_data[j * 5000 + 4992] +
                     raw_data[j * 5000 + 4993] +raw_data[j * 5000 + 4994] + raw_data[j * 5000 + 4995] + raw_data[j * 5000 + 4996] +
                     raw_data[j * 5000 + 4997] +raw_data[j * 5000 + 4998] + raw_data[j * 5000 + 4999]) / 24)
        temp.append((raw_data[j * 5000 + 4984] + raw_data[j * 5000 + 4985] +
                     raw_data[j * 5000 + 4986] + raw_data[j * 5000 + 4987] + raw_data[j * 5000 + 4988] +
                     raw_data[j * 5000 + 4989] + raw_data[j * 5000 + 4990] + raw_data[j * 5000 + 4991] + raw_data[j * 5000 + 4992] +
                     raw_data[j * 5000 + 4993] + raw_data[j * 5000 + 4994] + raw_data[j * 5000 + 4995] +
                     raw_data[j * 5000 + 4996] +raw_data[j * 5000 + 4997] + raw_data[j * 5000 + 4998] + raw_data[j * 5000 + 4999]) / 16)

        temp.append((raw_data[j * 5000 + 4992] + raw_data[j * 5000 + 4993] + raw_data[j * 5000 + 4994] +
                     raw_data[j * 5000 + 4995] + raw_data[j * 5000 + 4996] + raw_data[j * 5000 + 4997] +
                     raw_data[j * 5000 + 4998] + raw_data[j * 5000 + 4999]) / 8)
    data = np.array(temp)
    return data


def load_raw_data_conv4(path, count):
    # path 表示第几个实验者，count表示要处理的通道
    temp = []
    raw_data = build_single_raw_data(path, count)
    shape = raw_data.size
    flg = int(shape / 5000)
    for j in range(0, flg):
        for i in range(0, 620):
            tem = 0
            for k in range(45):
                tem += raw_data[j * 5000 + i * 8 + k]
            tem = tem / 45
            temp.append(tem)
        temp.append((raw_data[j * 5000 + 4960] + raw_data[j * 5000 + 4961] + raw_data[j * 5000 + 4962] +
                     raw_data[j * 5000 + 4963] + raw_data[j * 5000 + 4964] + raw_data[j * 5000 + 4965] +
                     raw_data[j * 5000 + 4966] + raw_data[j * 5000 + 4967] + raw_data[j * 5000 + 4968] + raw_data[j * 5000 + 4969] + raw_data[j * 5000 + 4970] +
                     raw_data[j * 5000 + 4971] + raw_data[j * 5000 + 4972] + raw_data[j * 5000 + 4973] +
                     raw_data[j * 5000 + 4974] + raw_data[j * 5000 + 4975] + raw_data[j * 5000 + 4976] +
                     raw_data[j * 5000 + 4977] + raw_data[j * 5000 + 4978] +
                     raw_data[j * 5000 + 4979] + raw_data[j * 5000 + 4980] + raw_data[j * 5000 + 4981] +
                     raw_data[j * 5000 + 4982] +raw_data[j * 5000 + 4983] + raw_data[j * 5000 + 4984] + raw_data[j * 5000 + 4985] +
                     raw_data[j * 5000 + 4986] + raw_data[j * 5000 + 4987] + raw_data[j * 5000 + 4988] +
                     raw_data[j * 5000 + 4989] + raw_data[j * 5000 + 4990] + raw_data[j * 5000 + 4991] +
                     raw_data[j * 5000 + 4992] +
                     raw_data[j * 5000 + 4993] + raw_data[j * 5000 + 4994] + raw_data[j * 5000 + 4995] +
                     raw_data[j * 5000 + 4996] + raw_data[j * 5000 + 4997] + raw_data[j * 5000 + 4998] +
                     raw_data[j * 5000 + 4999]) / 40)
        temp.append((raw_data[j * 5000 + 4968] + raw_data[j * 5000 + 4969] + raw_data[j * 5000 + 4970] +raw_data[j * 5000 + 4971] + raw_data[j * 5000 + 4972] + raw_data[j * 5000 + 4973] +
                     raw_data[j * 5000 + 4974] +raw_data[j * 5000 + 4975] +raw_data[j * 5000 + 4976] + raw_data[j * 5000 + 4977] + raw_data[j * 5000 + 4978] +
                     raw_data[j * 5000 + 4979] + raw_data[j * 5000 + 4980] + raw_data[j * 5000 + 4981] + raw_data[j * 5000 + 4982] +
                     raw_data[j * 5000 + 4983] + raw_data[j * 5000 + 4984] + raw_data[j * 5000 + 4985] +raw_data[j * 5000 + 4986] + raw_data[j * 5000 + 4987] + raw_data[j * 5000 + 4988] +
                     raw_data[j * 5000 + 4989] +raw_data[j * 5000 + 4990] + raw_data[j * 5000 + 4991] + raw_data[j * 5000 + 4992] +
                     raw_data[j * 5000 + 4993] + raw_data[j * 5000 + 4994] + raw_data[j * 5000 + 4995] +raw_data[j * 5000 + 4996] +raw_data[j * 5000 + 4997] + raw_data[j * 5000 + 4998] + raw_data[j * 5000 + 4999]) / 32)
        temp.append((raw_data[j * 5000 + 4976] + raw_data[j * 5000 + 4977] + raw_data[j * 5000 + 4978] +
                     raw_data[j * 5000 + 4979] + raw_data[j * 5000 + 4980] + raw_data[j * 5000 + 4981] +
                     raw_data[j * 5000 + 4982] +
                     raw_data[j * 5000 + 4983] + raw_data[j * 5000 + 4984] + raw_data[j * 5000 + 4985] +
                     raw_data[j * 5000 + 4986] + raw_data[j * 5000 + 4987] + raw_data[j * 5000 + 4988] +
                     raw_data[j * 5000 + 4989] +
                     raw_data[j * 5000 + 4990] + raw_data[j * 5000 + 4991] + raw_data[j * 5000 + 4992] +
                     raw_data[j * 5000 + 4993] + raw_data[j * 5000 + 4994] + raw_data[j * 5000 + 4995] +
                     raw_data[j * 5000 + 4996] +
                     raw_data[j * 5000 + 4997] + raw_data[j * 5000 + 4998] + raw_data[j * 5000 + 4999]) / 24)
        temp.append((raw_data[j * 5000 + 4984] + raw_data[j * 5000 + 4985] +
                     raw_data[j * 5000 + 4986] + raw_data[j * 5000 + 4987] + raw_data[j * 5000 + 4988] +
                     raw_data[j * 5000 + 4989] + raw_data[j * 5000 + 4990] + raw_data[j * 5000 + 4991] +
                     raw_data[j * 5000 + 4992] +
                     raw_data[j * 5000 + 4993] + raw_data[j * 5000 + 4994] + raw_data[j * 5000 + 4995] +
                     raw_data[j * 5000 + 4996] + raw_data[j * 5000 + 4997] + raw_data[j * 5000 + 4998] +
                     raw_data[j * 5000 + 4999]) / 16)

        temp.append((raw_data[j * 5000 + 4992] + raw_data[j * 5000 + 4993] + raw_data[j * 5000 + 4994] +
                     raw_data[j * 5000 + 4995] + raw_data[j * 5000 + 4996] + raw_data[j * 5000 + 4997] +
                     raw_data[j * 5000 + 4998] + raw_data[j * 5000 + 4999]) / 8)
    data = np.array(temp)
    return data


def load_raw_data_pool4(path,count):
    # path 表示第几个实验者，count表示要处理的通道
    temp = []
    raw_data = build_single_raw_data(path, count)
    shape = raw_data.size
    flg = int(shape / 5000)
    for j in range(0, flg):
        for i in range(0, 309):
            tem = 0
            for k in range(61):
                tem += raw_data[j * 5000 + i * 16 + k]
            tem = tem / 61
            temp.append(tem)
        temp.append((raw_data[j * 5000 + 4944] + raw_data[j * 5000 + 4945] + raw_data[j * 5000 + 4946] +
                         raw_data[j * 5000 + 4947] + raw_data[j * 5000 + 4948] + raw_data[j * 5000 + 4949] +
                         raw_data[j * 5000 + 4950] + raw_data[j * 5000 + 4951] + raw_data[j * 5000 + 4952] +
                         raw_data[j * 5000 + 4953] + raw_data[j * 5000 + 4954] +raw_data[j * 5000 + 4955] + raw_data[j * 5000 + 4956] +raw_data[j * 5000 + 4957] + raw_data[j * 5000 + 4958]
                         +raw_data[j * 5000 + 4959] + raw_data[j * 5000 + 4960] + raw_data[j * 5000 + 4961] +raw_data[j * 5000 + 4962] +
                         raw_data[j * 5000 + 4963] + raw_data[j * 5000 + 4964] + raw_data[j * 5000 + 4965] +
                         raw_data[j * 5000 + 4966] + raw_data[j * 5000 + 4967] + raw_data[j * 5000 + 4968] +
                         raw_data[j * 5000 + 4969] + raw_data[j * 5000 + 4970] +
                         raw_data[j * 5000 + 4971] + raw_data[j * 5000 + 4972] + raw_data[j * 5000 + 4973] +
                         raw_data[j * 5000 + 4974] + raw_data[j * 5000 + 4975] + raw_data[j * 5000 + 4976] +
                         raw_data[j * 5000 + 4977] + raw_data[j * 5000 + 4978] +
                         raw_data[j * 5000 + 4979] + raw_data[j * 5000 + 4980] + raw_data[j * 5000 + 4981] +
                         raw_data[j * 5000 + 4982] + raw_data[j * 5000 + 4983] + raw_data[j * 5000 + 4984] +
                         raw_data[j * 5000 + 4985] +
                         raw_data[j * 5000 + 4986] + raw_data[j * 5000 + 4987] + raw_data[j * 5000 + 4988] +
                         raw_data[j * 5000 + 4989] + raw_data[j * 5000 + 4990] + raw_data[j * 5000 + 4991] +
                         raw_data[j * 5000 + 4992] +
                         raw_data[j * 5000 + 4993] + raw_data[j * 5000 + 4994] + raw_data[j * 5000 + 4995] +
                         raw_data[j * 5000 + 4996] + raw_data[j * 5000 + 4997] + raw_data[j * 5000 + 4998] +
                         raw_data[j * 5000 + 4999]) / 56)
        temp.append((raw_data[j * 5000 + 4960] + raw_data[j * 5000 + 4961] + raw_data[j * 5000 + 4962] +
                     raw_data[j * 5000 + 4963] + raw_data[j * 5000 + 4964] + raw_data[j * 5000 + 4965] +
                     raw_data[j * 5000 + 4966] + raw_data[j * 5000 + 4967] + raw_data[j * 5000 + 4968] +
                     raw_data[j * 5000 + 4969] + raw_data[j * 5000 + 4970] +
                     raw_data[j * 5000 + 4971] + raw_data[j * 5000 + 4972] + raw_data[j * 5000 + 4973] +
                     raw_data[j * 5000 + 4974] + raw_data[j * 5000 + 4975] + raw_data[j * 5000 + 4976] +
                     raw_data[j * 5000 + 4977] + raw_data[j * 5000 + 4978] +
                     raw_data[j * 5000 + 4979] + raw_data[j * 5000 + 4980] + raw_data[j * 5000 + 4981] +
                     raw_data[j * 5000 + 4982] + raw_data[j * 5000 + 4983] + raw_data[j * 5000 + 4984] +
                     raw_data[j * 5000 + 4985] +
                     raw_data[j * 5000 + 4986] + raw_data[j * 5000 + 4987] + raw_data[j * 5000 + 4988] +
                     raw_data[j * 5000 + 4989] + raw_data[j * 5000 + 4990] + raw_data[j * 5000 + 4991] +
                     raw_data[j * 5000 + 4992] +
                     raw_data[j * 5000 + 4993] + raw_data[j * 5000 + 4994] + raw_data[j * 5000 + 4995] +
                     raw_data[j * 5000 + 4996] + raw_data[j * 5000 + 4997] + raw_data[j * 5000 + 4998] +
                     raw_data[j * 5000 + 4999]) / 40)

        temp.append((raw_data[j * 5000 + 4976] + raw_data[j * 5000 + 4977] + raw_data[j * 5000 + 4978] +
                     raw_data[j * 5000 + 4979] + raw_data[j * 5000 + 4980] + raw_data[j * 5000 + 4981] +
                     raw_data[j * 5000 + 4982] +
                     raw_data[j * 5000 + 4983] + raw_data[j * 5000 + 4984] + raw_data[j * 5000 + 4985] +
                     raw_data[j * 5000 + 4986] + raw_data[j * 5000 + 4987] + raw_data[j * 5000 + 4988] +
                     raw_data[j * 5000 + 4989] +
                     raw_data[j * 5000 + 4990] + raw_data[j * 5000 + 4991] + raw_data[j * 5000 + 4992] +
                     raw_data[j * 5000 + 4993] + raw_data[j * 5000 + 4994] + raw_data[j * 5000 + 4995] +
                     raw_data[j * 5000 + 4996] +
                     raw_data[j * 5000 + 4997] + raw_data[j * 5000 + 4998] + raw_data[j * 5000 + 4999]) / 24)
        temp.append((raw_data[j * 5000 + 4992] + raw_data[j * 5000 + 4993] + raw_data[j * 5000 + 4994] +
                     raw_data[j * 5000 + 4995] + raw_data[j * 5000 + 4996] + raw_data[j * 5000 + 4997] +
                     raw_data[j * 5000 + 4998] + raw_data[j * 5000 + 4999]) / 8)
    data = np.array(temp)
    return data

