import scipy.io as sio
import numpy as np
import tensorflow as tf

"""
这个脚本是用来获取整个数据集的产生的500*9 矩阵大小的 batch
build_batch()
return (123953,500,9)的样本集和（123953）便签集
"""
with open("H:/SpaceWork/EEG_Work/raw_path10.txt") as file_object:
	lines = file_object.readlines()  #
mat_path = []
for line in lines:
	mat_path.append(line.strip())  # 灏嗘瘡琛屽湴鍧€杩藉姞鍒颁竴涓暟缁勯噷
# print("ok")
with open("H:/SpaceWork/EEG_Work/lable.txt") as file_object:
	lines_lable = file_object.readlines()  # 浠庢枃浠朵腑璇诲彇姣忎竴琛岋紝灏嗚幏鍙栫殑鍐呭鏀惧埌list閲?
lable_value = []
for line in lines_lable:
	lable_value.append(int(line.strip()))  # 灏嗘瘡琛屾爣绛惧€艰拷鍔犲埌涓€涓暟缁勯噷

mat_dictionary = {}  # 鏋勯€犲瓧鍏?
for i in range(0, 15):
	mat_dictionary[mat_path[i]] = lable_value[i]


def get_lable(load_path):
	return mat_dictionary[load_path]


def tailor_train_batch():
	load_data0 = sio.loadmat(mat_path[0])  # 鍔犺浇mat鏂囦欢
	load_matrix = load_data0['data2']  # 鎻愬彇鍑鸿鏁版嵁
	load_data1 = sio.loadmat(mat_path[1])
	load_matrix1 = load_data1['data2']
	load_data2 = sio.loadmat(mat_path[2])
	load_matrix2 = load_data2['data2']
	load_data3 = sio.loadmat(mat_path[3])
	load_matrix3 = load_data3['data2']
	load_data4 = sio.loadmat(mat_path[4])
	load_matrix4 = load_data4['data2']
	load_data5 = sio.loadmat(mat_path[5])
	load_matrix5 = load_data5['data2']
	load_data6 = sio.loadmat(mat_path[6])
	load_matrix6 = load_data6['data2']
	load_data7 = sio.loadmat(mat_path[7])
	load_matrix7 = load_data7['data2']
	load_data8 = sio.loadmat(mat_path[8])
	load_matrix8 = load_data8['data2']
	load_data9 = sio.loadmat(mat_path[9])
	load_matrix9 = load_data9['data2']
	load_data10 = sio.loadmat(mat_path[10])
	load_matrix10 = load_data10['data2']
	load_data11 = sio.loadmat(mat_path[11])
	load_matrix11 = load_data11['data2']
	load_data12 = sio.loadmat(mat_path[12])
	load_matrix12 = load_data12['data2']
	load_data13 = sio.loadmat(mat_path[13])
	load_matrix13 = load_data13['data2']
	load_data14 = sio.loadmat(mat_path[14])
	load_matrix14 = load_data14['data2']
	shape = []
	for i in range(15):  # 取mat数据的行数,影响速度，直接套用上面已经加载的load_mat:下一步尝试去做
		load = sio.loadmat(mat_path[i])
		load_shape = load['data2']
		l_d = load_shape.shape
		shape.append(l_d[0])
	train_batch = []
	train_label = []
	for i in range(15):
		lo = sio.loadmat(mat_path[i])
		load = lo['data2']
		# print(shape[i]) #打印出每个的shape
		for j in range(50, int(shape[i] / 5000) - 1):
			batchx = load[j * 5000:(j + 1) * 5000]  # 鍙?28*9鐨勬暟鎹煩闃?
			batch = np.reshape(batchx, (5000, 9))
			label = get_lable(mat_path[i])
			train_batch.append(batch)
			train_label.append(label)
	# 对10号样本进行裁剪增加1倍数据集，平衡数据
	for j in range(int(shape[9] / 10000) - 1):
		batchx = load_matrix9[j * 5000 + 1370:(j + 1) * 5000 + 1370]
		batch = np.reshape(batchx, (5000, 9))
		label = get_lable(mat_path[9])
		batchz = load_matrix9[j * 5000 + 2490:(j + 1) * 5000 + 2490]
		batch1 = np.reshape(batchz, (5000, 9))
		train_batch.append(batch)
		train_label.append(label)
		train_batch.append(batch1)
		train_label.append(label)
	# 对11号样本进行裁剪增加2倍数据集，平衡数据
	for j in range(int(shape[10] / 10000) - 1):
		for i in range(1, 5):
			batchx = load_matrix10[j * 5000 + i * 790:(j + 1) * 5000 + i * 790]
			batch = np.reshape(batchx, (5000, 9))
			label = get_lable(mat_path[10])
			train_batch.append(batch)
			train_label.append(label)
		#  对13号样本进行裁剪增加8倍数据集，平衡数据
	for j in range(int(shape[12] / 10000) - 1):
		for i in range(1, 17):
			batchx = load_matrix12[j * 5000 + i * 230:(j + 1) * 5000 + i * 230]
			batch = np.reshape(batchx, (5000, 9))
			label = get_lable(mat_path[12])
			train_batch.append(batch)
			train_label.append(label)

		# 对15号样本进行裁剪增加6倍数据集，平衡数据
	for j in range(int(shape[14] / 10000) - 1):
		for i in range(1, 13):
			batchx = load_matrix14[j * 5000 + i * 290:(j + 1) * 5000 + i * 290]
			batch = np.reshape(batchx, (5000, 9))
			label = get_lable(mat_path[14])
			train_batch.append(batch)
			train_label.append(label)
	train_batch = np.array(train_batch)
	train_label = np.array(train_label)
	state = np.random.get_state()  # 打乱数据
	np.random.shuffle(train_batch)
	np.random.set_state(state)
	np.random.shuffle(train_label)

	# print(np.shape(train_batch)[1])
	# print(np.shape(train_batch)[2])
	return train_batch, train_label


#    return (train_batch, train_label),(test_batch,test_lable)


"""
# 结构化存储数据，发现存储的数据量太大了80G，故放弃了
tfrecords_filename = 'D:/EEG_Data/output.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename) # 创建.tfrecord文件，准备写入
for i in range(x.shape[0]):
    # EEG_raw = x[i]  # 取一个batch
    EEG_raw = x[i].tostring()
    feature = {
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[y[i]])),
        'EEG_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[EEG_raw]))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
writer.close()
"""
# #
# # b = one_hot(b)
# # print(b)
# # a = np.reshape(a,[100,500,9,1])
# # print(a)
# x, y = tailor_train_batch()
# print(np.shape(x))
#
# a, b = DataSet(x,y,100).next_batch(100)
# test = np.reshape(a,(100,500,9,1))
# # test1 = np.reshape(test,[100,500,9,1])
# print(np.shape(test))
# print(x)
