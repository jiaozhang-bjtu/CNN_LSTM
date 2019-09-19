import os
import tensorflow as tf
import numpy as np

from DATA.Tailor_Train_Data10 import tailor_train_batch
from DATASET.DataSet import DataSet

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def one_hot(labels, n_class=36):
	""" One-hot encoding """
	expansion = np.eye(n_class)
	y = expansion[:, labels - 1].T
	assert y.shape[1] == n_class, "Wrong number of labels!"
	return y


def standardize(train):
	""" Standardize data """
	# Standardize data
	X_train = (train - np.mean(train, axis=0)[None, :, :]) / np.std(train, axis=0)[None, :, :]
	return X_train


def batch_norm(x, train, scope='bn'):
	with tf.variable_scope(scope):
		n_out = x.get_shape()[-1].value
		beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
		gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
		tf.add_to_collection('biases', beta)
		tf.add_to_collection('weights', gamma)
		batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
		ema = tf.train.ExponentialMovingAverage(decay=0.99)

	def mean_var_with_update():
		ema_apply_op = ema.apply([batch_mean, batch_var])
		with tf.control_dependencies([ema_apply_op]):
			return tf.identity(batch_mean), tf.identity(batch_var)

	# mean, var = control_flow_ops.cond(phase_train,
	# mean, var = control_flow_ops.cond(phase_train,
	#   mean_var_with_update,
	#   lambda: (ema.average(batch_mean), ema.average(batch_var)))
	mean, var = mean_var_with_update()
	normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
	return normed


def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
	""" Batch normalization on convolutional maps and beyond...
	Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

	Args:
		inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
		is_training:   boolean tf.Varialbe, true indicates training phase
		scope:         string, variable scope
		moments_dims:  a list of ints, indicating dimensions for moments calculation
		bn_decay:      float or float tensor variable, controling moving average weight
	Return:
		normed:        batch-normalized maps
	"""
	with tf.variable_scope(scope) as sc:
		num_channels = inputs.get_shape()[-1].value
		beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
		                   name='beta', trainable=True)
		gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
		                    name='gamma', trainable=True)
		batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
		decay = bn_decay if bn_decay is not None else 0.9
		ema = tf.train.ExponentialMovingAverage(decay=decay)

		# Operator that maintains moving averages of variables.

		# Update moving average and return current batch's avg and var.
		def mean_var_with_update():
			ema_apply_op = tf.cond(is_training,
			                       lambda: ema.apply([batch_mean, batch_var]),
			                       lambda: tf.no_op())
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		# ema.average returns the Variable holding the average of var.
		mean, var = tf.cond(is_training,
		                    mean_var_with_update,
		                    lambda: (ema.average(batch_mean), ema.average(batch_var)))
		normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
	return normed


def batch_norm_for_fc(inputs, is_training, bn_decay, scope):
	""" Batch normalization on FC data.

	Args:
		inputs:      Tensor, 2D BxC input
		is_training: boolean tf.Varialbe, true indicates training phase
		bn_decay:    float or float tensor variable, controling moving average weight
		scope:       string, variable scope
	Return:
		normed:      batch-normalized maps
	"""
	return batch_norm_template(inputs, is_training, scope, [0, ], bn_decay)


def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope):
	""" Batch normalization on 2D convolutional maps.
	Args:
		inputs:      Tensor, 4D BHWC input maps
		is_training: boolean tf.Varialbe, true indicates training phase
		bn_decay:    float or float tensor variable, controling moving average weight
		scope:       string, variable scope
	Return:
		normed:      batch-normalized maps
	"""
	return batch_norm_template(inputs, is_training, scope, [0, 1, 2], bn_decay)


def BaseCNN(input_tensor, train, regularizer):
	# 卷积网第一层架构 输入为5000*9*16 的矩阵
	with tf.variable_scope('layer1-conv'):
		conv1_weights = tf.get_variable("weight", [3, 3, 1, 16],
		                                initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv1_biases = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0.0))
		conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME', name='conv1')
		res1 = tf.nn.bias_add(conv1, conv1_biases)
		bn1 = batch_norm_for_conv2d(res1, train, bn_decay, scope='BN')
		elu1 = tf.nn.elu(bn1)
	with tf.variable_scope('layer1-pool'):
		pool1 = tf.nn.max_pool(elu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
	# 卷积网第二层架构输入为2500*5*32
	with tf.variable_scope('layer2-conv'):
		conv2_weights = tf.get_variable("weight", [3, 3, 16, 32],
		                                initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv2_biases = tf.get_variable('biases', [32], initializer=tf.constant_initializer(0.0))
		conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME', name='conv2')
		res2 = tf.nn.bias_add(conv2, conv2_biases)
		bn2 = batch_norm_for_conv2d(res2, train, bn_decay, scope='BN')
		elu2 = tf.nn.elu(bn2)

	with tf.variable_scope('layer2-pool'):
		pool2 = tf.nn.max_pool(elu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
	# 卷积网第二层架构输入为1250*3*64
	with tf.variable_scope('layer3-conv'):
		conv3_weights = tf.get_variable("weight", [3, 3, 32, 64],
		                                initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv3_biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0))
		conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME', name='conv3')
		res3 = tf.nn.bias_add(conv3, conv3_biases)
		bn3 = batch_norm_for_conv2d(res3, train, bn_decay, scope='BN')
		elu3 = tf.nn.elu(bn3)
	with tf.variable_scope('layer3-pool'):
		pool3 = tf.nn.max_pool(elu3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
	# 卷积网第二层架构输入为625*2*128
	with tf.variable_scope('layer4-conv'):
		conv4_weights = tf.get_variable("weight", [3, 3, 64, 128],
		                                initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv4_biases = tf.get_variable('biases', [128], initializer=tf.constant_initializer(0.0))
		conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME', name='conv4')
		res4 = tf.nn.bias_add(conv4, conv4_biases)
		bn4 = batch_norm_for_conv2d(res4, train, bn_decay, scope='BN')
		elu4 = tf.nn.elu(bn4)
	with tf.variable_scope('layer4-pool'):
		pool4 = tf.nn.max_pool(elu4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
	# 将第四层池化层的输出转化为全连接层的输入格式，注意，因为每一层神经网络的输入输出都为一个batch的矩阵，
	# 所以这里得到的维度也包含一个batch的数据的个数
	pool_shape = pool4.get_shape().as_list()
	# 计算将矩阵拉直成向量之后的长度，这个长度就是矩阵长宽及深度的乘积，注意这里pool_shape[0]为一个batch中数据的个数
	nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
	# 将第四层的输出转变为一个batch的向量
	reshaped = tf.reshape(pool4, [pool_shape[0], nodes])
	with tf.variable_scope('layer5-fc1'):
		fc1_weights = tf.get_variable('weight', [nodes, 512], initializer=tf.truncated_normal_initializer(stddev=0.1))
		# 只有全连接层的权重需要加入正则化
		if regularizer != None:
			tf.add_to_collection('losses', regularizer(fc1_weights))
		fc1_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))
		fbn1 = tf.matmul(reshaped, fc1_weights) + fc1_biases
		fc_BN = batch_norm_for_fc(fbn1, train, bn_decay, scope='fc_BN')
		fc1 = tf.nn.elu(fc_BN, name='fc1')
		if train is not None: fc1 = tf.nn.dropout(fc1, 0.5)

	with tf.variable_scope('layer6-fc2'):
		fc2_weights = tf.get_variable('weight', [512, 36], initializer=tf.truncated_normal_initializer(0.1))
		if regularizer != None:
			tf.add_to_collection('losses', regularizer(fc2_weights))
		fc2_biases = tf.get_variable('biase', [36], initializer=tf.constant_initializer(0.1))
		logit = tf.matmul(fc1, fc2_weights) + fc2_biases
	out = {
		'conv1': conv1,
		'pool1': pool1,
		'conv2': conv2,
		'pool2': pool2,
		'conv3': conv3,
		'pool3': pool3,
		'conv4': conv4,
		'pool4': pool4,
		'fc1': fc1,
		'logit': logit
	}
	return out


bat = [542, 1037, 731, 522, 428, 287, 601, 419, 845, 802, 778, 1060, 91, 433, 208]
# 配置神经网络参数
Batch_Size = 60
Learning_Rate_Base = 0.001
Learning_Rate_Decay = 0.99
Regularazition_Rate = 0.001
Training_Steps = 205  # 裁剪后的数据量为12311/60 = 205
bn_decay = 0.9
Moving_Average_Decay = 0.99
Model_Save_Path = "CNN10_NIHSS_v"
Model_Name = "model.ckpt"


def train(train, label, num):
	with tf.name_scope("input"):
		input_x = tf.placeholder(tf.float32, [Batch_Size, 5000, 9, 1], name='EEG-input')  # 数据的输入，第一维表示一个batch中样例的个数,100要加上，
		#  failed to convert object of type <class 'list'> to Tensor. Contents: [None, 4096]. Consider casting elements to a supported type.
		input_y = tf.placeholder(tf.float32, [None, 36], name='EEG-lable')  # 一个batch里的lable
	# reshaped_x = np.reshape(input_x,(100,500,9,1))#类似于将输入的训练数据格式调整为一个四维矩阵，并将这个调整的数据传入sess.run过程

	regularlizer = tf.contrib.layers.l2_regularizer(Regularazition_Rate)
	is_training = tf.cast(True, tf.bool)
	out = BaseCNN(input_x, is_training, regularlizer)  # 将数据放进去训练，得到最后全连接层输出结果
	# pre_y = BaseCNN(input_x,False,None)
	y = out['logit']

	with tf.name_scope("loss_function"):
		# 损失代价
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(input_y, 1))
		cross_mean = tf.reduce_mean(cross_entropy)
		# 加上L2正则化来计算损失函数
		loss = cross_mean + tf.add_n(tf.get_collection('losses'))
		tf.summary.scalar('loss', loss)
	# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=tf.argmax(input_y, 1)), name='cost')

	# 给定滑动平均衰减率和训练轮数的变量
	with tf.name_scope("moving_average"):
		global_step = tf.Variable(0, trainable=False)
		variable_averages = tf.train.ExponentialMovingAverage(Moving_Average_Decay, global_step)
		variable_averages_op = variable_averages.apply(tf.trainable_variables())

	with tf.name_scope("train_step"):
		# 学习率的更新:滑动平均模型
		learning_rate = tf.train.exponential_decay(
			Learning_Rate_Base,
			global_step,
			Training_Steps,
			Learning_Rate_Decay
		)
		# 优化损失函数
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
		# 在训练神经网络模型是，每过一遍数据既要反向传播更新参数，又要更新每一个参数滑动平均值。
		with tf.control_dependencies([train_step, variable_averages_op]):
			train_op = tf.no_op(name='train')
	# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

	with tf.name_scope("train_acc"):
		# 得到的准确度
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(input_y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
		tf.summary.scalar('train_acc', accuracy)
	# 开始训练网络
	merged = tf.summary.merge_all()
	saver = tf.train.Saver()
	with tf.Session() as sess:
		writer = tf.summary.FileWriter("CNN10_NIHSS_v" + str(num) + "_logs/", sess.graph)
		tf.global_variables_initializer().run()
		ds = DataSet(train, label)
		epochs = 5
		for e in range(epochs):
			for i in range(Training_Steps):
				x, y = ds.next_batch(Batch_Size)
				xs = np.reshape(x, (Batch_Size, 5000, 9, 1))
				ys = one_hot(y)
				# Feed dictionary
				feed = {input_x: xs, input_y: ys}
				# train_s,lo,acc,step = sess.run([train_op,loss,accuracy,global_step],feed_dict=
				summary, trainloss, trainacc, _ = sess.run([merged, loss, accuracy, train_op], feed_dict=feed)
				writer.add_summary(summary, i)
				if i % 17 == 0:
					print("after %g epoch: train loss: %g ,Train acc: %g" % (e,trainloss, trainacc))
					saver.save(sess, os.path.join(Model_Save_Path + str(num), Model_Name), global_step=global_step)
			# test_acc = sess.run(accuracy,feed_dict={input_x:test_x,input_y:test_y})
			# print("测试精度为：%g" % test_acc)
		writer.close()
def evaluate(train, label, batnum, num):
	with tf.name_scope("input"):
		input_x = tf.placeholder(tf.float32, [batnum, 5000, 9, 1], name='EEG-input')  # 数据的输入，第一维表示一个batch中样例的个数
		input_y = tf.placeholder(tf.float32, [None, 36], name='EEG-lable')  # 一个batch里的lable
	# regularlizer = tf.contrib.layers.l2_regularizer(Regularazition_Rate)  # 本来测试的时候不用加这个
	no_training = tf.cast(False, tf.bool)
	out = BaseCNN(input_x, no_training, None)
	y = out['logit']
	with tf.name_scope("test_acc"):
		correct_predection = tf.equal(tf.argmax(y, 1), tf.argmax(input_y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_predection, tf.float32))
		tf.summary.scalar('test_acc', accuracy)
	variable_averages = tf.train.ExponentialMovingAverage(Moving_Average_Decay)
	variables_to_restore = variable_averages.variables_to_restore()
	saver = tf.train.Saver(variables_to_restore)
	acc = []

	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state(Model_Save_Path + str(num))
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
			# x, y = test_batch()
			# x = x[100: 200]
			# y = y[100: 200]
			# xs = standardize(x)
			for i in range(4):
				x_test,y_label = train[i*batnum:(i+1)*batnum],label[i*batnum:(i+1)*batnum]
				reshape_xs = np.reshape(x_test, (-1, 5000, 9, 1))
				ys = one_hot(y_label)
				acc_score = sess.run(accuracy, feed_dict={input_x: reshape_xs, input_y: ys})
				print("Afer %s training step, test accuracy = %g" % (global_step, acc_score))
				acc.append(acc_score)

		else:
			print("No checkpoint file found")

	return acc
def main(argv=None):
	"""`
	x,y = test_batch()
	# 十倍交叉验证法，数组无法劈分开too many indices for array
	for i in range(10):
		x_test,y_test = x[i*13400:(i+1)*13400],y[i*13400:(i+1)*13400]
		x_train1,y_train1 =x[0:i*13400],y[0:i*13400]
		x_train = np.concatenate((x_train1, x[(i+1)*13400:134000]), axis=0)
		y_train = np.concatenate((y_train1, y[(i + 1) * 13400:134000]), axis=0)
		train(x_train,y_train,i)
		mean = evaluate(x_test,y_test,13400,i)
		"""
	# x_train, y_train = train_batch()
	x_train, y_train = tailor_train_batch()
	train(x_train, y_train, 8)
	# tf.reset_default_graph()
	# x_test, y_test = test_batch() # 750
	# x_test, y_test = tailor_test_batch() # 1844
	# mean = evaluate(x_test, y_test, 461, 1) #batnum = len(test)/4  461 187.5
	# print(np.mean(mean))

if __name__ == '__main__':
	tf.compat.v1.app.run()

# direct to the local dir and run this in terminal:
# $ tensorboard --logdir logs
