import os
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

from DATA.Tailor_Test_Data10 import tailor_test_batch
from DATA.Tailor_Train_Data10 import tailor_train_batch
from DATASET.DataSet import DataSet

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Training Parameters
learning_rate = 0.0005
training_steps = 205
batch_size = 60
display_step = 17

# Network Parameters
    # data input (EEG shape: 5000*9)
timesteps = 5000 # timesteps
input_vec_size = 9 # 输入时间序列的维度
num_hidden = 128 # hidden layer num of features
num_classes = 7 # total classes (label num)
num_layer = 2 # LSTM层数
# tf Graph input
X = tf.placeholder("float", [None, timesteps, input_vec_size])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def one_hot(labels, n_class=7):
	""" One-hot encoding """
	expansion = np.eye(n_class)
	y = expansion[:, labels - 1].T
	assert y.shape[1] == n_class, "Wrong number of labels!"
	return y

def LSTM_Model(x,weights,biases):
	# Prepare data shape to match `rnn` function requirements
	# Current data input shape: (batch_size, timesteps, n_input)
	# Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

	# Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
	x = tf.unstack(x, timesteps, 1)

	# 使用多层的lstm结构
	# lstm_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0)
	#                                          for _  in range(num_layer)])
	lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
	lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, 0.5)
	# Get lstm cell output
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
	# outputs, states = tf.nn.dynamic_rnn(lstm_cell,x,dtype=tf.float32)
	# outputs是顶层LSTM在每一步的输出结果，他的维度是
	# [batch_size,time,num_hidden]

	# Linear activation, using rnn inner loop last output
	return tf.matmul(outputs[-1], weights['out']) + biases['out']

Model_Save_Path = "LSTM_MODEL_v"
Model_Name = "model.ckpt"
Learning_Rate_Base = 0.0005
Learning_Rate_Decay = 0.99
Regularazition_Rate = 0.001
Moving_Average_Decay = 0.99
batnum = 461
def Lstm_train(train,label,num):
	logits = LSTM_Model(X, weights, biases)
	prediction = tf.nn.softmax(logits)

	# Define loss and optimizer
	with tf.name_scope("loss_function"):
		# cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.argmax(Y, 1))
		# loss_op = tf.reduce_mean(cross_entropy)
		loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			logits=logits, labels=Y))
		tf.summary.scalar('loss', loss_op)

	# Evaluate model (with test logits, for dropout to be disabled)
	with tf.name_scope("train_acc"):
		correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		tf.summary.scalar('train_acc', accuracy)
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
			training_steps,
			Learning_Rate_Decay
		)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_step= optimizer.minimize(loss_op, global_step=global_step)
	with tf.control_dependencies([train_step, variable_averages_op]):
		train_op = tf.no_op(name='train')
	# Initialize the variables (i.e. assign their default value)
	init = tf.global_variables_initializer()
	merged = tf.summary.merge_all()
	saver = tf.train.Saver()
	# Start training
	with tf.Session() as sess:
		# Run the initializer
		sess.run(init)
		writer = tf.summary.FileWriter("LSTM_MODEL_v" + str(num) + "_logs/", sess.graph)
		ds = DataSet(train, label)

		epochs = 2
		for e in range(epochs):
			for step in range(1, training_steps + 1):
				batch_x, batch_y = ds.next_batch(batch_size)
				# Reshape data to get 5000 seq of 9 elements
				batch_x = batch_x.reshape((batch_size, timesteps, input_vec_size))
				batch_y = one_hot(batch_y)
				# Run optimization op (backprop)
				summary, loss, acc, _ = sess.run([merged, loss_op, accuracy, train_op], feed_dict={X: batch_x, Y: batch_y})
				writer.add_summary(summary, step)
				if step % display_step == 0 or step == 1:
					# Calculate batch loss and accuracy
					# loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
					saver.save(sess, os.path.join(Model_Save_Path + str(num), Model_Name), global_step=global_step)
					print("After Epoch " + str(e) + ",Step " + str(step) + ", Minibatch Loss= " + \
					      "{:.4f}".format(loss) + ", Training Accuracy= " + \
					      "{:.3f}".format(acc))
		x_test, y_test = tailor_test_batch()
		for i in range(4):
			x_test, y_label = x_test[i * batnum:(i + 1) * batnum], y_test[i * batnum:(i + 1) * batnum]
			reshape_xs = np.reshape(x_test, (-1, 5000, 9))
			ys = one_hot(y_label)
			acc_score = sess.run(accuracy, feed_dict={X: reshape_xs, Y: ys})
			print("Afer %s training step, test accuracy = %g" % (global_step, acc_score))
			acc.append(acc_score)

		writer.close()
		print("Optimization Finished!")

def Lstm_evaluate(train,label,batnum,num):
	saver = tf.train.Saver()
	acc = []
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state(Model_Save_Path + str(num))
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
			weight = sess.run(weights) # 提取原模型训练得到的参数大小值
			print(weight)

			biase = sess.run(biases) # 同上
			print(biase)
			logits = LSTM_Model(X, weight, biase)
			prediction = tf.nn.softmax(logits)
			# Define loss and optimizer
			loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
				logits=logits, labels=Y))

			# Evaluate model (with test logits, for dropout to be disabled)
			correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
			for i in range(4):
				x_test,y_label = train[i*batnum:(i+1)*batnum],label[i*batnum:(i+1)*batnum]
				reshape_xs = np.reshape(x_test, (-1, 5000, 9))
				ys = one_hot(y_label)
				acc_score = sess.run(accuracy, feed_dict={X: reshape_xs, Y: ys})
				print("Afer %s training step, test accuracy = %g" % (global_step, acc_score))
				acc.append(acc_score)
		else:
			print("No checkpoint file found")
	return acc

def main(argv=None):
	# x_train, y_train = train_batch()
	x_train, y_train = tailor_train_batch()
	Lstm_train(x_train, y_train, 2)
	# tf.reset_default_graph()
	# x_test, y_test = test_batch() # 750
	# x_test, y_test = tailor_test_batch()  # 1844
	# mean = Lstm_evaluate(x_test, y_test, 461, 1)  # batnum = len(test)/4  461 187.5
	# print(np.mean(mean))


if __name__ == '__main__':
	tf.compat.v1.app.run()