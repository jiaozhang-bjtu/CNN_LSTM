from __future__ import print_function

from keras import backend as K
from keras import Input, Model
from keras.utils.vis_utils import plot_model
import keras
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, TimeDistributed, BatchNormalization

from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from tensorflow.python.keras.models import load_model

# Embedding
from DATA.Tailor_Test_Data10 import tailor_test_batch
from DATA.Tailor_Train_Data10 import tailor_train_batch
from DATA.load_single_data10 import raw_test_batch10

maxlen = 5000
# Convolution
kernel_size = 3
filters = 18
pool_size = 2
# LSTM
lstm_output_size = 128
# Training
batch_size = 60
epochs = 6

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''
def CNN_LSTM_train():

	x_train, y_train = tailor_train_batch()
	x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
	print('x_train shape:', x_train.shape)
	model = Sequential()
	model.add(Conv1D(filters,
	                 kernel_size,
	                 padding='same',
	                 strides=1,name='conv1'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(MaxPooling1D(pool_size=pool_size,name='pool1'))
	model.add(Conv1D(36,
	                 kernel_size,
	                 padding='same',
	                 strides=1,name='conv2'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(MaxPooling1D(pool_size=pool_size,name='pool2'))
	model.add(Conv1D(72,
	                 kernel_size,
	                 padding='same',
	                 strides=1,name='conv3'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(MaxPooling1D(pool_size=pool_size,name='pool3'))
	model.add(Conv1D(144,
	                 kernel_size,
	                 padding='same',
	                 strides=1,name='conv4'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(MaxPooling1D(pool_size=pool_size,name='pool4'))
	model.add(LSTM(lstm_output_size,name='lstm'))
	model.add(Dense(7,name='dense1'))
	model.add(Activation('softmax',name='softmax'))
	adam = Adam(0.00325)
	model.compile(loss='sparse_categorical_crossentropy',
	              optimizer=adam,
	              metrics=['accuracy'])

	print('Train...')
	model.fit(x_train, y_train,
	          batch_size=batch_size,
	          epochs=epochs)
	model.save_weights('my_model_weights.h5')
	model.save('CNN_LSTM_model_v23.h5')   # HDF5 file

def test():
	keras.backend.clear_session()
	model = load_model('CNN_LSTM_model_v23.h5')

	# x_test, y_test = tailor_train_batch()
	x_test, y_test = tailor_test_batch()
	# intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('conv2').output)
	# intermediate_output = intermediate_layer_model.predict(x_test)
	score, acc = model.evaluate(x_test, y_test, batch_size=461)
	print('Test score:', score)
	print('Test accuracy:', acc)

def elva(data):
	x_test, y_test = raw_test_batch10(data)
	x_test = sequence.pad_sequences(x_test,maxlen=maxlen,dtype='float32')
	model = Sequential()
	model.add(Conv1D(filters,
	                 kernel_size,
	                 padding='same',
	                 strides=1, name='conv1'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(MaxPooling1D(pool_size=pool_size, name='pool1'))
	model.add(Conv1D(36,
	                 kernel_size,
	                 padding='same',
	                 strides=1, name='conv2'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(MaxPooling1D(pool_size=pool_size, name='pool2'))
	model.add(Conv1D(72,
	                 kernel_size,
	                 padding='same',
	                 strides=1, name='conv3'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(MaxPooling1D(pool_size=pool_size, name='pool3'))
	model.add(Conv1D(144,
	                 kernel_size,
	                 padding='same',
	                 strides=1, name='conv4'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(MaxPooling1D(pool_size=pool_size, name='pool4'))
	model.add(LSTM(lstm_output_size, name='lstm'))
	model.add(Dense(7, name='dense1'))
	model.add(Activation('softmax', name='softmax'))
	adam = Adam(0.00325)
	model.compile(loss='sparse_categorical_crossentropy',
	              optimizer=adam,
	              metrics=['accuracy'])


	model = load_model('CNN_LSTM_model_v23.h5')
	score, acc = model.evaluate(x_test, y_test)
	# model.load_weights("my_model_weights.h5", by_name=True)
	# intermediate_layer_model1 = K.function(inputs=[model.layers[0].inputs], outputs=[model.get_layer('conv1').output])
	# print(model.inputs)
	intermediate_layer_model1 = Model(inputs=model.inputs, outputs=model.get_layer('conv1').output)
	intermediate_layer_model2 = Model(inputs=model.inputs, outputs=model.get_layer('pool1').output)
	intermediate_layer_model3 = Model(inputs=model.inputs, outputs=model.get_layer('conv2').output)
	intermediate_layer_model4 = Model(inputs=model.inputs, outputs=model.get_layer('pool2').output)
	intermediate_layer_model5 = Model(inputs=model.inputs, outputs=model.get_layer('conv3').output)
	intermediate_layer_model6 = Model(inputs=model.inputs, outputs=model.get_layer('pool3').output)
	intermediate_layer_model7 = Model(inputs=model.inputs, outputs=model.get_layer('conv4').output)
	intermediate_layer_model8 = Model(inputs=model.inputs, outputs=model.get_layer('pool4').output)
	intermediate_layer_model9 = Model(inputs=model.inputs, outputs=model.get_layer('lstm').output)

	intermediate_output1 = intermediate_layer_model1.predict(x_test)
	print(intermediate_output1.shape)
	intermediate_output2 = intermediate_layer_model2.predict(x_test)
	print(intermediate_output2.shape)
	intermediate_output3 = intermediate_layer_model3.predict(x_test)
	print(intermediate_output3.shape)
	intermediate_output4 = intermediate_layer_model4.predict(x_test)
	print(intermediate_output4.shape)
	intermediate_output5 = intermediate_layer_model5.predict(x_test)
	print(intermediate_output5.shape)
	intermediate_output6 = intermediate_layer_model6.predict(x_test)
	print(intermediate_output6.shape)
	intermediate_output7 = intermediate_layer_model7.predict(x_test)
	print(intermediate_output7.shape)
	intermediate_output8 = intermediate_layer_model8.predict(x_test)
	print(intermediate_output8.shape)
	intermediate_output9 = intermediate_layer_model9.predict(x_test)
	print(intermediate_output9.shape)
	out = {
		'conv1': intermediate_output1,
		'pool1': intermediate_output2,
		'conv2': intermediate_output3,
		'pool2': intermediate_output4,
		'conv3': intermediate_output5,
		'pool3': intermediate_output6,
		'conv4': intermediate_output7,
		'pool4': intermediate_output8,
		'lstm': intermediate_output9,
	}
	return out



# CNN_LSTM_train()
# test()
def sum():
	for i in range(1):
		keras.backend.clear_session()
		output = elva(i)
sum()
