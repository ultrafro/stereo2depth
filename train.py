import os 
import sys 

import numpy as np 
import tensorflow as tf 
from loader import loader


max_pool_k = 21
width = 256
height = 256
batch_size = 1
learning_rate = 0.01
beta = 0.5
iterations = 100

weights, biases = {}, {}


def pipe(x_vector, batch_size=batch_size, phase_train=True):

	print(str(np.shape(x_vector)))

	m_1 = tf.nn.conv2d(x_vector, weights['wc1'], strides=[1,1,1,1], padding="SAME")
	m_1 = tf.nn.bias_add(m_1, biases['bc1'])
	m_1 = tf.contrib.layers.batch_norm(m_1, is_training=phase_train)
	m_1 = tf.nn.max_pool(m_1, ksize=[1,max_pool_k,max_pool_k,1], strides=[1,2,2,1], padding="SAME")
	print(str(np.shape(m_1)))

	m_2 = tf.nn.conv2d(m_1, weights['wc2'], strides=[1,1,1,1], padding="SAME")
	m_2 = tf.nn.bias_add(m_2, biases['bc2'])
	m_2 = tf.contrib.layers.batch_norm(m_2, is_training=phase_train)
	m_2 = tf.nn.max_pool(m_2, ksize=[1,max_pool_k,max_pool_k,1], strides=[1,2,2,1], padding="SAME")
	print(str(np.shape(m_2)))

	m_3 = tf.nn.conv2d(m_2, weights['wc3'], strides=[1,1,1,1], padding="SAME")
	m_3 = tf.nn.bias_add(m_3, biases['bc3'])
	m_3 = tf.contrib.layers.batch_norm(m_3, is_training=phase_train)
	m_3 = tf.nn.max_pool(m_3, ksize=[1,max_pool_k,max_pool_k,1], strides=[1,2,2,1], padding="SAME")
	print(str(np.shape(m_3)))

	m_4 = tf.nn.conv2d(m_3, weights['wc4'], strides=[1,1,1,1], padding="SAME")
	m_4 = tf.nn.bias_add(m_4, biases['bc4'])
	m_4 = tf.contrib.layers.batch_norm(m_4, is_training=phase_train)
	#m_4 = tf.nn.max_pool(m_4, ksize=[1,max_pool_k,max_pool_k,1], strides=[1,2,2,1], padding="SAME")
	print(str(np.shape(m_4)))

	m_5 = tf.nn.conv2d_transpose(m_4, weights['wd5'], output_shape=[batch_size, int(width/4), int(height/4), 32], strides=[1,2,2,1], padding='SAME')
	m_5 = tf.nn.bias_add(m_5, biases['bd5'])
	print(str(np.shape(m_5)))

	m_6 = tf.nn.conv2d_transpose(m_5, weights['wd6'], output_shape=[batch_size, int(width/2), int(height/2), 16], strides=[1,2,2,1], padding='SAME')
	m_6 = tf.nn.bias_add(m_6, biases['bd6'])
	print(str(np.shape(m_6)))

	m_7 = tf.nn.conv2d_transpose(m_6, weights['wd7'], output_shape=[batch_size, int(width), int(height), 8], strides=[1,2,2,1], padding='SAME')
	m_7 = tf.nn.bias_add(m_7, biases['bd7'])
	print(str(np.shape(m_7)))

	m_8 = tf.nn.conv2d_transpose(m_7, weights['wd8'], output_shape=[batch_size, int(width), int(height), 1], strides=[1,1,1,1], padding='SAME')
	m_8 = tf.nn.bias_add(m_8, biases['bd8'])
	print(str(np.shape(m_8)))



	return m_8



#define weights:
xavier_init = tf.contrib.layers.xavier_initializer()
zero_init = tf.zeros_initializer()

weights['wc1'] = tf.get_variable("wc1", shape=[width, height, 2, 16], initializer=xavier_init)
biases['bc1'] = tf.get_variable("bc1", shape=[16], initializer=zero_init)

weights['wc2'] = tf.get_variable("wc2", shape=[width, height, 16, 32], initializer=xavier_init)
biases['bc2'] = tf.get_variable("bc2", shape=[32], initializer=zero_init)

weights['wc3'] = tf.get_variable("wc3", shape=[width, height, 32, 64], initializer=xavier_init)
biases['bc3'] = tf.get_variable("bc3", shape=[64], initializer=zero_init)

weights['wc4'] = tf.get_variable("wc4", shape=[width, height, 64, 128], initializer=xavier_init)
biases['bc4'] = tf.get_variable("bc4", shape=[128], initializer=zero_init)

weights['wd5'] = tf.get_variable("wd5", shape=[width, height, 32, 128], initializer=xavier_init)
biases['bd5'] = tf.get_variable("bd5", shape=[32], initializer=zero_init)

weights['wd6'] = tf.get_variable("wd6", shape=[width, height, 16, 32], initializer=xavier_init)
biases['bd6'] = tf.get_variable("bd6", shape=[16], initializer=zero_init)

weights['wd7'] = tf.get_variable("wd7", shape=[width, height, 8, 16], initializer=xavier_init)
biases['bd7'] = tf.get_variable("bd7", shape=[8], initializer=zero_init)

weights['wd8'] = tf.get_variable("wd8", shape=[width, height, 1, 8], initializer=xavier_init)
biases['bd8'] = tf.get_variable("bd8", shape=[1], initializer=zero_init)





#define inputs:
x_vector = tf.placeholder(shape=[batch_size,width,height,2],dtype=tf.float32) 
y_vector = tf.placeholder(shape=[batch_size,width,height,1],dtype=tf.float32)
output = pipe(x_vector)

pipe_loss = tf.nn.l2_loss(output - y_vector)


#training
params = [var for var in tf.trainable_variables() if any(x in var.name for x in ['wc', 'bc', 'wd'])]
optimizer_pipe = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta).minimize(pipe_loss,var_list=params)

load = loader();


with tf.Session() as sess:
	with tf.device("/gpu:0"): #("/gpu:0"):
		sess.run(tf.global_variables_initializer())

		#training loop:
		for iter in range(iterations):

			#x = np.random.normal(0, 256, size=[batch_size, 2, width, height]).astype(np.float32)
			#y = np.random.normal(0, 256, size=[batch_size, width, height, 1]).astype(np.float32)

			data = load.getBatch(batch_size);
			x = data['x']
			y = data['y']

			pipe_loss = sess.run(pipe_loss, feed_dict={x_vector:x, y_vector:y})
			print("iteration: " + str(iter) + " pipe loss: " + str(pipe_loss))

			sess.run(optimizer_pipe, feed_dict={x_vector:x, y_vector:y})



	# #phase_train = True
	# strides = [1,2,2,2,1]
	# #with tf.variable_scope("dis", reuse=False):
	# d_1 = tf.nn.conv3d(x, weights['wd1'], strides=strides, padding="SAME")
	# d_1 = tf.nn.bias_add(d_1, biases['bd1'])
	# d_1 = tf.contrib.layers.batch_norm(d_1, is_training=phase_train)                               
	# d_1 = lrelu(d_1, leak_value)

	# d_2 = tf.nn.conv3d(d_1, weights['wd2'], strides=strides, padding="SAME") 
	# d_2 = tf.nn.bias_add(d_2, biases['bd2'])
	# d_2 = tf.contrib.layers.batch_norm(d_2, is_training=phase_train)
	# d_2 = lrelu(d_2, leak_value)
	        
	# d_3 = tf.nn.conv3d(d_2, weights['wd3'], strides=strides, padding="SAME")  
	# d_3 = tf.nn.bias_add(d_3, biases['bd3'])
	# d_3 = tf.contrib.layers.batch_norm(d_3, is_training=phase_train)
	# d_3 = lrelu(d_3, leak_value) 

	# d_4 = tf.nn.conv3d(d_3, weights['wd4'], strides=strides, padding="SAME")     
	# d_4 = tf.nn.bias_add(d_4, biases['bd4'])
	# d_4 = tf.contrib.layers.batch_norm(d_4, is_training=phase_train)
	# d_4 = lrelu(d_4)

	# d_5 = tf.nn.conv3d(d_4, weights['wd5'], strides=[1,1,1,1,1], padding="VALID")     
	# d_5 = tf.nn.bias_add(d_5, biases['bd5'])
	# d_5 = tf.contrib.layers.batch_norm(d_5, is_training=phase_train)
	# d_5 = tf.nn.sigmoid(d_5)

	# return d_5;