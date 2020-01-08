import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import configparser
import time
from siamdataformer import DatasetFormer





colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900', '#009999']
config = configparser.ConfigParser()
config.read('config.ini')



""" Define configuration values """
epochs = int(config["SIAMESE"]["EPOCHS"])
learning_rate = float(config["SIAMESE"]["LEARNING"])
batch_size = int(config["SIAMESE"]["BATCHSIZE"])
input_height = int(config["SIAMESE"]["HEIGHT"])
input_width = int(config["SIAMESE"]["WIDTH"])
filter_height = int(config["SIAMESE"]["FHEIGHT"])
filter_width = int(config["SIAMESE"]["FWIDTH"])
depth_in = int(config["SIAMESE"]["DIN"])
depth_out1 = int(config["SIAMESE"]["DOUT1"])
depth_out2 = int(config["SIAMESE"]["DOUT2"])
depth_out3 = int(config["SIAMESE"]["DOUT3"])
depth_out4 = int(config["SIAMESE"]["DOUT4"])
similar = int(config["SIAMESE"]["SIMILARITYCLASSES"])
display = int(config["SIAMESE"]["DISPLAY"])




""" Define placeholder for the problem """
left = tf.placeholder(tf.float32, [None, input_height,input_width,depth_in])
right = tf.placeholder(tf.float32, [None, input_height,input_width,depth_in])
similarity = tf.placeholder(tf.float32, [None, 1])

""" Weight and biases """
weights = {
    'wc1' : tf.Variable(tf.random_normal([filter_height, filter_width, depth_in, depth_out1], mean = 0.0, stddev = 1e-2)),
    'wc2' : tf.Variable(tf.random_normal([filter_height - 2, filter_width - 2, depth_out1, depth_out2], mean = 0.0, stddev = 1e-2)),
    'wc3' : tf.Variable(tf.random_normal([filter_height - 4, filter_width - 4, depth_out2, depth_out3], mean = 0.0, stddev = 1e-2)),
    'wc4' : tf.Variable(tf.random_normal([filter_height - 6, filter_width - 6, depth_out3, depth_out4], mean = 0.0, stddev = 1e-2)),
    'wc5' : tf.Variable(tf.random_normal([filter_height - 6, filter_width - 6, depth_out4, similar], mean = 0.0, stddev = 1e-2))
}

biases = {
    'bc1' : tf.Variable(tf.random_normal([depth_out1], mean = 0.0, stddev = 1e-2)),
    'bc2' : tf.Variable(tf.random_normal([depth_out2], mean = 0.0, stddev = 1e-2)),
    'bc3' : tf.Variable(tf.random_normal([depth_out3], mean = 0.0, stddev = 1e-2)),
    'bc4' : tf.Variable(tf.random_normal([depth_out4], mean = 0.0, stddev = 1e-2)),
    'bc5' : tf.Variable(tf.random_normal([similar], mean = 0.0, stddev = 1e-2))
}

""" Define Convolutional Layer for the Network """
"""
----------------------------------------------------------------
                    CONVOLUTION LAYER
----------------------------------------------------------------
"""
def conv2d(x,W,b,strides = 1):
    x = tf.nn.conv2d(x, W, strides = [1,strides, strides, 1], padding = 'SAME')
    x = tf.nn.bias_add(x,b)
    return x

""" Define Maxpooling Layer for the Network """
"""
-----------------------------------------------------------------
                    MAX POOLING LAYER
-----------------------------------------------------------------
"""
def maxpool2d(x,strides=2):
    return tf.nn.max_pool(x,ksize=[1,strides,strides,1], strides=[1,strides,strides,1], padding='SAME')

""" Define the Convolutional Net """
def convnet(x,weights,biases):

    """ Convolutional Operation 1 """
    conv1 = conv2d(x,weights['wc1'],biases['bc1'])
    conv1 = maxpool2d(conv1)

    """ Convolutional Operation 2 """
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2)

    """ Convolutional Operation 3 """
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpool2d(conv3)

    """ Convolutional Operation 4 """
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    conv4 = maxpool2d(conv4)

    """ Convolutional Operation 5 """
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
    conv5 = maxpool2d(conv5)

    """ Flatten them out """
    net = tf.contrib.layers.flatten(conv5)
    return net

""" Define the Contrastive Loss Function for the model """
def contrastive_loss(model1, model2, y, margin):
    distance = tf.sqrt(tf.reduce_sum(tf.pow(model1 - model2, 2), 1, keep_dims = True))
    similarity = y * tf.square(distance)
    dissimilarity = (1 - y) * tf.square(tf.maximum((margin - distance), 0))
    return tf.reduce_mean(dissimilarity + similarity) / 2


""" Define Tensorflow Ops """
leftoutput = convnet(left,weights,biases)
rightoutput = convnet(right, weights, biases)
margin = 0.5
global_step = tf.Variable(0, trainable=False)

""" Define Cost Function and Optimizer for that """

cost = contrastive_loss(leftoutput, rightoutput, similarity, margin)
optimizer = tf.train.MomentumOptimizer(learning_rate, 0.99, use_nesterov=True).minimize(cost, global_step=global_step)

""" Initialize Tensorflow Variables """
init = tf.global_variables_initializer()


""" Call the data """
data = DatasetFormer()
next_batch = data.get_siamese_batch

tfconfig = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.90
""" Begin Tensorflow Session """
start_time = time.time()
with tf.Session(config=tfconfig) as sess:
    sess.run(init)
    for i in range(epochs):
        batch_left, batch_right, batch_similarity = next_batch(batch_size)
        _,l = sess.run([optimizer,cost], feed_dict = {left:batch_left, right:batch_right, similarity:batch_similarity})
        if (i + 1) % display == 0:
            print('Cost : {:.9f}'.format(l))
            feat = sess.run(leftoutput,feed_dict={left:data.Xtest})
            labels = data.ytest
            f = plt.figure(figsize=(16,9))
            f.set_tight_layout(True)
            for j in range(10):
                plt.plot(feat[labels == j, 0].flatten(), feat[labels==j,1].flatten(), '.', c=colors[j], alpha=0.8)
                plt.legend([str(i) for i in range(10)])
                plt.savefig('img/%d.jpg'%(i+1))



        
        

