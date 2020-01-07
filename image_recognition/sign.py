import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import configparser


""" Load Dataset """
dftrain = pd.read_csv('SIGN_data/sign_mnist_train.csv')
dftest = pd.read_csv('SIGN_data/sign_mnist_test.csv')

Xtrain, ytrain = dftrain.loc[:,'pixel1':].values, dftrain.loc[:,'label'].values
Xtest, ytest = dftest.loc[:,'pixel1':].values, dftest.loc[:,'label'].values

Xtrain = Xtrain.astype('float32')
Xtest = Xtest.astype('float32')

Xtrain /=255.
Xtest /=255.

config = configparser.ConfigParser()
config.read('config.ini')

""" Load Configurations """
learning_rate = float(config["SMNIST"]["LEARNING"])
epochs = int(config["SMNIST"]["EPOCHS"])
batch_size = int(config["SMNIST"]["BATCHSIZE"])
num_batches = Xtrain.shape[0] // batch_size
dropout = float(config["SMNIST"]["DROPOUT"])
input_height = int(config["SMNIST"]["HEIGHT"])
input_width = int(config["SMNIST"]["WIDTH"])
filter_height = int(config["SMNIST"]["FHEIGHT"])
filter_width = int(config["SMNIST"]["FWIDTH"])
n_classes = int(config["SMNIST"]["CLASSES"])
depth_in = int(config["SMNIST"]["DIN"])
depth_out1 = int(config["SMNIST"]["DOUT1"])
depth_out2 = int(config["SMNIST"]["DOUT2"])
depth_out3 = int(config["SMNIST"]["DOUT3"])
display = int(config["SMNIST"]["DISPLAY"])

""" Define placeholder """
x = tf.placeholder(tf.float32, [None, input_height*input_width*depth_in])
y = tf.placeholder(tf.float32 , [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

""" Define weights and biases """ 
weights = {
    'wc1' : tf.Variable(tf.random_normal([filter_height, filter_width, depth_in, depth_out1])),
    'wc2' : tf.Variable(tf.random_normal([filter_height, filter_width, depth_out1, depth_out2])),
    'wd1' : tf.Variable(tf.random_normal([(input_height // 4) * (input_width // 4) * depth_out2, depth_out3])),
    'wout' : tf.Variable(tf.random_normal([depth_out3, n_classes]))
}

biases = {
    'bc1' : tf.Variable(tf.random_normal([depth_out1])),
    'bc2' : tf.Variable(tf.random_normal([depth_out2])),
    'bd1' : tf.Variable(tf.random_normal([depth_out3])),
    'bout' : tf.Variable(tf.random_normal([n_classes])) 
}

""" Define Convolutional Operation """
"""
--------------------------------------------------------
                CONVOLUTIONAL LAYER
--------------------------------------------------------

"""

def conv2d(x,W,b,strides = 1):
    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return x

""" Define Maxpooling Layer """
"""
---------------------------------------------------------------
                    MAX POOLING LAYER 
---------------------------------------------------------------

"""

def maxpool2d(x,strides=2):
    return tf.nn.max_pool(x,ksize=[1,strides,strides,1],strides=[1,strides,strides,1],padding='SAME')

""" Create feed forward Neural Network """
def convnet(x,weights,biases,dropout):
    x = tf.reshape(x,shape=[-1,input_height,input_width,depth_in])

    """ Convolution Operation 1 """
    conv1 = conv2d(x,weights['wc1'],biases['bc1'])
    conv1 = maxpool2d(conv1)
    conv1 = tf.nn.relu(conv1)
    # conv1 = tf.nn.dropout(conv1, dropout)

    """ Convolution Operation 2 """
    conv2 = conv2d(conv1, weights['wc2'],biases['bc2'])
    conv2 = maxpool2d(conv2)
    conv2 = tf.nn.relu(conv2)
    # conv2 = tf.nn.dropout(conv2, dropout)

    """ Fully Connected Layer """ 
    fc = tf.reshape(conv2, shape=[-1, weights['wd1'].get_shape().as_list()[0]])
    fc = tf.add(tf.matmul(fc, weights['wd1']), biases['bd1'])
    fc = tf.nn.relu(fc)

    """ Output Class Prediction """
    out = tf.add(tf.matmul(fc, weights['wout']), biases['bout'])
    out =tf.nn.softmax(out)
    return out


""" Define Tensorflow Ops """
pred = convnet(x,weights,biases,keep_prob)

""" Define Cost Function and it's Optimizer """
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

""" Calculate model evaluation or rather it's accuracy """
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

""" Initialize Tensorflow Variables """
init = tf.global_variables_initializer()

""" One Hot Encode labels """
def oneHot(data,n_classes):
    onehot = np.zeros([data.shape[0], n_classes])
    for i in range(len(data)):
        onehot[i, data[i]] = 1.
    return onehot

ytrain = oneHot(ytrain, n_classes)
ytest = oneHot(ytest,n_classes)

""" Start tensorflow session """
start_time = time.time()
with tf.Session() as sess:
    sess.run(init)
    for i in range(epochs):
        for j in range(num_batches):
            batchX, batchY = Xtrain[j*batch_size:(j+1)*batch_size,], ytrain[j*batch_size:(j+1)*batch_size, ]
            sess.run(optimizer, feed_dict = {x:batchX, y:batchY, keep_prob:dropout})
            loss, acc = sess.run([cost, accuracy], feed_dict = {x:batchX, y:batchY, keep_prob:dropout})
            if j % display == 0:
                print('Epoch : {} / Batch_Number: {}, Cost: {:.9f}, Accuracy: {:5f}'.format(i,j,loss,acc))


    print('Optimization Complete')
    y1 = sess.run(pred, feed_dict={x:Xtest, keep_prob:1.})
    test_classes = np.argmax(y1, 1)
    print('Test Accuracy : {:.5f}'.format(sess.run(accuracy, feed_dict={x:Xtest, y:ytest, keep_prob:1.})))
    alpha = [chr(ord('a') + i) for i in range(0,26)]
    f, a = plt.subplots(1,26, figsize=(10,2))
    for i in range(26):
        a[i].imshow(np.reshape(Xtest[i,:],(28,28)))
        print(alpha[test_classes[i]])
    plt.show()
    end_time = time.time()
    print('Total time taken : {}'.format(end_time - start_time))

