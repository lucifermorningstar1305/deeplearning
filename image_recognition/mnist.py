import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import time
import configparser

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
config = configparser.ConfigParser()
config.read('config.ini')


## Set the value of the parameters
learning_rate = float(config["MNIST"]["LEARNING"])
epochs = int(config["MNIST"]["EPOCHS"])
batch_size = int(config["MNIST"]["BATCHSIZE"])
num_batches = mnist.train.num_examples // batch_size
input_height = int(config["MNIST"]["HEIGHT"])
input_width = int(config["MNIST"]["WIDTH"])
n_classes = int(config["MNIST"]["CLASSES"])
dropout = float(config["MNIST"]["DROPOUT"])
filter_height = int(config["MNIST"]["FHEIGHT"])
filter_width = int(config["MNIST"]["FWIDTH"])
depth_in = int(config["MNIST"]["DIN"])
depth_out1 =  int(config["MNIST"]["DOUT1"])
depth_out2 = int(config["MNIST"]["DOUT2"])
display = int(config["MNIST"]["DISPLAY"])


## Input Output Definitions
x = tf.placeholder(tf.float32, [None, input_height*input_width])
y = tf.placeholder(tf.float32, [None,n_classes])
keep_prob = tf.placeholder(tf.float32)

## Store the weights 
"""
Number of weights of filters to be learned in 'wc1' => filter_height*filter_width*filter_depth_in*filter_depth_out1
Number of weights of filters to be learned in 'wc2' => filter_height*filter_width*filter_depth_out1*filter_depth_out2

Number of Connections to the fully Connected layer => Each maxpooling operation reduces the image_size to 1/4.
So two maxpooling reduces the image size to 1/16. There are depth_out2 number of images each of size 1/16 of the original image size of 
input_height*input_width. So there is total of (1/16)*input_height*input_width*depth_out2 pixel_outputs which when connected to the fully_connected
layers with 1024 units would provide (1/16)*input_height*input_width*depth_out2*1024 connections.

"""

weights = {
    'wc1' : tf.Variable(tf.random_normal([filter_height, filter_width, depth_in, depth_out1])),
    'wc2' : tf.Variable(tf.random_normal([filter_height, filter_width, depth_out1, depth_out2])),
    'wd1' : tf.Variable(tf.random_normal([(input_height//4)*(input_width//4)*depth_out2, 1024])),
    'wout' : tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1' : tf.Variable(tf.random_normal([depth_out1])),
    'bc2' : tf.Variable(tf.random_normal([depth_out2])),
    'bd1' : tf.Variable(tf.random_normal([1024])),
    'bout' : tf.Variable(tf.random_normal([n_classes]))
}



"""
--------------------------------------------------------------
                CONVOLUTION LAYER
--------------------------------------------------------------

"""

def conv2d(x,W,b,strides = 1):
    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1], padding='SAME')
    x = tf.nn.bias_add(x,b)
    return x


"""
-----------------------------------------------------------------
                MAX POOLING LAYER
-----------------------------------------------------------------

"""

def maxpool2d(x, strides=2):
    return tf.nn.max_pool(x,ksize=[1,strides,strides,1],strides=[1,strides,strides,1],padding='SAME')


"""
Create feed forward neural network 
Reshape the input in the 4 dimensional image
1st dimension : image index
2nd dimension : height
3rd dimension : width
4th dimension : depth

"""

def convnet(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1,input_height, input_width, depth_in])

    ## Convolution Layer 1
    conv1 = conv2d(x,weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1,2)

    ## Convolution Layer 2
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2,2)

    ## Fully connected Layer
    fc1 = tf.reshape(conv2 , shape=[-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    ## Apply Dropout
    fc1 = tf.nn.dropout(fc1,dropout)

    ## Output Class prediction
    out = tf.add(tf.matmul(fc1, weights['wout']), biases['bout'])
    return out


"""
Defining the tensorflow Ops for different activities

"""
pred = convnet(x,weights, biases, keep_prob)


"""
Define Loss Function and Optimizer
"""

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


"""
Evaluate model
"""
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


"""
Initializing all variable

"""
init = tf.global_variables_initializer()

"""
Launch the Graph

"""
start_time = time.time()

with tf.Session() as sess:
    sess.run(init)
    for i in range(epochs):
        for j in range(num_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict= {x:batch_x, y:batch_y, keep_prob:dropout})
            loss, acc = sess.run([cost, accuracy], feed_dict={x:batch_x, y:batch_y, keep_prob:dropout})
            if i % display == 0:
                print("Epoch: {} , cost : {:.9f}, Training accuracy: {:.5f}".format(i+1, loss, acc))
        
    
    print('Optimization Complete')

    y1 = sess.run(pred, feed_dict={x:mnist.test.images[:256], keep_prob:1})
    test_classes = np.argmax(y1,1)
    print('Testing Accuracy: {}'.format(sess.run(accuracy, feed_dict={x:mnist.test.images[:256], y:mnist.test.labels[:256], keep_prob:1.})))
    f, a = plt.subplots(1, 10, figsize=(10,2))
    for  i in range(10):
        a[i].imshow(np.reshape(mnist.test.images[i], (28,28)))
        print(test_classes[i])
    plt.show()
    end_time = time.time()
    print('Total_Time to process: {}'.format(end_time - start_time))
        
