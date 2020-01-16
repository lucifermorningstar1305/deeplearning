import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import configparser
import time
import os
from sklearn.model_selection import train_test_split

""" Load dataset from Csv"""
data = pd.read_csv(os.path.join(os.getcwd(),'bottle.csv'))

config = configparser.ConfigParser()
config.read(os.path.join(os.getcwd(), 'config.ini'))

X, y = data.loc[:60000,'Salnty'], data.loc[:60000,'T_degC']

print('Salinty Description')
print(X.describe())
print('*'*63)
print('Water Temperature Description')
print(y.describe())

print("Number of null values in input Data: {}".format(X.isna().sum()))
print("Number of null values in output Data: {}".format(y.isna().sum()))

""" Replacing nan values by mean """
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace = True)

print("Number of null values in input Data: {}".format(X.isna().sum()))
print("Number of null values in output Data: {}".format(y.isna().sum()))

X, y = X.values, y.values

X = X.reshape([-1,1])
y = y.reshape([-1,1])

X = X.astype(float)
y = y.astype(float)


print('Length of Input Data: {} Length of Output Data: {}'.format(X.shape, y.shape))

""" Split Train/Test data """
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state = 42)

""" Normalize the data """
# Xtrain = (Xtrain - np.mean(Xtrain, axis=0))/np.std(Xtrain,axis=0)
# Xtest = (Xtest - np.mean(Xtest, axis = 0))/np.std(Xtest,axis=0)
Xtrain = (Xtrain - Xtrain.min()) / (Xtrain.max() - Xtrain.min())
Xtest = (Xtest - Xtest.min()) / (Xtest.max() - Xtest.min())
""" Plot Data using matplotlib """
plt.scatter(Xtrain,ytrain)
plt.xlabel('Salinity of the Water')
plt.ylabel('Water Temperature in degree Centigrade')
z = np.polyfit(Xtrain.flatten(),ytrain.flatten(),1)
p = np.poly1d(z)
plt.plot(Xtrain,p(Xtrain), "r--")
plt.show()



""" Configurations """
epochs = int(config['OCEAN']['EPOCHS'])
learning_rate = float(config['OCEAN']['LEARNING_RATE'])
hidden1 = int(config['OCEAN']['H1INPUT'])
batch_size = int(config['OCEAN']['BATCHSIZE'])
display = int(config['OCEAN']['DISPLAY'])
num_batches = Xtrain.shape[0] // batch_size

""" Define placeholders """
varx = tf.placeholder(tf.float32, [None, Xtrain.shape[1]], name='inputs')
vary = tf.placeholder(tf.float32, [None, ytrain.shape[1]], name='labels')


""" Define Weights and Biases """
weights = {
    'wd1' : tf.Variable(tf.random_normal([Xtrain.shape[1], hidden1])),
    'wd2' : tf.Variable(tf.random_normal([hidden1, ytrain.shape[1]]))
}

biases = {
    'bd1' : tf.Variable(tf.random_normal([hidden1])),
    'bd2' : tf.Variable(tf.random_normal([ytrain.shape[1]]))
}
""" Learning the data """
pred = tf.add(tf.matmul(tf.add(tf.matmul(varx,weights['wd1']),biases['bd2']), weights['wd2']), biases['bd2'])

""" Define Error and Cost Function and Optimizer"""
error = pred- vary
cost = tf.reduce_mean(tf.square(error))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

""" Evaluate Model """
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(vary,1))

""" Initialize all variables """
init = tf.global_variables_initializer()

""" Run Tensorflow Session """
loss_curve = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(epochs):
        for j in range(num_batches):
            batchX, batchY = Xtrain[j*batch_size:(j+1)*batch_size,:], ytrain[j*batch_size:(j+1)*batch_size,:]
            _, loss = sess.run([optimizer, cost], feed_dict = {varx:batchX, vary:batchY})
            
            if j % display == 0:
                print('Epoch/Batch {}/{} MSE: {:.9f}'.format(i,j,loss))
                loss_curve.append(loss)



    
    print("Optimization Complete")
    plt.plot(loss_curve,color='blue',label='loss-curve')
    plt.legend()
    plt.show()
    
    rmse = tf.sqrt(tf.reduce_mean((pred-vary) ** 2))
    print('RMSE value : {:.5f}'.format(sess.run(rmse, feed_dict={varx:Xtest,vary:ytest})))

    y1 = sess.run(pred,feed_dict={varx:Xtest})

    plt.scatter(Xtest,ytest)
    plt.plot(Xtest,y1,color='red', label='trend')
    plt.legend()
    plt.show()






