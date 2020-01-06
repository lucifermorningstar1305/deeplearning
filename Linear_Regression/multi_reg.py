import numpy as np
import importlib
importlib.import_module('mpl_toolkits.mplot3d').Axes3D
import matplotlib.pyplot as plt

#load the data
X=[]
Y=[]
for line in open('data_2d.csv'):
	x1,x2,y=line.split(',')
	X.append([1,float(x1),float(x2)])
	Y.append(float(y))

#turn X and y to numpy arrays
X=np.asarray(X)
Y=np.asarray(Y)

#plot the data 
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(X[:,0],X[:,1],Y)
plt.show()

#calculate weights of the model
w=np.linalg.solve(np.dot(X.T,X),np.dot(X.T,Y))
Yhat=np.dot(X,w)

#r-squared value
d1=Y-Yhat
d2=Y-Y.mean()
r2=1-(d1.dot(d1)/d2.dot(d2))
print 'R-squared value is: %s' %(r2)