import cv2
import matplotlib 
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(2,2))
row = 1
column = 2
img = cv2.imread('monalisa.jpg')
# cv2.imshow('Frame',img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray,cmap='gray')
# plt.show()

## Applying Gaussian Filter
GaussFilter = np.array([[1,1,1], [1,1,1], [1,1,1]])/float(9)
Gm = scipy.signal.convolve2d(gray,GaussFilter,mode='same')
# plt.imshow(Gm,cmap='gray')
# plt.show()
imgs = np.array([gray, Gm])
labels = ['Original','Filtered']
for i in range(1, column*row+1):
    ax = fig.add_subplot(row,column,i)
    ax.set_title(labels[i-1])
    plt.imshow(imgs[i-1], cmap='gray')
plt.show()

