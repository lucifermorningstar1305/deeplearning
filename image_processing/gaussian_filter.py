import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal



fig = plt.figure(figsize=(2,2))
row = 2
column = 2

Hg = np.zeros((20,20))

img = cv2.imread('monalisa.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


for i in range(20):
    for j in range(20):
        Hg[i,j] = np.exp(-((i-10) ** 2 + (j-10)**2)/10)



## Applying Gaussian Blur
gaussian_blur = scipy.signal.convolve2d(gray, Hg, mode='same')


gray_high = gray - gaussian_blur


gray_enhanced = gray + 0.025 * gray_high

imgs = np.array([ gray, gaussian_blur, gray_high, gray_enhanced])
labels = ['Original', 'Filtered', 'High Components', 'Enhanced']
for i in range(1, column*row+1):

    ax = fig.add_subplot(row,column,i)
    ax.set_title(labels[i-1])
    plt.imshow(imgs[i-1], cmap='gray')
plt.show()
