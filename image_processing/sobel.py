import numpy as np
import scipy.signal
import cv2
import matplotlib.pyplot as plt



fig = plt.figure(figsize=(2,2))
row = 2
column = 2



img = cv2.imread('monalisa.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# plt.imshow(gray, cmap='gray')
# plt.show()

## Applying Median Filter to reduce Salt and Pepper Noise
Gm = cv2.medianBlur(gray,3)
# plt.imshow(Gm,cmap='gray')
# plt.show()

Hx = np.array([[1,0,-1], [2,0,-2],[1,0,-1]], dtype=np.float32)
Hy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)
Gx = scipy.signal.convolve2d(Gm, Hx, mode ='same')
Gy = scipy.signal.convolve2d(Gm,Hy,mode = 'same')

# plt.imshow(Gx, cmap='gray')
# plt.show()
# plt.imshow(Gy, cmap='gray')
# plt.show()

G = (Gx*Gx + Gy*Gy) ** 0.5 
# plt.imshow(G, cmap='gray')
# plt.show()

imgs = np.array([ gray, Gx, Gy, G ])
labels = ['Original', 'Horizontal', 'Vertical', 'Filtered']
for i in range(1, column*row+1):

    ax = fig.add_subplot(row,column,i)
    ax.set_title(labels[i-1])
    plt.imshow(imgs[i-1], cmap='gray')
plt.show()
