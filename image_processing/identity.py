import numpy as np
import scipy.signal
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('monalisa.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

plt.imshow(gray, cmap='gray')
plt.show()


Im = np.array([[0,0,0], [0,1,0],[0,0,0]], dtype=np.float32)
G = scipy.signal.convolve2d(gray, Im, mode='same')
plt.imshow(G, cmap='gray')
plt.show()