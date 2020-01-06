import numpy as np
import cv2
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(2,2))
row = 1
column = 2


img = cv2.imread('monalisa.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


## Applying Median Filter
Gm = cv2.medianBlur(gray,3)

imgs = np.array([gray, Gm])
labels = ['Original','Filtered']
for i in range(1, column*row+1):
    ax = fig.add_subplot(row,column,i)
    ax.set_title(labels[i-1])
    plt.imshow(imgs[i-1], cmap='gray')
plt.show()


