import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('index.jpeg')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,4))

plt.figure(figsize=(20,8))

img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img2 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)

lower = np.array([0, 20, 40], dtype = "uint8")
upper = np.array([30, 255, 255], dtype = "uint8")

mask = cv2.inRange(img2,lower,upper)

mask1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

img3 = cv2.bitwise_and(img1,img1, mask=mask1)

plt.figure(figsize=(20,8))
plt.imshow(img3)
plt.show()
