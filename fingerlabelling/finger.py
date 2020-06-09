import numpy as np
import matplotlib.pyplot as plt
import cv2
##takes image
img = cv2.imread('index.jpeg')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,4))
#conversion to differnet color spaces
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
#HSV range for skin
lower = np.array([0, 20, 40])
upper = np.array([30, 255, 255])

mask = cv2.inRange(img2,lower,upper)
#remove noise
mask1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

img3 = cv2.bitwise_and(img1,img1, mask=mask1)
##find convex hull
img_gray = cv2.cvtColor(img3,cv2.COLOR_RGB2GRAY)
ret,thresh = cv2.threshold(img_gray, 40, 255,0)
contours,hierarchy = cv2.findContours(thresh,2,1)
cnt = contours[0]
hull = cv2.convexHull(cnt,returnPoints = False)
defects = cv2.convexityDefects(cnt,hull)
for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e-1][0])
    far = tuple(cnt[f][0])
#highlighting relavent points
    cv2.circle(img1,end,5,[0,250,0],2)
    cv2.circle(img1,far,2,[0,250,0],2)
    cv2.line(img1,far,start,[255,255,255],2)
    cv2.line(img1,far,end,[255,255,255],2)
plt.imshow(img1)
plt.show()
