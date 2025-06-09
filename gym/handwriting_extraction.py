import cv2
import numpy as np
import matplotlib.pyplot as plt

img_name = '039.JPG'

img = cv2.imread(img_name)
height, width = img.shape[:2]

img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(img_lab)

# 生成l通道的直方图
hist = cv2.calcHist([l], [0], None, [256], [0, 256])
hist = hist / (height * width)
# 绘制直方图
plt.plot(hist)
plt.show()

# 将l通道大于100的像素点置为255，小于100的像素点置为0
ret, binary = cv2.threshold(l, 100, 255, cv2.THRESH_BINARY)
cv2.imshow('binary', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()