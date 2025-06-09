import cv2

img = cv2.imread('008.JPG', 0)
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))

cl1 = clahe.apply(img)
cv2.imwrite('008_clahe.jpg', cl1)