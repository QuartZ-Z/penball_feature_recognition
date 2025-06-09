import os
import cv2
import numpy as np

def load_image(data_folder):
    file_list = os.listdir(data_folder)
    image_list = []
    for file in file_list:
        if file.lower().endswith('.jpg') or file.lower().endswith('.png'):
            image_path = os.path.join(data_folder, file)
            image_list.append(image_path)
            print(f"Loaded image: {image_path}")
    return image_list

def detect_circle(image):
    # Convert the image to grayscale

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    Canny = cv2.Canny(gray, 100, 200)
    cv2.imshow('Canny', Canny)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=2, minDist=600,
                               param1=200, param2=40, minRadius=25, maxRadius=40)

    if circles is not None:
        # Convert the (x, y) coordinates and radius of the circles to integers
        circles = np.uint16(np.around(circles))

        # Draw the circles on the original image
        for i in circles[0, :]:
            # Draw the outer circle
            cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)

    return image

data_folder = 'test_images'
# image_list = ['test.jpg']
image_list = load_image(data_folder)

for image_path in image_list:
    image = cv2.imread(image_path)
    # Detect circles in the image
    output_image = detect_circle(image)
    # Display the output image
    # cv2.imshow('Detected Circles', output_image)
    cv2.imwrite('HT_output/' + os.path.basename(image_path), output_image)
    # cv2.waitKey(0)
cv2.destroyAllWindows()