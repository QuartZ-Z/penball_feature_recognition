import os
import shutil
import cv2
import numpy as np

source_folder = '@original_images/'
output_folder = '@formatted_images/'
scale = 1.0
filter_thres = 10000

def format_path(path):
    return path.replace(os.sep, '-')

def format_image(img):
    height, width = img.shape[:2]
    img = cv2.resize(img, (int(width * scale), int(height * scale)))
    height = int(height * scale)
    width = int(width * scale)
    
    # img_guassian = cv2.GaussianBlur(img, (5, 5), 0)
    # hsv = cv2.cvtColor(img_guassian, cv2.COLOR_BGR2HSV)
    # Canny = cv2.Canny(hsv, 50, 100)

    # # cv2.imshow('Canny', Canny)
    # # cv2.waitKey(0)
    
    # mask = Canny.copy()
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # mask = cv2.dilate(mask, kernel, iterations = 2)
    # mask = cv2.erode(mask, kernel, iterations = 2)

    # num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # small_components = np.where(stats[:, cv2.CC_STAT_AREA] < area_threshold)[0]
    # for component in small_components:
    #     mask[labels == component] = 0

    # mask = cv2.bitwise_not(mask)
    # num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # small_components = np.where(stats[:, cv2.CC_STAT_AREA] < area_threshold)[0]
    # for component in small_components:
    #     mask[labels == component] = 0

    # mask = cv2.bitwise_not(mask)
    # # cv2.imshow('mask', mask)

    # skeleton = skimage.morphology.skeletonize(mask // 255)
    # skeleton = skeleton.astype(np.uint8) * 255
    # # cv2.imshow('skeleton', skeleton)

    # img = cv2.bitwise_and(img, img, mask = mask)
    return img
    

def process_files(source_folder, output_folder):
    shutil.rmtree(output_folder, ignore_errors=True)
    os.makedirs(output_folder, exist_ok=True)
    for root, dirs, files in os.walk(source_folder):
        # print(root, dirs, files)
        for file in files:
            if file.lower().endswith('.jpg') or file.lower().endswith('.png'):
                relative_path = os.path.relpath(os.path.join(root, file), source_folder)
                formatted_path = format_path(relative_path)
                output_path = os.path.join(output_folder, formatted_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                img = cv2.imread(os.path.join(root, file))

                if img is None:
                    print(f'Could not read {file}')
                    continue
                else:
                    img = format_image(img)

                cv2.imwrite(output_path, img)
                print(f'Processed {output_path}')

if __name__ == '__main__':
    process_files(source_folder, output_folder)