import cv2
import numpy as np
import os
import shutil
import random

input_folder = '_100g_white_a4_circle/'
output_folder = '_100g_white_a4_circle_augmented/'

input_train_folder = os.path.join(input_folder, 'train')
input_val_folder = os.path.join(input_folder, 'val')
input_test_folder = os.path.join(input_folder, 'test')
output_train_folder = os.path.join(output_folder, 'train')
output_val_folder = os.path.join(output_folder, 'val')
output_test_folder = os.path.join(output_folder, 'test')

def switch_color(image, annotation, mode):
    if mode == 0:
        return image, annotation
    elif mode == 1:
        return np.roll(image, 1, axis = 2), annotation
    else:
        return np.roll(image, 2, axis = 2), annotation

def flip(image, annotation, mode):
    if mode == 0:
        return image, annotation
    elif mode == 1:
        return cv2.flip(image, 1), [(id, 1 - x, y, w, h) for id, x, y, w, h in annotation]
    elif mode == 2:
        return cv2.flip(image, 0), [(id, x, 1 - y, w, h) for id, x, y, w, h in annotation]
    elif mode == 3:
        return cv2.flip(image, -1), [(id, 1 - x, 1 - y, w, h) for id, x, y, w, h in annotation]
    
def add_noise(image, annotation, noise_factor):
    noise = np.random.normal(0, noise_factor, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8), annotation

def rotate(image, annotation, angle):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    new_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    new_annotation = []
    # for temp in annotation:
    #     print(temp, len(temp))
    for id, x, y, w, h in annotation:
        x = x * width - width / 2
        y = y * height - height / 2
        new_x = x * np.cos(np.radians(angle)) + y * np.sin(np.radians(angle)) + width / 2
        new_y = y * np.cos(np.radians(angle)) - x * np.sin(np.radians(angle)) + height / 2
        new_x = new_x / width
        new_y = new_y / height
        new_annotation.append((id, new_x, new_y, w, h))
    return new_image, new_annotation

def regularize_annotation(annotation):
    new_annotation = []
    for id, x, y, w, h in annotation:
        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y + h / 2
        x1 = max(0, min(1, x1))
        y1 = max(0, min(1, y1))
        x2 = max(0, min(1, x2))
        y2 = max(0, min(1, y2))
        new_annotation.append((id, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return new_annotation

def augment_image(image_path, label_path, output_folder, noise_factor = 20, angle = 0):
    image = cv2.imread(image_path)
    with open(label_path, 'r') as f:
        annotation = [tuple(map(float, line.strip().split())) for line in f]
    for mode in range(12):
        flip_mode = mode % 4
        color_mode = (mode // 4) % 3
        # print('Augmenting', image_path, 'with mode', mode, ': ', annotation)
        new_image, new_annotation = flip(image, annotation, flip_mode)
        # print(new_annotation)
        new_image, new_annotation = switch_color(new_image, new_annotation, color_mode)
        # print(new_annotation)
        new_image, new_annotation = add_noise(new_image, new_annotation, noise_factor)
        # print(new_annotation)
        if angle < 0:
            angle = random.randint(-15, 0)
        elif angle > 0:
            angle = random.randint(0, 15)
        # print(angle)
        new_image, new_annotation = rotate(new_image, new_annotation, angle)
        # print(new_annotation)
        new_annotation = regularize_annotation(new_annotation)
        # print(new_annotation)
        image_name = f'images/augmented_{mode}_{noise_factor}_{angle}_{os.path.basename(image_path)}'
        label_name = f'labels/augmented_{mode}_{noise_factor}_{angle}_{os.path.basename(label_path)}'
        output_image_path = os.path.join(output_folder, image_name)
        output_label_path = os.path.join(output_folder, label_name)
        cv2.imwrite(output_image_path, new_image)
        with open(output_label_path, 'w') as f:
            for id, x, y, w, h in new_annotation:
                f.write(f'{int(id)} {x} {y} {w} {h}\n')

if __name__ == '__main__':
    shutil.rmtree(output_train_folder, ignore_errors = True)
    shutil.rmtree(output_val_folder, ignore_errors = True)
    shutil.rmtree(output_test_folder, ignore_errors = True)
    os.makedirs(output_train_folder, exist_ok = True)
    os.makedirs(output_val_folder, exist_ok = True)
    os.makedirs(output_test_folder, exist_ok = True)

    shutil.copytree(input_val_folder, output_val_folder, dirs_exist_ok = True)
    shutil.copytree(input_test_folder, output_test_folder, dirs_exist_ok = True)

    input_image_folder = os.path.join(input_train_folder, 'images')
    input_label_folder = os.path.join(input_train_folder, 'labels')
    output_image_folder = os.path.join(output_train_folder, 'images')
    output_label_folder = os.path.join(output_train_folder, 'labels')

    os.makedirs(output_image_folder, exist_ok = True)
    os.makedirs(output_label_folder, exist_ok = True)

    image_names = os.listdir(input_image_folder)
    # print(image_names)
    print(len(image_names))
    for name in image_names:
        print('Augmenting', name)
        if name.lower().endswith('.jpg') or name.lower().endswith('.png'):
            image_path = os.path.join(input_image_folder, name)
            label_path = os.path.join(input_label_folder, name[:-4] + '.txt')
            augment_image(image_path, label_path, os.path.join(output_folder, 'train'), angle = 1)
            augment_image(image_path, label_path, os.path.join(output_folder, 'train'), angle = -1)