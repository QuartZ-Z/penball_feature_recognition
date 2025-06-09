import os
import shutil
import random

source_folder = './'
output_folder = '_phone_riceyellow_a4/'
train_ratio = 0.0
val_ratio = 0.0
test_ratio = 1.0

if __name__ == "__main__":
    source_images = os.path.join(source_folder, 'images')
    source_labels = os.path.join(source_folder, 'labels')
    train_images = os.path.join(output_folder, 'train', 'images')
    train_labels = os.path.join(output_folder, 'train', 'labels')
    val_images = os.path.join(output_folder, 'val', 'images')
    val_labels = os.path.join(output_folder, 'val', 'labels')
    test_images = os.path.join(output_folder, 'test', 'images')
    test_labels = os.path.join(output_folder, 'test', 'labels')
    
    shutil.rmtree(os.path.join(output_folder, 'train'), ignore_errors = True)
    shutil.rmtree(os.path.join(output_folder, 'val'), ignore_errors = True)
    shutil.rmtree(os.path.join(output_folder, 'test'), ignore_errors = True)
    os.makedirs(train_images, exist_ok=True)
    os.makedirs(train_labels, exist_ok=True)
    os.makedirs(val_images, exist_ok=True)
    os.makedirs(val_labels, exist_ok=True)
    os.makedirs(test_images, exist_ok=True) 
    os.makedirs(test_labels, exist_ok=True)

    image_names = os.listdir(source_images)
    random.shuffle(image_names)
    train_size = int(len(image_names) * train_ratio)
    test_size = int(len(image_names) * test_ratio)
    val_size = len(image_names) - train_size - test_size
    train_names = image_names[:train_size]
    val_names = image_names[train_size:train_size + val_size]
    test_names = image_names[train_size + val_size:]

    for name in train_names:
        # print(f'{name} is in the train_images')
        image_path = os.path.join(source_images, name)
        label_path = os.path.join(source_labels, name[:-4] + '.txt')
        shutil.copy(image_path, train_images)
        shutil.copy(label_path, train_labels)

    for name in val_names:
        # print(f'{name} is in the val_images')
        image_path = os.path.join(source_images, name)
        label_path = os.path.join(source_labels, name[:-4] + '.txt')
        shutil.copy(image_path, val_images)
        shutil.copy(label_path, val_labels)

    for name in test_names:
        # print(f'{name} is in the test_images')
        image_path = os.path.join(source_images, name)
        label_path = os.path.join(source_labels, name[:-4] + '.txt')
        shutil.copy(image_path, test_images)
        shutil.copy(label_path, test_labels)