import os

data_folder = '_phone_white_a4/'
train_label_folder = data_folder + 'train/labels/'
val_label_folder = data_folder + 'val/labels/'
test_label_folder = data_folder + 'test/labels/'

def count_labels(label_folder):
    label_list = os.listdir(label_folder)
    total = 0
    for label_file in label_list:
        label_path = os.path.join(label_folder, label_file)
        with open(label_path, 'r', encoding='utf-8') as f:
            num_lines = sum(1 for _ in f)
            # print(f"{label_file}: {num_lines} lines")
            total += num_lines
    print(f"Total number of files in {label_folder}: {len(label_list)}")
    print(f"Total number of labels in {label_folder}: {total}")
    return total

count_labels(train_label_folder)
count_labels(val_label_folder)
count_labels(test_label_folder)