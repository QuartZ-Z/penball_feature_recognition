from ultralytics import YOLO
import torch
import os
import numpy as np
import cv2
from torchmetrics.detection.mean_ap import MeanAveragePrecision as MAP

gpu_count = torch.cuda.device_count()
gpu_list = [i for i in range(torch.cuda.device_count())]

image_folder = 'test_images/_phone_white_a4/test/images'
model_path = 'best_classification.pt'
# image_width = 1360
# image_height = 1024

def xywh2xyxy(xywh):
    x1 = xywh[0] - xywh[2] / 2
    y1 = xywh[1] - xywh[3] / 2
    x2 = xywh[0] + xywh[2] / 2
    y2 = xywh[1] + xywh[3] / 2
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def load_image(data_folder):
    file_list = os.listdir(data_folder)
    image_list = []
    for file in file_list:
        if file.lower().endswith('.jpg') or file.lower().endswith('.png'):
            image_path = os.path.join(data_folder, file)
            image_list.append(image_path)
            print(f"Loaded image: {image_path}")
    return image_list

def detect(model_path, image_list):
    print(model_path, image_list)
    model = YOLO(model = model_path)
    results = model.predict(source = image_list, imgsz = 640, device = gpu_list)
    
    box_detection = []
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        box_count = len(boxes)
        print(f"Detected {box_count} boxes in image.")
        current_boxes = {
            'boxes': torch.zeros((box_count, 4), dtype=torch.float32),
            'scores': torch.zeros((box_count), dtype=torch.float32),
            'labels': torch.zeros((box_count), dtype=torch.int64)
        }
        for i, box in enumerate(boxes):
            current_boxes['boxes'][i] = box.xyxyn[0]
            current_boxes['scores'][i] = box.conf
            current_boxes['labels'][i] = box.cls
        box_detection.append(current_boxes)
    
    return box_detection

def load_labels(label_path):
    with open(label_path, 'r') as file:
        lines = file.readlines()
    
    current_boxes = {
        'boxes': torch.zeros((len(lines), 4), dtype=torch.float32),
        'scores': torch.zeros((len(lines)), dtype=torch.float32),
        'labels': torch.zeros((len(lines)), dtype=torch.int64)
    }

    for i, line in enumerate(lines):
        parts = line.strip().split()
        current_boxes['labels'][i] = int(parts[0])  # Class ID
        xywhn = np.array([float(x) for x in parts[1:5]], dtype=np.float32)
        current_boxes['boxes'][i] = torch.tensor(xywh2xyxy(xywhn))
        current_boxes['scores'][i] = 1.0  # Default score

    return current_boxes

# def work(box_detection):
#     count = len(box_detection['boxes'])
#     for i in range(count):
#         for j in range(i + 1, count):
#             cls1 = int(box_detection['labels'][i].item())
#             conf1 = float(box_detection['scores'][i].item())
#             xywh1 = box_detection['boxes'][i].numpy()
#             cls2 = int(box_detection['labels'][j].item())
#             conf2 = float(box_detection['scores'][j].item())
#             xywh2 = box_detection['boxes'][j].numpy()
#             x1 = xywh1[0]
#             y1 = xywh1[1]
#             size1 = np.sqrt(xywh1[2] ** 2 + xywh1[3]  ** 2)
#             x2 = xywh2[0]
#             y2 = xywh2[1]
#             size2 = np.sqrt(xywh2[2] ** 2 + xywh2[3]  ** 2)
#             dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#             print(f"Box1: {cls1}, {conf1}, ({x1}, {y1}, {size1})")
#             print(f"Box2: {cls2}, {conf2}, ({x2}, {y2}, {size2})")
#             print(f"Distance: {dist} | k: {dist / std_dist}")

def visualize_boxes(image_path, box_detection):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    for i in range(len(box_detection['boxes'])):
        box = box_detection['boxes'][i].numpy()
        cls = int(box_detection['labels'][i].item())
        score = float(box_detection['scores'][i].item())
        x1, y1, x2, y2 = map(float, box)
        cv2.rectangle(image, (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height)), (0, 255, 0), 2)
        cv2.putText(image, f'{cls}', (int(x1 * width), int(y1 * height - 10)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

    scale = 0.25
    img_show = cv2.resize(image, (int(width * scale), int(height * scale)))
    cv2.imshow('Detected Boxes', img_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_list = load_image(image_folder)
    label_list = [image[:-4].replace('/images', '/labels') + '.txt' for image in image_list]

    detect_box_lists = detect(model_path, image_list)
    label_box_lists = [load_labels(label) for label in label_list]

    print(detect_box_lists)
    print(label_box_lists)

    n = len(detect_box_lists)
    # for i in range(n):
    #     print(f"Image {i + 1}:")
    #     box_list = detect_box_lists[i]
    #     if len(box_list) > 0:
    #         work(box_list)
    #     else:
    #         print("No boxes detected.")
    MAP_obj = MAP()

    for i in range(n):
        MAP_obj.update([detect_box_lists[i]], [label_box_lists[i]])
    print(MAP_obj.compute())

    # for i in range(n):
    #     print(f"Visualizing image {i + 1}:")
    #     visualize_boxes(image_list[i], detect_box_lists[i])