from ultralytics import YOLO
import torch
import os
import numpy as np
import math
import cv2

gpu_count = torch.cuda.device_count()
gpu_list = [i for i in range(torch.cuda.device_count())]

image_folder = 'calc_images/straight/'
model_path = 'best_circle_augmented.pt'

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
            current_boxes['boxes'][i] = box.xywh[0]
            current_boxes['scores'][i] = box.conf
            current_boxes['labels'][i] = box.cls
        box_detection.append(current_boxes)
    
    return box_detection

def visualize_boxes(image_path, box_detection):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    for i in range(len(box_detection['boxes'])):
        box = box_detection['boxes'][i].numpy()
        cls = int(box_detection['labels'][i].item())
        score = float(box_detection['scores'][i].item())
        x1 = box[0] - box[2] / 2
        y1 = box[1] - box[3] / 2
        x2 = box[0] + box[2] / 2
        y2 = box[1] + box[3] / 2
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f'Class: {cls}, Score: {score:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    scale = 0.5
    img_show = cv2.resize(image, (int(width * scale), int(height * scale)))
    cv2.imshow('Detected Boxes', img_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

px_to_mm = 2.5735e-3  # Conversion factor from pixels to mm

if __name__ == "__main__":
    image_list = load_image(image_folder)
    box_detection = detect(model_path, image_list)
    
    gamma_list = []
    for i, boxes in enumerate(box_detection):
        print(f"Image {i+1}: {len(boxes['boxes'])} boxes detected")
        visualize_boxes(image_list[i], boxes)
        if len(boxes['boxes']) >= 2:
            box1 = boxes['boxes'][0].numpy()
            box2 = boxes['boxes'][1].numpy()
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            
            # Calculate the distance in pixels
            distance_px = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Convert to mm
            distance_mm = distance_px * px_to_mm
            
            # print(f"Distance between first two boxes in image {i+1}: {distance_mm:.3f} mm")

            gamma_list.append(distance_mm / (math.pi * 0.7))

    print("Gamma values:", gamma_list)
    print("Average Gamma:", np.mean(gamma_list))
    print("Standard Deviation of Gamma:", np.std(gamma_list))