from ultralytics import YOLO
import torch
import os

gpu_count = torch.cuda.device_count()
gpu_list = [i for i in range(torch.cuda.device_count())]

if __name__ == '__main__':
    model = YOLO(model = 'yolo11n.pt')
    dataset_name = 'train_100g-white-a4-circle_augmented'
    augment = False
    train_id = 101

    box_weight = 5
    cls_weight = 1
    dfl_weight = 2
    epochs = 100
    dropout = 0.2
    # if augment == True:
    #     hsv_hue = 0.015
    #     hsv_saturation = 0.7
    #     hsv_value = 0.4
    #     translate = 0.1
    #     scale = 0.5
    #     fliplr = 0.5
    #     mosaic = 0.5
    # else:
    #     hsv_hue = 0
    #     hsv_saturation = 0
    #     hsv_value = 0
    #     translate = 0
    #     scale = 0
    #     fliplr = 0
    #     mosaic = 0

    results = model.train(data = "datasets/_" + dataset_name + "/data.yaml", epochs = epochs, imgsz = 640,
        device = gpu_list, save_period = 10,
        project = 'runs/' + dataset_name,
        name = 'train_' + str(train_id) + '_epoch' + str(epochs) + '_dropout' + str(dropout), exist_ok = True,
        # hsv_h = hsv_hue, hsv_s = hsv_saturation, hsv_v = hsv_value,
        # translate = translate, scale = scale, fliplr = fliplr, mosaic = mosaic,
        box = box_weight, cls = cls_weight, dfl = dfl_weight, dropout = dropout,
        batch = 4, workers = 4, lr0 = 1e-3, momentum = 0.937, optimizer = 'Adam')

