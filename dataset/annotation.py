import cv2
import os
import shutil

image_folder = 'images/'
label_folder = 'labels/'
visualization_folder = 'visualizations/'
annotation = []
image = None
image_now = None
rect_flag = False
x1, y1, x2, y2 = -1, -1, -1, -1

clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
scale = 0.2

def on_mouse(event, x, y, flags, params):
    global x1, y1, x2, y2, rect_flag, image_now
    if event == cv2.EVENT_LBUTTONDOWN:
        if rect_flag == False:
            rect_flag = True
            x1 = x
            y1 = y
            # print(x1, y1, x2, y2)
        else:
            rect_flag = False
            x2 = x
            y2 = y
            # print(x1, y1, x2, y2)
            cv2.rectangle(image_now, (x1, y1), (x2, y2), (0, 255, 0), 2)
            image_show = image_now.copy()
            cv2.imshow('image', image_show)

def annotate(image_path, label_path, visualization_path = None):
    global annotation, image_now, image, rect_flag, x1, y1, x2, y2
    print(f'Annotating {image_path}')
    annotation = []
    rect_flag = False
    image = cv2.imread(image_path)
    # image_process = cv2.cvtColor(clahe.apply(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR)
    image_process = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    image_now = image_process.copy()
    x1, y1, x2, y2 = -1, -1, -1, -1
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse)
    while True:
        image_show = image_now.copy()
        cv2.imshow('image', image_show)
        key = cv2.waitKey(0)
        if key == ord('q'):
            print('Quitting annotation')
            exit(0)
        elif key == ord('c'):
            print('Clearing annotation')
            annotation = []
            image_now = image_process.copy()
            x1, y1, x2, y2 = -1, -1, -1, -1
            rect_flag = False
        elif ord('0') <= key <= ord('9'):
            print('Try adding annotation')
            if rect_flag == False and x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1:
                x = (x1 + x2) // 2
                y = (y1 + y2) // 2
                w = abs(x1 - x2)
                h = abs(y1 - y2)
                annotation.append((key - ord('0'), x, y, w, h))
                text_pos = (x, y)
                cv2.putText(image_now, str(key - ord('0')), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                x1, y1, x2, y2 = -1, -1, -1, -1
        elif key == ord('s'):
            print('Saving annotation')
            width, height = image_process.shape[1], image_process.shape[0]
            with open(label_path, 'w') as f:
                for label, x, y, w, h in annotation:
                    f.write(f'{label} {x / width} {y / height} {w / width} {h / height}\n')
            cv2.imwrite(visualization_path, image_now)
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    shutil.rmtree(visualization_folder, ignore_errors=True)
    shutil.rmtree(label_folder, ignore_errors=True)
    os.makedirs(visualization_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)

    image_names = os.listdir(image_folder)
    for name in image_names:
        if name.lower().endswith('.jpg') or name.lower().endswith('.png'):
            image_path = os.path.join(image_folder, name)
            label_path = os.path.join(label_folder, name[:-4] + '.txt')
            visualization_path = os.path.join(visualization_folder, name)
            annotate(image_path, label_path, visualization_path)