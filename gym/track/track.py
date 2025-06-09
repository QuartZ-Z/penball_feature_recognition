import cv2
import numpy as np
import skimage
import scipy

size = 512
scale = 1.0
radius = 136 # 0.7 mm ~ 136 pixel
area_threshold = 1000

def resize_image(image, target_size):
    height, width = image.shape[:2]
    scale = target_size / min(height, width)
    new_height = int(height * scale)
    new_width = int(width * scale)
    resized_image = cv2.resize(image, (new_width, new_height))
    print(f"Resized image to: {new_height}x{new_width}")
    return resized_image, scale


def format_image(img):
    img_guassian = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(img_guassian, cv2.COLOR_BGR2HSV)
    Canny = cv2.Canny(hsv, 50, 100)

    cv2.imshow('Canny', Canny)
    cv2.imwrite('Canny.jpg', Canny)
    # cv2.waitKey(0)
    
    mask = Canny.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.dilate(mask, kernel, iterations = 2)
    mask = cv2.erode(mask, kernel, iterations = 5)
    mask = cv2.dilate(mask, kernel, iterations = 3)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    small_components = np.where(stats[:, cv2.CC_STAT_AREA] < area_threshold)[0]
    for component in small_components:
        mask[labels == component] = 0

    mask = cv2.bitwise_not(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    small_components = np.where(stats[:, cv2.CC_STAT_AREA] < area_threshold)[0]
    for component in small_components:
        mask[labels == component] = 0

    mask = cv2.bitwise_not(mask)
    cv2.imshow('mask', mask)
    cv2.imwrite('mask.jpg', mask)

    skeleton = skimage.morphology.medial_axis(mask)
    skeleton = skeleton.astype(np.uint8) * 255
    cv2.imshow('skeleton', skeleton)
    cv2.imwrite('skeleton.jpg', skeleton)
    return skeleton

def get_artery(skl, x1, y1):
    print("x1, y1:", x1, y1)
    height, width = skl.shape
    dist = np.zeros((height, width), dtype=np.uint32)
    last = np.zeros((height, width, 2), dtype=np.int32)

    dist[y1, x1] = 1
    queue = [(x1, y1)]
    while queue:
        x, y = queue.pop(0)
        # print(x, y, dist[y, x])
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and skl[ny, nx] == 255 and dist[ny, nx] == 0:
                dist[ny, nx] = dist[y, x] + 1
                last[ny, nx] = (x, y)
                queue.append((nx, ny))
    
    farthest_point = np.unravel_index(np.argmax(dist), dist.shape)
    x2, y2 = farthest_point[1], farthest_point[0]
    print("farthest_point:", farthest_point)

    dist = np.zeros((height, width), dtype=np.uint32)
    last = np.zeros((height, width, 2), dtype=np.int32)

    dist[y2, x2] = 1
    queue = [(x2, y2)]
    while queue:
        x, y = queue.pop(0)
        # print(x, y, dist[y, x])
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and skl[ny, nx] == 255 and dist[ny, nx] == 0:
                dist[ny, nx] = dist[y, x] + 1
                last[ny, nx] = (x, y)
                queue.append((nx, ny))

    new_farthest_point = np.unravel_index(np.argmax(dist), dist.shape)
    x3, y3 = new_farthest_point[1], new_farthest_point[0]
    print("new_farthest_point:", new_farthest_point)

    path = []
    current = (x3, y3)
    while current != (x2, y2):
        # print(current)
        path.append(current)
        current = tuple(last[current[1], current[0]])
    path.append((x2, y2))
    path.reverse()
    return path, dist

def cut_skeleton(skl):
    height, width = skl.shape
    new_skl = np.zeros(skl.shape, dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            if skl[i, j] == 255:
                path, dist = get_artery(skl, j, i)
                for p in path:
                    new_skl[p[1], p[0]] = 255
                vis = np.where(dist > 0, 255, 0).astype(np.uint8)
                skl = cv2.bitwise_and(skl, cv2.bitwise_not(vis))
                # cv2.imshow('new_skeleton', new_skl)
                # cv2.waitKey(0)

    cv2.imshow('new_skeleton', new_skl)
    cv2.imwrite('new_skeleton.jpg', new_skl)

    return new_skl

def locate_root(skl, x, y): # 在骨架图像中找到距离 (x, y) 最近的白色像素
    # 获取所有白色像素的坐标
    white_pixels = np.column_stack(np.where(skl == 255))
    
    # 如果没有白色像素，返回 None
    if len(white_pixels) == 0:
        return None

    # 计算所有白色像素到 (x, y) 的欧几里得距离
    distances = np.sqrt((white_pixels[:, 0] - y) ** 2 + (white_pixels[:, 1] - x) ** 2)
    
    # 找到距离最小的白色像素
    nearest_index = np.argmin(distances)
    nearest_pixel = white_pixels[nearest_index]

    return nearest_pixel[1], nearest_pixel[0]  # 返回 (x, y) 坐标

def sample_skeleton(skl, x1, y1, x2, y2, step = 20): # 在(x1,y1)到(x2,y2)之间的白色像素八连通路径上每隔step像素采样1个点

    if skl[y1, x1] == 0 or skl[y2, x2] == 0:
        return None
    
    height, width = skl.shape
    dist = np.zeros((height, width), dtype=np.uint32)
    last = np.zeros((height, width, 2), dtype=np.int32)

    dist[y1, x1] = 1
    queue = [(x1, y1)]
    while queue:
        x, y = queue.pop(0)
        # print(x, y, dist[y, x])
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and skl[ny, nx] == 255 and dist[ny, nx] == 0:
                dist[ny, nx] = dist[y, x] + 1
                last[ny, nx] = (x, y)
                queue.append((nx, ny))
    
    if dist[y2, x2] == 0:
        return None
    else:
        print("dist:", dist[y2, x2])
        print("last:", last[y2, x2])

    # 反向追踪路径
    path = []
    current = (x2, y2)
    while current != (x1, y1):
        # print(current)
        path.append(current)
        current = tuple(last[current[1], current[0]])
    path.append((x1, y1))
    path.reverse()

    sample = [(x1, y1)]
    n = max(1, len(path) // step)
    for i in range(1, n - 1):
        index = int(i * step)
        if index < len(path):
            sample.append(path[index])
    sample.append((x2, y2))
    return sample

def get_rotate_matrix(path, r):
    result = np.eye(3)
    if len(path) < 2:
        return result
    
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        rot = np.zeros((3, 3))
        rot[0, 2] = -dx / r
        rot[1, 2] = -dy / r
        rot[2, 0] = dx / r
        rot[2, 1] = dy / r
        rot = scipy.linalg.expm(rot)
        result = np.dot(rot, result)

    return result
    

if __name__ == "__main__":
    img = cv2.imread('014.JPG')

    if img is None:
        print("Error: Unable to load image.")
        exit(1)

    img, scale = resize_image(img, size)
    skl = format_image(img)
    skl = cut_skeleton(skl)
    
    height, width = skl.shape
    x1 = 322
    y1 = 153
    x2 = 161
    y2 = 459
    
    root_x1, root_y1 = locate_root(skl, x1, y1)
    if root_x1 is not None and root_y1 is not None:
        cv2.circle(img, (x1, y1), 5, (0, 255, 0), -1)
        cv2.line(img, (x1, y1), (root_x1, root_y1), (0, 0, 255), 2)
        print(f"Root1 located at: ({root_x1}, {root_y1})")
    else:
        print("Root1 not found.")
        exit(1)

    root_x2, root_y2 = locate_root(skl, x2, y2)
    if root_x2 is not None and root_y2 is not None:
        cv2.circle(img, (x2, y2), 5, (0, 255, 0), -1)
        cv2.line(img, (x2, y2), (root_x2, root_y2), (0, 0, 255), 2)
        print(f"Root2 located at: ({root_x2}, {root_y2})")
    else:
        print("Root2 not found.")
        exit(1)

    sample = sample_skeleton(skl, root_x1, root_y1, root_x2, root_y2, 20)
    if sample is not None:
        for i in range(len(sample) - 1):
            cv2.line(img, sample[i], sample[i + 1], (255, 0, 0), 2)
    else:
        print("No path found.")
        exit(1)

    cv2.imshow('img', img)
    cv2.imwrite('result.jpg', img)

    matrix = get_rotate_matrix(sample, 271 * scale)
    print("Rotation matrix:", matrix)
    rot_vec = cv2.Rodrigues(matrix[:3, :3])[0]
    print("Rotation vector:", rot_vec)

    cv2.waitKey(0)
    cv2.destroyAllWindows()