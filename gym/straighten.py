import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "input1.jpg"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. 图像预处理
# 增强对比度
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)

# 阈值分割
_, binary = cv2.threshold(enhanced, 50, 255, cv2.THRESH_BINARY_INV)

# 3. 提取轮廓
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 找到面积最大的轮廓（假设是目标条带）
contour = max(contours, key=cv2.contourArea)

# 创建空白图像用于可视化轮廓
mask = np.zeros_like(binary)
cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

# 提取条带的骨架
skeleton = cv2.ximgproc.thinning(mask)

# 4. 骨架路径拟合
# 获取骨架非零点坐标
points = np.column_stack(np.where(skeleton > 0))

# 通过x坐标排序生成平滑路径
sorted_points = points[np.argsort(points[:, 1])]
smooth_path = cv2.approxPolyDP(sorted_points, epsilon=2, closed=False)

# 5. 拉直变换
h, w = mask.shape
output_width = 500  # 拉直后的宽度
output_height = 50  # 拉直后的高度

# 生成目标直线
straight_line = np.array([[i, output_height // 2] for i in range(output_width)], dtype=np.float32)

# 映射变换
M = cv2.estimateAffinePartial2D(smooth_path[:, 0, :], straight_line)[0]
straightened = cv2.warpAffine(image, M, (output_width, output_height), flags=cv2.INTER_LINEAR)

# 可视化结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image[:, :, ::-1])
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(straightened[:, :, ::-1])
plt.title("Straightened Image")
plt.axis("off")
plt.show()