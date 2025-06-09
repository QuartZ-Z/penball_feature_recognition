import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def adaptive_retinex(image, sigma=100):
    """
    自适应Retinex算法
    :param image: 输入图像 (BGR 格式)
    :param sigma: 高斯滤波器标准差
    :return: 增强后的图像
    """
    # 转换到灰度空间或分离颜色通道
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # 估计照明分量
    illumination = cv2.GaussianBlur(image, (0, 0), sigma)

    # 计算反射分量（取对数避免数值问题）
    reflection = np.log1p(image) - np.log1p(illumination)

    # 归一化到 [0, 1]
    enhanced = cv2.normalize(reflection, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # 转换回 [0, 255] 并返回
    enhanced = (enhanced * 255).astype(np.uint8)
    print(enhanced.shape)
    return enhanced

# 加载图像
input_image = cv2.imread("input5.jpg")
enhanced_image = adaptive_retinex(input_image)

# 显示结果
input_image_pil = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
enhanced_image_pil = Image.fromarray(cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB))
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(input_image_pil)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(enhanced_image_pil)
plt.axis("off")
plt.show()