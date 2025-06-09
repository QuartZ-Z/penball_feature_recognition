import cv2
import numpy as np
import matplotlib.pyplot as plt

def fourier_transform(image_path):
    # 读取图像并转换为灰度图
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 计算傅里叶变换
    f = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f)  # 移动低频到中心
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))  # 幅值取对数，便于可视化

    # 可视化原图和频谱
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray')
    plt.title('Input Image'), plt.axis('off')
    plt.subplot(1, 2, 2), plt.imshow(magnitude_spectrum)
    plt.title('Magnitude Spectrum'), plt.axis('off')
    plt.show()

# 调用函数
fourier_transform("2.jpg")