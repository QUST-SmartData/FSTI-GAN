import cv2
import numpy as np

# 定义文件名变量
fileName = "099_2945.png"

ImagePATH = f"C:\\Users\\Toby\\Desktop\\test_compare_img\\DeepLesion_test_compare_img\\ablation_experiment_04600\\Ground_Truth\\{fileName}"
maskPATH = r"F:\Dataset\mask_04600\04600.png"
# 读取图像和掩码
image = cv2.imread(ImagePATH)  # 24位图像
mask = cv2.imread(maskPATH, cv2.IMREAD_GRAYSCALE)  # 8位灰度掩码

mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

# 将掩码的白色区域变成白色（255），黑色区域保持不变
white_areas = mask == 255
mask[white_areas] = 255

# 将处理后的掩码与图像相加
result = cv2.add(image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

# 保存结果，使用format来动态设置文件名称
cv2.imwrite(f"C:\\Users\\Toby\\Desktop\\test_compare_img\\DeepLesion_test_compare_img\\ablation_experiment_04600\\{fileName}", result)