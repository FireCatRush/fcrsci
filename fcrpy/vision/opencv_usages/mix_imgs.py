import os
import cv2
import numpy as np


def mix_imgs(imgs, alphas):
    if isinstance(imgs, np.ndarray):
        raise TypeError("请将单个图像传入为图像集合（列表或元组）")

    if not isinstance(imgs, (list, tuple)):
        raise TypeError("imgs 应该是列表或元组类型")

    if len(imgs) == 0:
        raise ValueError("imgs 不能为空")

    # 如果 alphas 是单一元素，重复该元素以匹配图片数量
    if isinstance(alphas, (int, float)):
        alphas = [alphas] * len(imgs)
    elif len(alphas) != len(imgs):
        raise ValueError("alphas 的长度必须与 imgs 的长度相同，或为单一元素")

    # 检查每张图片的尺寸是否一致
    img_shape = None
    decoded_imgs = []
    for img, alpha in zip(imgs, alphas):
        # 处理输入是路径的情况
        if isinstance(img, str) and os.path.isfile(img):
            # 如果是路径，尝试判断文件名的语言（判断是否包含非英文字符）
            if any(ord(char) > 127 for char in img):
                img_data = cv2.imdecode(np.fromfile(img, dtype=np.uint8), cv2.IMREAD_COLOR)
            else:
                img_data = cv2.imread(img, cv2.IMREAD_COLOR)
            decoded_imgs.append(img_data)
        elif isinstance(img, np.ndarray):
            decoded_imgs.append(img)
        else:
            raise TypeError("imgs 中的元素应为图像路径或 NumPy 数组")

        # 检查所有图像的尺寸是否一致
        if img_shape is None:
            img_shape = decoded_imgs[-1].shape
        elif decoded_imgs[-1].shape != img_shape:
            raise ValueError("所有图片的尺寸必须相同")

    # 初始化结果图像
    dst = np.zeros_like(img_shape, dtype=np.float32)

    # 计算混合结果
    for img, alpha in zip(decoded_imgs, alphas):
        beta = 1 - alpha
        # 按像素加权混合
        # dst += img.astype(np.float32) * alpha
        dst = np.add(dst, img.astype(np.float32) * alpha)
    # 将结果归一化至[0, 255]范围
    dst = np.clip(dst, 0, 255).astype(np.uint8)

    return dst
