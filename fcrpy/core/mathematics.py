import numpy as np

def gaussian_pdf(x, mean, std):
    """正态分布函数"""
    p = (1 / (std * np.sqrt(2 * np.pi)))
    return p * np.exp(-(x - mean) ** 2 / (2 * std ** 2))
