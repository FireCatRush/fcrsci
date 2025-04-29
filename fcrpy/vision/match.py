import cv2
import numpy as np


def template_matching_cv(source, template, method=cv2.TM_CCOEFF_NORMED, threshold=0.8):
    """
    基于OpenCv的模板图像匹配
    Args:
        source (array-like): 要搜索的大图像的路径
        template (array-like): 模板图像的路径
        method (int): 匹配方法，默认为cv2.TM_CCOEFF_NORMED
            可选值:
                cv2.TM_CCOEFF: 相关系数匹配法
                cv2.TM_CCOEFF_NORMED: 归一化相关系数匹配法
                cv2.TM_CCORR: 相关匹配法
                cv2.TM_CCORR_NORMED: 归一化相关匹配法
                cv2.TM_SQDIFF: 平方差匹配法
                cv2.TM_SQDIFF_NORMED: 归一化平方差匹配法
        threshold (float): 匹配阈值，范围0-1，只返回匹配度高于此值的结果
    Returns:
        list: 包含所有匹配到的位置，每个位置为(x, y, w, h, confidence)，
              其中(x, y)是左上角坐标，(w, h)是模板尺寸，confidence是匹配度
    """

    if source is None or template is None:
        raise ValueError("无法读取图像或模板")

    h, w = template.shape[:2]
    result = cv2.matchTemplate(source, template, method)

    # 根据匹配方法确定如何解释结果
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        # 对于这些方法，较小的值表示更好的匹配
        result = 1 - result

    locations = []
    loc = np.where(result >= threshold)

    # 去除重叠的匹配结果
    matched_indices = set()
    for pt in zip(*loc[::-1]):  # 反转坐标为 (x, y)
        overlapped = False
        for idx, (x, y, _, _, _) in enumerate(locations):
            # 检查是否与已有匹配位置重叠
            if abs(pt[0] - x) < w // 2 and abs(pt[1] - y) < h // 2:
                # 如果新匹配的置信度更高，替换旧的
                if result[pt[1], pt[0]] > result[y, x] and idx not in matched_indices:
                    locations[idx] = (pt[0], pt[1], w, h, float(result[pt[1], pt[0]]))
                    matched_indices.add(idx)
                overlapped = True
                break

        if not overlapped:
            locations.append((pt[0], pt[1], w, h, float(result[pt[1], pt[0]])))

    return locations