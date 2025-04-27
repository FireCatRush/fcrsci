import numpy as np


def mean_squared_error(y_true, y_pred) -> float:
    """
    计算均方误差（MSE）
    Args:
        y_true (array-like): 真实标签
        y_pred (array-like): 预测标签
    Returns:
        float: 均方误差
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")

    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred) -> float:
    """
    计算平均绝对误差（MAE）
    Args:
        y_true (array-like): 真实标签
        y_pred (array-like): 预测标签
    Returns:
        float: 平均绝对误差
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")

    return np.mean(np.abs(y_true - y_pred))


def gaussian_negative_log_likelihood(y_true, y_pred_mean, y_pred_std, eps: float = 1e-6) -> float:
    """
    负高斯对数似然（Gaussian Negative Log-Likelihood）
    Args:
        y_true (array-like): 真实标签
        y_pred_mean (array-like): 预测的均值
        y_pred_std (array-like): 预测的标准差
        eps (float): 为了数值稳定性，加到标准差上的小常数
    Returns:
        float: 高斯负对数似然（取均值）
    """
    y_true = np.asarray(y_true)
    y_pred_mean = np.asarray(y_pred_mean)
    y_pred_std = np.asarray(y_pred_std)

    if not (y_true.shape == y_pred_mean.shape == y_pred_std.shape):
        raise ValueError(f"Shape mismatch: shapes are {y_true.shape}, {y_pred_mean.shape}, {y_pred_std.shape}")

    var = (y_pred_std + eps) ** 2  # 防止标准差为0
    nll = 0.5 * np.log(2 * np.pi * var) + 0.5 * ((y_true - y_pred_mean) ** 2) / var
    return np.mean(nll)