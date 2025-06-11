import numpy as np
import pickle
from typing import Optional, Callable


class LinearRegressionCustom:
    def __init__(self, fit_intercept=True, regularization=None, reg_strength=0.01, loss: Optional[Callable] = None):
        """
        :param fit_intercept: 是否拟合截距
        :param regularization: None/'l2'/'l1'
        :param reg_strength: 正则化系数
        :param loss: 自定义损失函数，输入(y_true, y_pred)，输出损失float
        """
        self.fit_intercept = fit_intercept
        self.regularization = regularization
        self.reg_strength = reg_strength
        self.loss = loss
        self.coef_ = None  # 权重
        self.intercept_ = None  # 截距

    def _add_intercept(self, X):
        if self.fit_intercept:
            intercept = np.ones((X.shape[0], 1))
            return np.hstack([intercept, X])
        return X

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        X_ = self._add_intercept(X)
        n, d = X_.shape

        # 普通最小二乘
        if self.regularization is None:
            # OLS解析解
            beta = np.linalg.pinv(X_.T @ X_) @ X_.T @ y
        elif self.regularization == 'l2':
            # 岭回归（L2正则化）
            I = np.eye(d)
            if self.fit_intercept:
                I[0, 0] = 0  # 截距不正则
            beta = np.linalg.pinv(X_.T @ X_ + self.reg_strength * I) @ X_.T @ y
        elif self.regularization == 'l1':
            # L1正则需用优化算法（这里只做简单的梯度下降）
            from sklearn.linear_model import Lasso
            model = Lasso(alpha=self.reg_strength, fit_intercept=self.fit_intercept)
            model.fit(X, y.ravel())
            self.coef_ = model.coef_
            self.intercept_ = model.intercept_
            return self
        else:
            raise ValueError("Unknown regularization: %s" % self.regularization)

        if self.fit_intercept:
            self.intercept_ = beta[0, 0]
            self.coef_ = beta[1:, 0]
        else:
            self.intercept_ = 0
            self.coef_ = beta[:, 0]
        return self

    def predict(self, X):
        X = np.array(X)
        if self.fit_intercept:
            return X @ self.coef_ + self.intercept_
        else:
            return X @ self.coef_

    def score(self, X, y, metric='r2'):
        y_pred = self.predict(X)
        y_true = np.array(y)
        if metric == 'r2':
            u = ((y_true - y_pred) ** 2).sum()
            v = ((y_true - y_true.mean()) ** 2).sum()
            return 1 - u / v
        elif metric == 'mae':
            return np.mean(np.abs(y_true - y_pred))
        elif metric == 'rmse':
            return np.sqrt(np.mean((y_true - y_pred) ** 2))
        elif self.loss:
            return self.loss(y_true, y_pred)
        else:
            raise ValueError("Unknown metric")

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.__dict__.update(pickle.load(f))
        return self
