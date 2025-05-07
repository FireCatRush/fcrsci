import numpy as np
from functools import lru_cache


def matmul_chain(*mats, cost="flops"):
    """
    矩阵链乘法原型函数(不考虑非标准矩阵链，不做排序), 利用交换律和动态规划减少矩阵计算过程中的临时内存
    ------------------
    Args:
        *mats: array_like 至少两个密集的2D NumPy数组，它们的内部维度必须匹配。
        cost:  "flops", "memory", "flops+memory"}, 可选
                要最小化的度量标准:
                    flops  -> Σ m·n·k     (经典方法，最小化浮点运算次数)
                    memory -> max(temp_size) 在链乘过程中的最大临时空间
                    flops+memory -> flops · temp_size  (粗略的联合度量)

    Returns: np.ndarray 输入矩阵的乘积，按最优顺序计算。

    Example:
    >>> A = np.random.rand(10, 30)  # 10×30
    >>> B = np.random.rand(30, 5)  # 30×5
    >>> C = np.random.rand(5, 60)  # 5×60

    >>> result = matmul_chain(A, B, C)
    手动追踪算法执行过程:
    1. 首先收集维度:
    dims = [10, 30, 5, 60]

    2. DP求解最优顺序:
    best(0, 0) = (0, 0, None)  # 单个矩阵A
    best(1, 1) = (0, 0, None)  # 单个矩阵B
    best(2, 2) = (0, 0, None)  # 单个矩阵C

    best(0, 1) 计算A和B的最优乘法:
      k=0: 10×30×5 = 1500个浮点操作，结果大小 = 10×5 = 50
      返回 (1500, 50, 0)

    best(1, 2) 计算B和C的最优乘法:
      k=1: 30×5×60 = 9000个浮点操作，结果大小 = 30×60 = 1800
      返回 (9000, 1800, 1)

    best(0, 2) 计算A, B, C的最优乘法:
      k=0: cost(A) + cost(B×C) + cost((B×C)×A)
           = 0 + 9000 + 10×30×60 = 27000
      k=1: cost(A×B) + cost(C) + cost((A×B)×C)
           = 1500 + 0 + 10×5×60 = 4500
      k=1更优，返回 (4500, 600, 1)

    3. 递归执行乘法:
    multiply(0, 2) -> multiply(0, 1) @ multiply(2, 2)
                    -> (A @ B) @ C
    """

    if len(mats) < 2:
        raise ValueError("需要至少两个矩阵")

    # 1. 收集维度信息
    # 构成[第一个矩阵的行 + 所有矩阵的列]的列表
    # 在矩阵乘法中，如果要将两个矩阵A和B相乘，A的列数必须等于B的行数。假设A是m×n的矩阵，B是n×p的矩阵，那么结果C=A×B是一个m×p的矩阵。
    # 这里不考虑乱序的方法
    # 假设我们有三个矩阵：
    # A: 10×30 的矩阵
    # B: 30×5 的矩阵
    # C: 5×60 的矩阵
    # 得到[10, 30, 5, 60] 代表 10x30, 30x5, 5x60的维度信息
    dims = [mats[0].shape[0]] + [m.shape[1] for m in mats]

    # 2. 使用动态规划找到最小成本
    @lru_cache(maxsize=None)
    def best(i, j):
        """
        计算矩阵链 A_i 到 A_j 的最优括号方式。找出所有可能的括号组合中，使得总计算成本最小的方案。
        返回(成本, 临时维度大小, 分割点k)。
        i和j是列表中任意的两个矩阵索引，表示计算矩阵A_i到A_j的最优乘法顺序。
        """
        if i == j:
            return 0, 0, None

        # best_cost: 依据成本度量进行相应的cost计算，例如，浮点运算次数总和，最大内存使用量等，初始一个无穷大的数
        # best_temp: 存储当前最优方案中需要的临时存储空间的大小。表示计算过程中产生的最大中间结果矩阵的元素总数。
        # besk_k: 存储实现最优解的分割点的位置
        best_cost, best_temp, best_k = float("inf"), None, None
        for k in range(i, j):
            # 在k处分割矩阵链，分别计算A_i到A_k的最优解和计算A_(k+1)到A_j的最优解
            c1, t1, _ = best(i, k)
            c2, t2, _ = best(k + 1, j)

            # 获取两个结果矩阵的维度，从矩阵乘法规则来说，结果矩阵的维度等于第一个矩阵的行x最后一个矩阵的列
            # 维度说明:
            # dims[k + 1]既是上一个矩阵链的列，又是下一个矩阵链的行
            # A_i..k 的形状是 dims[i] × dims[k+1]
            # A_k+1..j 的形状是 dims[k+1] × dims[j+1]
            m, n, p = dims[i], dims[k + 1], dims[j + 1]

            # 计算当前划分的浮点运算数量：A_i..k 和 A_k+1..j 的乘法运算量
            this_flops = m * n * p
            # 计算当前划分需要的临时存储空间
            # 左半链的大小，右半链的大小，结果矩阵的大小
            this_temp = max(t1, t2, m * p)

            # 根据选择的成本类型计算指标
            if cost == "flops":
                # 最小化浮点运算次数
                metric = c1 + c2 + this_flops
            elif cost == "memory":
                # 最小化临时内存使用量，减少中间矩阵的大小通常也会减少总的计算量
                metric = max(this_temp, t1, t2)
            else:  # 联合度量
                metric = (c1 + c2 + this_flops) * this_temp

            # 更新最优解
            if metric < best_cost:
                best_cost, best_temp, best_k = metric, this_temp, k
        return best_cost, best_temp, best_k

    # 3. 递归执行最优顺序的矩阵乘法
    def multiply(i, j):
        """
        递归执行矩阵A_i到A_j的乘法，按照最优括号顺序。
        """
        if i == j:  # 相同矩阵索引 => 输入只有一个矩阵 => 可以在递归中执行乘法
            return mats[i]
        # 获取最优分割点
        _, _, k = best(i, j)
        # 递归计算左半部分和右半部分，然后相乘
        # @ 是Python 3.5+中的矩阵乘法运算符
        return multiply(i, k) @ multiply(k + 1, j)

    # 从整个矩阵链开始计算
    return multiply(0, len(mats) - 1)