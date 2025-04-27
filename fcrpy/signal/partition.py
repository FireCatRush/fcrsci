import numpy as np

def interval_optimal_partition_dp(data, fitness_func):
    """
    动态规划寻找数据在区间上的最优划分算法, 时间复杂度O(n^2)
    source from:
        "An Algorithm for Optimal Partitioning of Data on an Interval":
        https://ieeexplore.ieee.org/abstract/document/1381461
    说明:
        在一组区间I内的N个数据的数组中，划分了M个区块B_M, P代表Bm的集合。不同组件不能有间隙，区块数量满足0<=M<=N。
        定义一个加性适应度函数g(Bm)(理解为损失函数)分布在P的任何分区中, 使用动态规划寻找最优划分。依据论文涉及Scargle的
        贝叶斯适应度模型似乎是有效的(A New Method to Analyze Structure in Photon Counting Data)。

    Args:
        data (array-like): 一组一维数据
        fitness_func (callback): 适应度函数

    Returns:
        最优分割的块（列表的列表）和相应的最大适应度值
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError(f"Expected a 1-dimensional array, but got {data.ndim}-dimensional array.")

    N = len(data)
    opt = [0] * (N + 1)
    lastchange = [0] * (N + 1)

    def block_fitness(j, n):
        return fitness_func(data[j - 1:n])

    opt[0] = 0
    for n in range(1, N + 1):
        max_fitness = float('-inf')
        best_j = 0

        for j in range(1, n + 1):
            current_fitness = opt[j - 1] + block_fitness(j, n)

            if current_fitness > max_fitness:
                max_fitness = current_fitness
                best_j = j

        opt[n] = max_fitness
        lastchange[n] = best_j

    blocks = []
    n = N

    while n > 0:
        j = lastchange[n]
        blocks.insert(0, data[j - 1:n])
        n = j - 1

    return blocks, opt[N]