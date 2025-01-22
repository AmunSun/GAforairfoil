import numpy as np
import random
import matplotlib.pyplot as plt
from .solver import solver
from .plotting import plotairfoil

def randp(p, range_values):
    """
    此函数用于遗传算法（GA）的特殊随机化函数，目的是在给定的范围 `range_values` 内为输入向量 `p` 生成随机的个体（新的参数向量）。

    参数:
    p (list or np.ndarray): 输入向量，通常表示遗传算法中的个体参数。
    range_values (list or np.ndarray): 用于指定随机化的范围。

    返回:
    np.ndarray: 经过随机化处理后的个体参数向量。
    """
    p = np.array(p)
    range_values = np.array(range_values)

    if len(range_values) == len(p):
        # 当 range_values 的长度与 p 的长度相等时
        new_p = []
        for i in range(len(p)):
            # 在以 p[i] 为中心，范围为 2*range_values[i] 的区间内生成一个随机数
            new_value = 2 * range_values[i] * np.random.rand() + p[i] - range_values[i]
            new_p.append(new_value)
        p = np.array(new_p)
    elif len(range_values) == 2 * len(p):
        # 当 range_values 的长度是 p 的长度的两倍时
        new_p = []
        for i in range(len(p)):
            # 随机范围是从 range_values[2*i] 到 range_values[2*i + 1]
            new_value = (range_values[2*i + 1] - range_values[2*i]) * np.random.rand() + range_values[2*i]
            new_p.append(new_value)
        p = np.array(new_p)

    return p

def GAairfoil(genNo, p0, range_value, uinf, AOA, Npanel):
    """
    此函数基于升力系数值优化翼型形状。
    :param genNo: 交配的代数
    :param p0: 要优化的原始翼型
    :param range_value: 用于改变PARSEC参数的随机化范围
    :param uinf: 流动自由流速度
    :param AOA: 翼型攻角
    :param Npanel: 适应度函数的面板数量
    :return: 原始升力系数、最适合的升力系数、最适合的个体
    """
    # 计算原始升力系数
    cloriginal, _ = solver(p0, uinf, AOA, Npanel)
    # 遗传参数
    popsize = 48  # 种群大小
    transprob = 0.05  # 超越百分比
    crossprob = 0.75  # 交叉百分比
    mutprob = 0.2  # 变异百分比
    newpop = []

    for k in range(1, genNo + 1):
        cl = []
        p = []
        # 种群评估（从第二代开始）
        for i in range(len(newpop)):
            p1 = newpop[i]
            clnew, _ = solver(p1, uinf, AOA, Npanel)  # 适应度评估
            cl.append(clnew)
            p.append(p1)

        # 第一代种群初始化
        for i in range(popsize - len(newpop)):
            p1 = randp(p0, range_value)
            clnew, maxThickness = solver(p1, uinf, AOA, Npanel)  # 适应度评估
            # 几何约束
            if maxThickness > 0.1:
                clnew = cloriginal
            elif maxThickness < 0.01:
                clnew = cloriginal
            cl.append(clnew)
            p.append(p1)

        pop = np.array(p)
        # 约束升力系数
        for i in range(len(cl)):
            if cl[i] <= cloriginal:
                cl[i] = cloriginal

        # 按适应度对个体进行排序
        fi = np.array(cl) / np.sum(cl)
        ind = np.argsort(fi)[::-1]  # 降序排序
        fittest = fi[ind[:int(np.ceil(transprob * popsize))]]
        ind = ind[:int(np.ceil(transprob * popsize))]

        if k != genNo:
            newpop = pop[ind]
            # 交叉
            for i in range(int(np.ceil(crossprob * popsize))):
                indv1 = random.randint(0, popsize - 1)
                indv2 = random.randint(0, popsize - 1)
                crossindex = random.randint(0, 10)
                new_indv = np.concatenate((pop[indv1, :crossindex + 1], pop[indv2, crossindex + 1:]))
                newpop = np.vstack((newpop, new_indv))

            # 变异
            for i in range(int(np.ceil(mutprob * popsize))):
                indv = pop[random.randint(0, popsize - 1)]
                mutindex = random.randint(0, 10)
                pmut = randp(p0, range_value)
                indv[mutindex] = pmut[mutindex]
                newpop = np.vstack((newpop, indv))

    # 选择锦标赛获胜者或最进化的个体
    fittest_individual = pop[ind[0]]
    clfittest = cl[ind[0]]
    if clfittest == cloriginal:
        fittest_individual = p0


    plotairfoil(fittest_individual, 'k',' ')  # 绘制优化后的翼型
    plt.axis('equal')
    plotairfoil(p0, 'r',' ')  # plt.hold() is no longer necessary in newer matplotlib versions
    plt.legend(['Optimized', 'original'])
    plt.xlabel('X/C')
    plt.ylabel('Y/C')
    plt.title('Airfoil shape')

    return cloriginal, clfittest, fittest_individual
