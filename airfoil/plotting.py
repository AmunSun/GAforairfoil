import numpy as np
import matplotlib.pyplot as plt
from .solver import solver
from .parsec import parsec, yCoord2
#定义绘图
def plotairfoil(p, color, label):
    """
    Plots the airfoil shape for given PARSEC parameters.
    
    Parameters:
    - p: List of PARSEC parameters.
    - color: Color of the plot line.
    - label: Label for the legend.
    """
    dbeta = np.pi / 200
    beta = 0
    Z_u0 = []
    Z_d0 = []
    x0 = []
    a = parsec(p)  # Compute PARSEC coefficients

    while beta <= np.pi:
        x_val = (1 - np.cos(beta)) / 2
        x0.append(x_val)
        zu, zd = yCoord2(a, x_val)
        Z_u0.append(zu)
        Z_d0.append(zd)
        beta += dbeta

    Z_d0 = Z_d0[::-1]  # Reverse the lower surface
    X = (1 - np.cos(np.linspace(0, 2 * np.pi, len(x0) * 2))) / 2
    Y = np.concatenate([Z_u0, Z_d0])

    plt.plot(X, Y, color, label=label)
# 定义 graphCl 函数，用于绘制机翼升力系数随攻角变化的曲线
def graphCl(p, Npanel, uinf, step, c):
    """
    此函数绘制机翼的升力系数与攻角的关系图。
    :param p: 机翼的 PARSEC 参数，用于描述机翼的几何形状
    :param Npanel: 用于求解升力系数 (Cl) 时划分的面板数量
    :param uinf: 自由来流的速度大小
    :param step: 攻角变化的步长
    :param c: 绘图时线条的颜色
    :return: cl 存储不同攻角下计算得到的升力系数数组，cdp 为空数组
    """
    # 计算攻角范围
    angle = np.arange(-30, 31, step)
    cl = []
    cdp = []

    # 遍历每个攻角
    for i in range(len(angle)):
        # 调用 solver 函数计算升力系数
        cl1, _ = solver(p, uinf, angle[i] * np.pi / 180, Npanel)
        cl.append(cl1)

    # 绘制 Cl 曲线
    plt.figure()
    plt.plot(angle, cl, color=c, marker='o', linewidth=1.5)
    plt.axis([-60, 60, -2.5, 2.5])

    return cl, cdp