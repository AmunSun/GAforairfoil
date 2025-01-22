import numpy as np

def parsec(p):
    """
    This function determines a=[a1, a2, ...an] to solve the airfoil polynomial.
    Zn=an(p)*X^(n-1/2), where n is the number of coordinates for the upper or
    lower surface. 

    Input is a vector of PARSEC parameters p=[p1, p2, ...pn] where
    p1=rle         
    p2=Xup
    p3=Yup
    p4=YXXup
    p5=Xlow
    p6=Ylow
    p7=YXXlow
    p8=yte
    p9=delta yte (t.e. thickness)
    p10=alpha te
    p11=beta te
    """
    # 初始化向量
    c1 = np.array([1, 1, 1, 1, 1, 1])
    c2 = np.array([p[1]**(1/2), p[1]**(3/2), p[1]**(5/2), p[1]**(7/2), p[1]**(9/2), p[1]**(11/2)])
    c3 = np.array([1/2, 3/2, 5/2, 7/2, 9/2, 11/2])
    c4 = np.array([(1/2)*p[1]**(-1/2), (3/2)*p[1]**(1/2), (5/2)*p[1]**(3/2), (7/2)*p[1]**(5/2), (9/2)*p[1]**(7/2), (11/2)*p[1]**(9/2)])
    c5 = np.array([(-1/4)*p[1]**(-3/2), (3/4)*p[1]**(-1/2), (15/4)*p[1]**(1/2), (35/4)*p[1]**(3/2), (53/4)*p[1]**(5/2), (99/4)*p[1]**(7/2)])
    c6 = np.array([1, 0, 0, 0, 0, 0])

    # 构建上表面矩阵
    Cup = np.vstack((c1, c2, c3, c4, c5, c6))

    c7 = np.array([1, 1, 1, 1, 1, 1])
    c8 = np.array([p[4]**(1/2), p[4]**(3/2), p[4]**(5/2), p[4]**(7/2), p[4]**(9/2), p[4]**(11/2)])
    c9 = np.array([1/2, 3/2, 5/2, 7/2, 9/2, 11/2])
    c10 = np.array([(1/2)*p[4]**(-1/2), (3/2)*p[4]**(1/2), (5/2)*p[4]**(3/2), (7/2)*p[4]**(5/2), (9/2)*p[4]**(7/2), (11/2)*p[4]**(9/2)])
    c11 = np.array([(-1/4)*p[4]**(-3/2), (3/4)*p[4]**(-1/2), (15/4)*p[4]**(1/2), (35/4)*p[4]**(3/2), (53/4)*p[4]**(5/2), (99/4)*p[4]**(7/2)])
    c12 = np.array([0, 0, 0, 0, 0, 1])

    # 构建下表面矩阵
    Clo = np.vstack((c7, c8, c9, c10, c11, c12))

    # 构建上表面和下表面的右侧向量
    bup = np.array([p[7]+p[8]/2, p[2], np.tan(np.deg2rad(p[9] - p[10]/2)), 0, p[3], np.sqrt(2*p[0])])
    blo = np.array([p[7]+p[8]/2, p[5], np.tan(np.deg2rad(p[9] - p[10]/2)), 0, p[6], np.sqrt(2*p[0])])

    # 求解线性方程组
    aup = np.linalg.solve(Cup, bup)
    alower = np.linalg.solve(Clo, blo)

    # 合并结果
    a = np.zeros((12, 1))
    a[:6, 0] = aup
    a[6:, 0] = alower

    return a


def yCoord2(a, x):
    """
    此函数根据给定的 PARSEC 系数和 x 坐标点计算 y 坐标。
    
    参数:
    a (array-like): PARSEC 系数数组，长度应为 12。
    x (array-like): x 坐标点数组。
    
    返回:
    yu (array): 上表面 y 坐标数组。
    yl (array): 下表面 y 坐标数组。
    """
    # 将输入转换为 NumPy 数组，以便进行元素级运算
    a = np.array(a)
    x = np.array(x)
    
    # 计算上表面 y 坐标
    yu = a[0] * x**0.5 + a[1] * x**1.5 + a[2] * x**2.5 + a[3] * x**3.5 + a[4] * x**4.5 + a[5] * x**5.5
    
    # 计算下表面 y 坐标
    yl = a[6] * x**0.5 + a[7] * x**1.5 + a[8] * x**2.5 + a[9] * x**3.5 + a[10] * x**4.5 + a[11] * x**5.5
    
    return yu, yl
