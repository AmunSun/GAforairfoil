# main.py
import numpy as np
import time
from matplotlib import pyplot as plt
from airfoil.genetic import GAairfoil
from airfoil.plotting import plotairfoil
from airfoil.gurobi_opt import gurobi_airfoil_optimization
def main():
    # 初始翼型参数（以 NACA 0012 为例）
    p0 = np.array([0.0155, 0.296632, 0.060015, -0.4515, 0.296632, -0.06055, 0.453, 0, 0.001260, 0, 7.36])  # NACA 0012
    range_ = np.array([0.0015, 0.025, 0.015, -0.01, 0.02, -0.015, 0.075, 0, 0, -0.175, 0.05])
    Npanel = 200
    uinf = 1
    AOA = 5 * np.pi / 180
    genNo = 20  # 遗传算法代数

    start_time = time.time()
    # 调用遗传算法优化翼型
    cloriginal, clfittest, fittest = GAairfoil(genNo, p0, range_, uinf, AOA, Npanel)
    end_time = time.time()
    print(f"运行时间：{end_time - start_time:.2f} s")
    # 打印优化结果
    print(f"Original   Cl= {cloriginal}")
    print(f"Optimized  Cl= {clfittest}")

    # 绘制原始翼型和优化后的翼型
    plotairfoil(fittest, 'k', 'Optimized')
    plotairfoil(p0, 'r', 'Original')
    plt.axis('equal')
    plt.legend(['Optimized', 'Original'])
    plt.xlabel('X/C')
    plt.ylabel('Y/C')
    plt.title('Airfoil Shape Comparison')
    plt.grid(True)
    plt.show()

    grbstart_time = time.time()
    gurobi_cl_optimized, gurobi_optimized_params = gurobi_airfoil_optimization(p0, range_, uinf, AOA, Npanel)
    grbend_time = time.time()
    print(f"Gurobi运行时间：{grbend_time - grbstart_time:.2f} s")
    print(f"Optimized Cl: {gurobi_cl_optimized}")
    print(f"Optimized Parameters: {gurobi_optimized_params}")

if __name__ == "__main__":
    main()
