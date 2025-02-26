import numpy as np
import matplotlib.pyplot as plt

# 定义 Gibbs 自由能曲线函数
def G_phase(x, a, b, c):
    return a * x**2 + b * x + c

# 温度区对应参数 (示例)
temperatures = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']
coefficients = [
    {'V': (1, -2, 1)},  # T1: 气相
    {'V': (1, -2, 1), 'L': (2, -3, 0)},  # T2: 气+液
    {'L': (2, -3, 0)},  # T3: 单液相
    {'L1': (1, -1, 0), 'L2': (1, -2, 1)},  # T4: B贫/B富液
    {'L1': (1, -1, 0), 'L2': (1, -2, 1)},  # T5: B贫/B富液
    {'L': (2, -3, 0)}  # T6: 单液相
]

# 绘图
x = np.linspace(0, 1, 100)
for i, T in enumerate(temperatures):
    plt.figure()
    plt.title(f"Gibbs Free Energy at {T}")
    for phase, (a, b, c) in coefficients[i].items():
        plt.plot(x, G_phase(x, a, b, c), label=phase)
    plt.xlabel('x_B')
    plt.ylabel('G')
    plt.legend()
    plt.savefig(f'G_{T}.png')
    plt.show()