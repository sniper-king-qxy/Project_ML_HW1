import numpy as np
import matplotlib.pyplot as plt

# 定义 Rint 模型参数
# 电池电动势，单位：伏特（V）
E = 3.7
# 电池内阻，单位：欧姆（Ω）
R_int = 0.1

# 生成时间序列，从 0 到 1000 秒，共 1000 个时间点
time = np.linspace(0, 1000, 1000)


# 定义充电电流函数
# 在充电阶段，前 600 秒以 0.5A 恒定电流充电，之后充电电流为 0
def charge_current(t):
    return 0.5 if t < 600 else 0


# 定义放电电流函数
# 在放电阶段，600 秒之后以 0.3A 恒定电流放电，之前放电电流为 0
def discharge_current(t):
    return -0.3 if t >= 600 else 0


# 初始化存储电压值的数组，与时间序列长度相同
voltage = np.zeros_like(time)

# 遍历每个时间点，计算该时刻的总电流和对应的电池端电压
for i, t in enumerate(time):
    # 计算该时刻的充电电流
    I_charge = charge_current(t)
    # 计算该时刻的放电电流
    I_discharge = discharge_current(t)
    # 计算该时刻的总电流
    I = I_charge + I_discharge
    # 根据 Rint 模型公式计算电池端电压
    voltage[i] = E + I * R_int

# 绘制充放电曲线
plt.plot(time, voltage)
# 设置 x 轴标签
plt.xlabel('Time (s)')
# 设置 y 轴标签
plt.ylabel('Voltage (V)')
# 设置图表标题
plt.title('Rint Model Charge-Discharge Curve')
# 显示网格线
plt.grid(True)
# 显示绘制的图表
plt.show()
