import control
import matplotlib.pyplot as plt


# 定义一个函数来绘制开环函数的负反馈闭环根轨迹
def plot_root_locus(num, den, title):
    # 创建开环传递函数对象
    open_loop_sys = control.TransferFunction(num, den)
    # 绘制根轨迹，默认绘制负反馈闭环系统的根轨迹
    plt.figure()
    control.root_locus(open_loop_sys)
    # 设置图形标题
    plt.title(title)
    # 设置 x 轴标签
    plt.xlabel('Real Axis')
    # 设置 y 轴标签
    plt.ylabel('Imaginary Axis')


# 定义第一个开环函数
# 分子多项式系数，按照 s 的降幂排列
num1 = [1,2]
# 分母多项式系数，按照 s 的降幂排列
den1 = [1, 2, 3]
title1 = 'Root Locus of Closed-loop System with Open-loop TF 1 (Negative Feedback)'
# 调用函数绘制第一个开环函数的根轨迹
plot_root_locus(num1, den1, title1)

# # 定义第二个开环函数
# # 分子多项式系数，按照 s 的降幂排列
# num2 = [1]
# # 分母多项式系数，按照 s 的降幂排列
# den2 = [1, 4, 4]
# title2 = 'Root Locus of Closed-loop System with Open-loop TF 2 (Negative Feedback)'
# # 调用函数绘制第二个开环函数的根轨迹
# plot_root_locus(num2, den2, title2)

# 显示所有绘制的图形
plt.show()
