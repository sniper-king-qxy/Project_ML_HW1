# 导入 numpy 库，用于进行高效的数值计算和数组操作
import numpy as np
# 导入 pandas 库，用于数据处理和分析，常用来处理表格数据
import pandas as pd
# 从 sklearn 库的 model_selection 模块中导入 train_test_split 函数
# 该函数用于将数据集划分为训练集和测试集
from sklearn.model_selection import train_test_split
# 从 sklearn 库的 tree 模块中导入 DecisionTreeClassifier 类
# 该类用于创建决策树分类器模型
from sklearn.tree import DecisionTreeClassifier
# 从 sklearn 库的 metrics 模块中导入 accuracy_score 函数
# 该函数用于计算模型预测结果的准确率
from sklearn.metrics import accuracy_score
# 从 sklearn 库的 tree 模块中导入 plot_tree 函数
# 该函数用于可视化决策树模型
from sklearn.tree import plot_tree
# 导入 matplotlib 库的 pyplot 模块，用于创建各种可视化图表
import matplotlib.pyplot as plt
# 从 sklearn 库的 datasets 模块中导入 load_iris 函数
# 该函数用于加载经典的鸢尾花数据集
from sklearn.datasets import load_iris

# 设置 matplotlib 使用支持中文的字体，这里选择黑体字体
# 若系统中没有黑体字体，可尝试其他中文字体，如 'Microsoft YaHei'（微软雅黑）
plt.rcParams['font.family'] = 'SimHei'
# 解决 matplotlib 中负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False


# 步骤1: 加载数据集
# 调用 load_iris 函数加载鸢尾花数据集
iris = load_iris()
# 查看数据集的基本信息
print("数据集基本信息：")
iris.keys()

# 查看特征数据
X = iris.data
print("特征数据形状：", X.shape)
print("前几行特征数据：")
print(X[:5])

# 查看目标标签
y = iris.target
print("目标标签形状：", y.shape)
print("前几个目标标签：")
print(y[:5])

# 查看特征名称
feature_names = iris.feature_names
print("特征名称：", feature_names)

# 查看类别名称
target_names = iris.target_names
print("类别名称：", target_names)
# 将鸢尾花数据集的特征数据转换为 pandas 的 DataFrame 格式
# 并使用鸢尾花数据集的特征名称作为列名，方便后续查看和处理
X = pd.DataFrame(iris.data, columns=iris.feature_names)
# 提取鸢尾花数据集的目标变量（即类别标签）
y = iris.target

# 步骤2: 划分训练集和测试集
# 使用 train_test_split 函数将特征数据 X 和目标变量 y 划分为训练集和测试集
# test_size=0.2 表示将 20% 的数据作为测试集，剩余 80% 作为训练集
# random_state=42 用于设置随机种子，保证每次运行代码时划分的结果一致
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 步骤3: 创建决策树分类器
# 实例化一个 DecisionTreeClassifier 对象，使用基尼不纯度（gini）作为划分标准
# max_depth=3 表示决策树的最大深度为 3，可防止模型过拟合
clf = DecisionTreeClassifier(criterion='gini', max_depth=3)

# 步骤4: 训练模型
# 调用决策树分类器的 fit 方法，使用训练集的特征数据 X_train 和目标变量 y_train 对模型进行训练
clf.fit(X_train, y_train)

# 步骤5: 模型预测
# 调用训练好的决策树分类器的 predict 方法，对测试集的特征数据 X_test 进行预测
# 得到预测的类别标签 y_pred
y_pred = clf.predict(X_test)

# 步骤6: 评估模型性能
# 使用 accuracy_score 函数计算模型在测试集上的准确率
# 即预测正确的样本数占总样本数的比例
accuracy = accuracy_score(y_test, y_pred)
# 打印模型在测试集上的准确率，保留两位小数
print(f"模型在测试集上的准确率: {accuracy:.2f}")

# 步骤7: 可视化决策树
# 创建一个新的图形窗口，设置图形的大小为宽 15 英寸，高 10 英寸
plt.figure(figsize=(15, 10))
# 调用 plot_tree 函数绘制决策树
# feature_names=iris.feature_names 指定特征名称，用于在决策树节点上显示特征信息
# class_names=iris.target_names 指定类别名称，用于在决策树叶子节点上显示类别信息
# filled=True 表示给决策树的节点填充颜色，不同颜色代表不同的类别
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
# 设置图形的标题为“决策树可视化”
plt.title("决策树可视化")
# 显示绘制好的图形
plt.show()