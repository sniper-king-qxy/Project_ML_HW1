# 导入必要的库
# sklearn.datasets 用于加载数据集
from sklearn.datasets import load_iris
# train_test_split 用于将数据集划分为训练集和测试集
from sklearn.model_selection import train_test_split
# RandomForestClassifier 是随机森林分类器的实现
from sklearn.ensemble import RandomForestClassifier
# accuracy_score 用于计算模型的准确率
from sklearn.metrics import accuracy_score
import pandas as pd

# 步骤1：加载数据集
# 加载鸢尾花数据集
iris = load_iris()
# 将特征数据存储在DataFrame中，方便查看和处理
X = pd.DataFrame(iris.data, columns=iris.feature_names)
# 目标标签
y = iris.target

# 步骤2：划分训练集和测试集
# test_size=0.3 表示将30%的数据作为测试集
# random_state=42 是随机数种子，保证每次划分的结果一致
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 步骤3：创建随机森林分类器
# n_estimators=100 表示使用100棵决策树
# random_state=42 保证每次运行时随机森林的构建结果一致
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 步骤4：训练模型
# 使用训练集数据对随机森林分类器进行训练
rf_classifier.fit(X_train, y_train)

# 步骤5：进行预测
# 使用训练好的模型对测试集数据进行预测
y_pred = rf_classifier.predict(X_test)

# 步骤6：评估模型
# 计算模型的准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型的准确率: {accuracy * 100:.2f}%")