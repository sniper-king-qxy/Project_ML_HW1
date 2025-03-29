import os
import pandas as pd
from pathlib import Path
import logging

# # 配置日志记录（2025年日志最佳实践）
# logging.basicConfig(
#     filename='data_processing.log',
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s: %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )


def process_file(input_path, output_root):
    """处理单个CSV文件的核心逻辑（含新版时间验证机制）"""
    try:
        # 读取原始数据（强化编码兼容性）
        df = pd.read_csv(
            input_path,
            skiprows=30,
            header=None,
            usecols=[0, 8, 9, 10, 11],
            dtype={0: 'object'},
            encoding='utf-8-sig'  # 处理BOM标记
        )

        # 列重命名与类型转换
        df.columns = ['Time_Stamp', 'Voltage', 'Current', 'Temperature', 'SOC']
        df['Time_Stamp'] = pd.to_datetime(
            df['Time_Stamp'],
            format='%m/%d/%Y %I:%M:%S %p',
            errors='coerce'
        )

        # # 数据清洗（新增2025年时间有效性验证）
        # cutoff_date = pd.Timestamp('2025-01-01')
        # valid_mask = (df['Time_Stamp'] >= cutoff_date) & (df['Time_Stamp'].notna())
        # df = df[valid_mask].sort_values('Time_Stamp')

        # 分组聚合（优化大数据集性能）
        grouped = df.groupby('Time_Stamp', as_index=False, observed=True).mean()

        # 数值处理（支持科学计数法数据）
        numeric_cols = ['Voltage', 'Current', 'Temperature', 'SOC']
        grouped[numeric_cols] = grouped[numeric_cols].apply(pd.to_numeric, errors='coerce')
        grouped['SOC'] = grouped['SOC'].add(3.0).round(6)/3

        # 构建输出路径（保持目录结构）
        rel_path = Path(input_path).relative_to(input_root)
        output_path = Path(output_root) / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存结果（优化大文件写入性能）
        grouped.to_csv(
            output_path,
            index=False,
            date_format='%Y-%m-%d %H:%M:%S',
            float_format='%.8f'
        )

        print(f" 成功处理: {input_path} -> {output_path}")

    except Exception as e:
        print(f" 处理失败: {input_path} - {str(e)}")


# 配置路径（示例路径）
input_root = 'C:\\Users\\admin\\Desktop\\数据\\LG 18650HG2 Li-ion Battery Data\\LG 18650HG2 Li-ion Battery Data\\LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020\\Data_myself'  # 原始数据根目录
output_root = 'C:\\Users\\admin\\Desktop\\数据\\LG 18650HG2 Li-ion Battery Data\\LG 18650HG2 Li-ion Battery Data\\LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020\\Filter'  # 处理结果根目录

# 递归遍历所有CSV文件（含子目录）
for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.lower().endswith('.csv'):
            input_path = os.path.join(root, file)
            process_file(input_path, output_root)

print(f"处理完成于：{pd.Timestamp.now().strftime('%Y-%m-%d  %H:%M:%S')}")