import pandas as pd
import os
from tqdm import tqdm  # 用于显示进度条（需安装：pip install tqdm）


def merge_csv_with_index(folder_path, output_file='merged_data.csv'):
    """
    合并指定文件夹内所有CSV文件，并将第一列替换为合并后的行号

    参数：
    folder_path : str - CSV文件所在文件夹路径
    output_file : str - 输出文件名（默认merged_data.csv ）
    """
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dfs = []

    # 第一阶段：文件读取与预处理
    for file in tqdm(all_files, desc='正在读取文件'):
        file_path = os.path.join(folder_path, file)

        # 自动检测编码格式（处理中文编码问题）
        try:
            for encoding in ['utf-8', 'gbk', 'gb2312', 'latin1']:
                try:
                    # 分块读取（适用于大文件）
                    reader = pd.read_csv(file_path, encoding=encoding, chunksize=10000)
                    temp_df = pd.concat([chunk.iloc[:, 1:] for chunk in reader])
                    break
                except UnicodeDecodeError:
                    continue
            dfs.append(temp_df)
        except Exception as e:
            print(f"文件 {file} 读取失败: {str(e)}")
            continue

            # 第二阶段：数据合并与处理
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)

        # 添加行号列（从1开始）
        merged_df.insert(0, 'RowID', merged_df.index + 1)

        # 第三阶段：结果输出
        merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"合并完成！共处理 {len(merged_df)} 行数据，结果已保存至 {output_file}")
        return merged_df.shape
    else:
        print("未找到有效CSV文件")
        return None

    # 使用示例


if __name__ == "__main__":
    # 设置包含CSV文件的文件夹路径
    csv_folder = (r"C:\Users\admin\Desktop\数据\LG 18650HG2 Li-ion Battery Data\LG 18650HG2"
                  r" Li-ion Battery Data\LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020\Filter")  # 替换为实际路径
    output_path=(r"C:\Users\admin\Desktop\数据\LG 18650HG2 Li-ion Battery Data\LG 18650HG2 Li-ion"
                 r" Battery Data\LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020\merged_data.csv")
    merge_csv_with_index(csv_folder,output_path)