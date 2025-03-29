import csv

def read_csv_from_line_31(file_path):
    data = []
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            # 过滤掉空字符
            filtered_content = (line.replace('\0', '') for line in csvfile)
            reader = csv.reader(filtered_content)
            # print(reader)
            # for row in reader:
            #     data.append(row)
            # 跳过前 30 行
            for _ in range(31):
                next(reader)
            # 从第 31 行开始读取数据
            for row in reader:
                data.append(row)
        return data
    except FileNotFoundError:
        print("错误: 文件未找到!")
    except Exception as e:
        print(f"错误: 发生了一个未知错误: {e}")
    return []

if __name__ == "__main__":
    file_path = r'C:\Users\admin\Desktop\数据\4fx8cjprxm-1\25 degC\773_HPPC.csv'  # 请替换为实际的 CSV 文件路径
    result = read_csv_from_line_31(file_path)
    if result:
        for row in result[51000:]:
            print(row)