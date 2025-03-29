# predict_soc.py
import torch
import pandas as pd
import joblib
import argparse
import sys
from typing import Union, Dict, List
from model import FNN, Config_data  # 确保存在model.py 文件


class SOCPredictor:
    """电池SOC预测器（支持单样本/批量预测）"""

    def __init__(self, model_path: str, scaler_path: str):
        """
        初始化预测器
        Args:
            model_path: 模型文件路径(.pth)
            scaler_path: 标准化器文件路径(.save)
        """
        try:
            self.scaler = joblib.load(scaler_path)
            self.model = self._load_model(model_path)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"初始化失败: {str(e)}")

    def _load_model(self, path: str) -> torch.nn.Module:
        """安全加载PyTorch模型"""
        model = FNN(Config_data())

        # 跨版本兼容性处理
        load_kwargs = {'map_location': torch.device('cpu')}
        if torch.__version__ >= '1.13.0':
            load_kwargs['weights_only'] = True

        try:
            model.load_state_dict(torch.load(path, **load_kwargs))
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")
        return model

    def _preprocess(self, raw_data: Union[pd.DataFrame, Dict, List]) -> pd.DataFrame:
        """统一数据预处理流程"""
        if isinstance(raw_data, dict):
            df = pd.DataFrame([raw_data])
        elif isinstance(raw_data, list):
            df = pd.DataFrame(raw_data)
        elif isinstance(raw_data, pd.DataFrame):
            df = raw_data
        else:
            raise TypeError("输入类型应为DataFrame/dict/list")

        # 校验必要字段
        required_cols = ['Voltage', 'Current', 'Temperature']
        if missing := [col for col in required_cols if col not in df.columns]:
            raise ValueError(f"缺少必要字段: {missing}")

        return df[required_cols]

    def predict(self, input_data: Union[pd.DataFrame, Dict, List]) -> pd.DataFrame:
        """
        通用预测方法
        Args:
            input_data: 支持DataFrame/字典/列表三种格式
        Returns:
            包含原始数据和预测结果的DataFrame
        """
        df = self._preprocess(input_data)

        # 特征标准化
        scaled = self.scaler.transform(df.values)

        # 执行预测
        with torch.no_grad():
            tensor_data = torch.FloatTensor(scaled)
            predictions = self.model(tensor_data).numpy().flatten()

            # 构造结果
        result = df.copy()
        result['SOC_Prediction(%)'] = predictions.round(2)
        return result

    def predict_single(self, voltage: float, current: float, temp: float) -> float:
        """
        单样本预测便捷接口
        Args:
            voltage (V): 电压值 (3.0-4.2)
            current (A): 电流值 (0-5)
            temp (°C): 温度值 (-20-60)
        """
        try:
            return self.predict({
                'Voltage': float(voltage),
                'Current': float(current),
                'Temperature': float(temp)
            })['SOC_Prediction(%)'].iloc[0]
        except ValueError as e:
            raise ValueError(f"参数格式错误: {str(e)}")


def main():
    """命令行接口主函数"""
    parser = argparse.ArgumentParser(
        description='电池SOC预测系统 v2.1',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--file', help='批量预测CSV文件路径')
    group.add_argument('-p', '--params', nargs=3, type=float,
                       metavar=('VOLT', 'CURR', 'TEMP'),
                       help='单样本预测参数')
    parser.add_argument('-o', '--output', help='批量结果保存路径')

    args = parser.parse_args()

    try:
        predictor = SOCPredictor(
            model_path="./checkpoints/best_model.pth",
            scaler_path="./checkpoints/scaler.save"
        )
    except Exception as e:
        print(f"❌ 系统初始化失败: {str(e)}")
        sys.exit(1)

    if args.file:
        try:
            df = pd.read_csv(args.file)
            results = predictor.predict(df)

            if args.output:
                results.to_csv(args.output, index=False)
                print(f"✅ 预测结果已保存至 {args.output}")
            else:
                print("\n预测结果：")
                print(results.to_markdown(index=False))

        except Exception as e:
            print(f"❌ 批量预测失败: {str(e)}")
            sys.exit(2)

    elif args.params:
        try:
            soc = predictor.predict_single(*args.params)
            print(f"📊 预测SOC值: {soc*100}%")
        except Exception as e:
            print(f"❌ 参数预测失败: {str(e)}")
            sys.exit(3)


if __name__ == "__main__":
    main()