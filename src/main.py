from data_loader import *
from data_processor import DataProcessor
from models import BaselineModels
from evaluation import *
import logging

def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    logging.info("开始数据处理...")
    
    # 加载数据
    traffic_data = load_traffic_data()
    weather_data = load_weather_data()
    
    # 数据处理
    processor = DataProcessor()
    # ... 处理数据
    
    # 模型训练和评估
    # ... 训练和评估模型
    
if __name__ == "__main__":
    main() 