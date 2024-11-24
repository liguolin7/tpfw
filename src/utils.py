import os
from config import *

def create_directories():
    """创建必要的目录结构"""
    directories = [
        PROCESSED_DATA_DIR,
        os.path.join(RESULTS_DIR, 'metrics'),
        os.path.join(RESULTS_DIR, 'feature_importance')
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory) 