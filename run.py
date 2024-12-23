"""
主运行脚本
"""
import importlib
import sys

# 确保每次运行都重新加载配置
if 'src.config' in sys.modules:
    importlib.reload(sys.modules['src.config'])
    # 重新加载相关模块
    if 'src.main' in sys.modules:
        importlib.reload(sys.modules['src.main'])
    if 'src.models' in sys.modules:
        importlib.reload(sys.modules['src.models'])

from src.main import main

if __name__ == "__main__":
    main() 