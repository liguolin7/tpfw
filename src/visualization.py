import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import RESULTS_DIR, VISUALIZATION_CONFIG
import logging

class ExperimentVisualizer:
    def __init__(self, results_data):
        """初始化可视化器"""
        self.results = results_data
        self.figures_dir = os.path.join(RESULTS_DIR, 'additional_figures')
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # 设置可视化样式
        plt.style.use(VISUALIZATION_CONFIG['style'])
        sns.set_palette(VISUALIZATION_CONFIG['color_palette'])
    
    def plot_metrics_comparison(self):
        """绘制不同模型指标对比图"""
        metrics = ['rmse', 'mae', 'r2', 'mape']
        models = ['LSTM', 'GRU', 'CNN_LSTM']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            baseline_values = [self.results['baseline'][model][metric] for model in models]
            enhanced_values = [self.results['enhanced'][model][metric] for model in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            axes[idx].bar(x - width/2, baseline_values, width, label='Baseline')
            axes[idx].bar(x + width/2, enhanced_values, width, label='Enhanced')
            axes[idx].set_title(f'{metric.upper()} Comparison')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(models)
            axes[idx].legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'metrics_comparison.png'))
        plt.close()
    
    def plot_improvement_heatmap(self):
        """绘制性能提升热力图"""
        models = ['LSTM', 'GRU', 'CNN_LSTM']
        metrics = ['rmse', 'mae', 'r2', 'mape']
        
        improvement_data = np.array([
            [39.24, 44.12, 0.16, 66.10],  # LSTM
            [57.99, 54.12, 1.18, 43.89],  # GRU
            [8.37, 4.33, 1.35, 35.70]     # CNN_LSTM
        ])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(improvement_data, 
                   annot=True, 
                   fmt='.2f', 
                   xticklabels=metrics,
                   yticklabels=models,
                   cmap='YlOrRd')
        plt.title('Performance Improvement Heatmap (%)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'improvement_heatmap.png'))
        plt.close()
    
    def plot_model_ranking(self):
        """绘制模型综合排名图"""
        models = ['LSTM', 'GRU', 'CNN_LSTM']
        avg_improvements = {
            'LSTM': np.mean([39.24, 44.12, 0.16, 66.10]),
            'GRU': np.mean([57.99, 54.12, 1.18, 43.89]),
            'CNN_LSTM': np.mean([8.37, 4.33, 1.35, 35.70])
        }
        
        plt.figure(figsize=(10, 6))
        models_sorted = sorted(avg_improvements.items(), key=lambda x: x[1], reverse=True)
        
        sns.barplot(x=[m[0] for m in models_sorted], y=[m[1] for m in models_sorted])
        plt.title('Average Performance Improvement by Model')
        plt.xlabel('Model')
        plt.ylabel('Average Improvement (%)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'model_ranking.png'))
        plt.close()
    
    def plot_weather_impact(self):
        """绘制天气数据分布和缺失值分析"""
        missing_values = pd.Series({
            'WT01': 1440, 'WT02': 1152, 'WT03': 12384,
            'WT08': 0, 'WT13': 1152, 'WT14': 4032,
            'WT16': 4320, 'WT21': 11520
        })
        
        plt.figure(figsize=(12, 6))
        missing_values.plot(kind='bar')
        plt.title('Missing Values in Weather Features')
        plt.xlabel('Weather Features')
        plt.ylabel('Number of Missing Values')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'weather_missing_values.png'))
        plt.close()
    
    def plot_data_distribution(self):
        """绘制数据集分布"""
        plt.figure(figsize=(15, 5))
        
        # 数据集大小对比
        sizes = {
            'Training': 22395,
            'Validation': 4798,
            'Testing': 4800
        }
        
        plt.subplot(1, 2, 1)
        plt.pie(sizes.values(), labels=sizes.keys(), autopct='%1.1f%%')
        plt.title('Dataset Split Distribution')
        
        # 特征维度对比
        dimensions = {
            'Baseline': 219,
            'Enhanced': 227
        }
        
        plt.subplot(1, 2, 2)
        plt.bar(dimensions.keys(), dimensions.values())
        plt.title('Feature Dimensions Comparison')
        plt.ylabel('Number of Features')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'data_distribution.png'))
        plt.close()
    
    def generate_all_visualizations(self):
        """生成所有可视化图表"""
        logging.info("开始生成额外的可视化图表...")
        
        self.plot_metrics_comparison()
        self.plot_improvement_heatmap()
        self.plot_model_ranking()
        self.plot_weather_impact()
        self.plot_data_distribution()
        
        logging.info(f"可视化图表已保存到: {self.figures_dir}")

def main():
    # 构造实验结果数据（从experiment_output.txt中提取）
    results = {
        'baseline': {
            'LSTM': {'rmse': 0.0487, 'mae': 0.0381, 'r2': 0.9975, 'mape': 80.1125},
            'GRU': {'rmse': 0.1161, 'mae': 0.0924, 'r2': 0.9858, 'mape': 161.5758},
            'CNN_LSTM': {'rmse': 0.2722, 'mae': 0.2110, 'r2': 0.9221, 'mape': 448.6342}
        },
        'enhanced': {
            'LSTM': {'rmse': 0.0296, 'mae': 0.0213, 'r2': 0.9991, 'mape': 27.1620},
            'GRU': {'rmse': 0.0488, 'mae': 0.0424, 'r2': 0.9975, 'mape': 90.6618},
            'CNN_LSTM': {'rmse': 0.2495, 'mae': 0.2019, 'r2': 0.9346, 'mape': 288.4759}
        }
    }
    
    # 创建可视化器并生成图表
    visualizer = ExperimentVisualizer(results)
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main() 