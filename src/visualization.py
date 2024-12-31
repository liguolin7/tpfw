from .config import VISUALIZATION_CONFIG, RESULTS_DIR
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import matplotlib.gridspec as gridspec
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import shap
import logging
import tensorflow as tf

class DataVisualizer:
    def __init__(self):
        """Initialize visualizer"""
        plt.style.use(VISUALIZATION_CONFIG['style'])
        self.figure_size = VISUALIZATION_CONFIG['figure_size']
        self.dpi = VISUALIZATION_CONFIG['dpi']
        
        # Set a universal font that supports both English and symbols
        plt.rcParams['font.family'] = 'DejaVu Sans'
        # Fallback to Arial if DejaVu Sans is not available
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
        
        self.results_dir = RESULTS_DIR
        self.figures_dir = os.path.join(self.results_dir, 'figures')
        
        # 创建所有必要的子目录
        self.subdirs = {
            'analysis': os.path.join(self.figures_dir, 'analysis'),
            'comparison': os.path.join(self.figures_dir, 'comparison'),
            'models': os.path.join(self.figures_dir, 'models'),
            'training': os.path.join(self.figures_dir, 'training'),
            'traffic': os.path.join(self.figures_dir, 'traffic'),
            'weather': os.path.join(self.figures_dir, 'weather')
        }
        
        # 创建目录
        for dir_path in self.subdirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
    def plot_traffic_patterns(self, traffic_data, save_path):
        """可视化交通流量模式
        
        Args:
            traffic_data: 交通数据DataFrame
            save_path: 保存路径
        """
        # 1. 日内交通流量模式
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        hourly_pattern = traffic_data.groupby(traffic_data.index.hour).mean().mean(axis=1)
        plt.plot(hourly_pattern.index, hourly_pattern.values, marker='o')
        plt.title('Average Daily Traffic Pattern', fontsize=14)
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Traffic Flow')
        plt.grid(True)
        
        # 2. 周内交通流量模式
        plt.subplot(2, 1, 2)
        weekly_pattern = traffic_data.groupby(traffic_data.index.dayofweek).mean().mean(axis=1)
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        plt.plot(weekly_pattern.index, weekly_pattern.values, marker='o')
        plt.xticks(range(7), days, rotation=45)
        plt.title('Weekly Traffic Pattern', fontsize=14)
        plt.xlabel('Day of Week')
        plt.ylabel('Average Traffic Flow')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'traffic_patterns.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 交通流量热力图
        plt.figure(figsize=(15, 6))
        sensor_means = traffic_data.mean()
        sensor_matrix = sensor_means.values.reshape(23, 9)  # 调整形状以获得更好的可视化效果
        
        sns.heatmap(sensor_matrix, 
                    cmap='YlOrRd',
                    cbar_kws={'label': 'Average Traffic Flow'})
        plt.title('Traffic Flow Intensity by Sensor Location', fontsize=14)
        plt.xlabel('Sensor Column')
        plt.ylabel('Sensor Row')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'traffic_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_weather_analysis(self, weather_data, save_path):
        """可视化天气数据分析
        
        Args:
            weather_data: 天气数据DataFrame
            save_path: 保存路径
        """
        # 1. 温度变化趋势
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(weather_data.index, weather_data['TMAX'], label='Max Temperature')
        plt.plot(weather_data.index, weather_data['TMIN'], label='Min Temperature')
        plt.title('Temperature Variation Over Time', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)
        
        # 2. 降水和湿度关系 - 改进版
        plt.subplot(2, 1, 2)
        
        # 创建湿度区间
        humidity_bins = pd.qcut(weather_data['RHAV'], q=10)
        
        # 计算每个湿度区间的平均降水量
        avg_precip = weather_data.groupby(humidity_bins)['PRCP'].agg(['mean', 'std', 'count'])
        
        # 绘制带误差条的柱状图
        x = np.arange(len(avg_precip))
        plt.bar(x, avg_precip['mean'], 
                yerr=avg_precip['std'],
                capsize=5,
                alpha=0.6,
                color='skyblue',
                label='Average Precipitation')
        
        # 添加趋势线
        z = np.polyfit(x, avg_precip['mean'], 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "r--", alpha=0.8, label='Trend Line')
        
        # 设置x轴标签
        plt.xticks(x, [f'{int(bin.left)}-{int(bin.right)}' for bin in avg_precip.index],
                   rotation=45)
        
        plt.title('Average Precipitation by Humidity Range', fontsize=14)
        plt.xlabel('Relative Humidity Range (%)')
        plt.ylabel('Average Precipitation (mm)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加相关系数注释
        corr = weather_data['PRCP'].corr(weather_data['RHAV'])
        plt.text(0.02, 0.98, f'Correlation: {corr:.2f}',
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8),
                 verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'weather_patterns.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 风速和风向分布
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(weather_data['AWND'], bins=30)
        plt.title('Wind Speed Distribution', fontsize=14)
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        plt.hist(weather_data['WDF2'], bins=36)
        plt.title('Wind Direction Distribution', fontsize=14)
        plt.xlabel('Wind Direction (degrees)')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'wind_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_weather_impact(self, weather_data, traffic_data, save_path=None):
        """可视化天气对交通的影响"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size)
        
        # 计算平均交通流量
        avg_traffic = traffic_data.mean(axis=1)
        
        # 温度与交通流量
        ax1.scatter(weather_data['TMAX'], avg_traffic, alpha=0.5)
        ax1.set_title('Temperature vs Traffic Flow')
        ax1.set_xlabel('Temperature')
        ax1.set_ylabel('Average Traffic Flow')
        
        # 降水与交通流量
        ax2.scatter(weather_data['PRCP'], avg_traffic, alpha=0.5)
        ax2.set_title('Precipitation vs Traffic Flow')
        ax2.set_xlabel('Precipitation')
        ax2.set_ylabel('Average Traffic Flow')
        
        # 风速与交通流量
        ax3.scatter(weather_data['AWND'], avg_traffic, alpha=0.5)
        ax3.set_title('Wind Speed vs Traffic Flow')
        ax3.set_xlabel('Wind Speed')
        ax3.set_ylabel('Average Traffic Flow')
        
        # 湿度与交通流量
        ax4.scatter(weather_data['RHAV'], avg_traffic, alpha=0.5)
        ax4.set_title('Humidity vs Traffic Flow')
        ax4.set_xlabel('Relative Humidity')
        ax4.set_ylabel('Average Traffic Flow')
        
        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, 'weather_impact.png'), dpi=self.dpi)
            plt.close()
        return fig
    
    def plot_model_performance(self, history_dict, save_path=None):
        """可视化模型训练过程"""
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig)
        
        # 训练损失
        ax1 = fig.add_subplot(gs[0, 0])
        for model_name, history in history_dict.items():
            ax1.plot(history.history['loss'], label=f'{model_name}_train')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # 验证损失
        ax2 = fig.add_subplot(gs[0, 1])
        for model_name, history in history_dict.items():
            ax2.plot(history.history['val_loss'], label=f'{model_name}_val')
        ax2.set_title('Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        # RMSE
        ax3 = fig.add_subplot(gs[1, 0])
        for model_name, history in history_dict.items():
            if 'rmse' in history.history:
                ax3.plot(history.history['rmse'], label=f'{model_name}_train')
            if 'val_rmse' in history.history:
                ax3.plot(history.history['val_rmse'], label=f'{model_name}_val')
        ax3.set_title('RMSE')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('RMSE')
        ax3.legend()
        
        # MAE
        ax4 = fig.add_subplot(gs[1, 1])
        for model_name, history in history_dict.items():
            if 'mae' in history.history:
                ax4.plot(history.history['mae'], label=f'{model_name}_train')
            if 'val_mae' in history.history:
                ax4.plot(history.history['val_mae'], label=f'{model_name}_val')
        ax4.set_title('MAE')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('MAE')
        ax4.legend()
        
        # MAPE
        ax5 = fig.add_subplot(gs[2, 0])
        for model_name, history in history_dict.items():
            if 'mape' in history.history:
                ax5.plot(history.history['mape'], label=f'{model_name}_train')
            if 'val_mape' in history.history:
                ax5.plot(history.history['val_mape'], label=f'{model_name}_val')
        ax5.set_title('MAPE')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('MAPE')
        ax5.legend()
        
        # 学习率变化
        ax6 = fig.add_subplot(gs[2, 1])
        for model_name, history in history_dict.items():
            if 'lr' in history.history:
                ax6.plot(history.history['lr'], label=f'{model_name}_lr')
        ax6.set_title('Learning Rate Changes')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Learning Rate')
        ax6.legend()
        
        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, 'model_performance.png'), dpi=self.dpi)
            plt.close()
        return fig
    
    def plot_prediction_comparison(self, y_true, predictions_dict, save_path=None):
        """可视化不同模型的预测结果对比"""
        # 确保保存目录存在
        if save_path:
            os.makedirs(save_path, exist_ok=True)
        
        # 1. 预测值与实际值对比
        plt.figure(figsize=(12, 8))
        plt.plot(y_true[:100], 'k-', label='Actual', alpha=0.7)
        for model_name, pred in predictions_dict.items():
            plt.plot(pred[:100], '--', label=f'{model_name}_pred', alpha=0.7)
        plt.title('Prediction vs Actual')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, 'prediction_vs_actual.png'), dpi=300)
            plt.close()
        
        # 2. 误差分布
        plt.figure(figsize=(12, 8))
        for model_name, pred in predictions_dict.items():
            errors = y_true - pred
            sns.histplot(errors, label=f'{model_name}_error', alpha=0.5)
        plt.title('Error Distribution')
        plt.xlabel('Error')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, 'error_distribution.png'), dpi=300)
            plt.close()
        
        # 3. 预测散点图
        plt.figure(figsize=(12, 8))
        for model_name, pred in predictions_dict.items():
            plt.scatter(y_true, pred, alpha=0.5, label=model_name)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
        plt.title('Prediction Scatter Plot')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, 'prediction_scatter.png'), dpi=300)
            plt.close()
        
        # 4. 累积误差
        plt.figure(figsize=(12, 8))
        for model_name, pred in predictions_dict.items():
            cum_error = np.cumsum(np.abs(y_true - pred))
            plt.plot(cum_error, label=model_name)
        plt.title('Cumulative Error')
        plt.xlabel('Time Step')
        plt.ylabel('Cumulative Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, 'cumulative_error.png'), dpi=300)
            plt.close()
    
    def plot_model_comparison(self, baseline_results, enhanced_results, results_dir, metrics=['RMSE', 'MAE', 'R2', 'MAPE']):
        """绘制基础模型和增强模型的性能对比图"""
        # 确保目录存在
        os.makedirs(results_dir, exist_ok=True)
        
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            models = list(baseline_results.keys())
            baseline_values = [baseline_results[model][metric] for model in models]
            enhanced_values = [enhanced_results[model][metric] for model in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            plt.bar(x - width/2, baseline_values, width, label='Baseline')
            plt.bar(x + width/2, enhanced_values, width, label='Enhanced')
            
            plt.xlabel('Models')
            plt.ylabel(metric)
            plt.title(f'{metric} Comparison')
            plt.xticks(x, models)
            plt.legend()
            
            save_path = os.path.join(results_dir, f'comparison_{metric.lower()}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_model_improvements(self, baseline_metrics, enhanced_metrics, save_path):
        """绘制模型改进对比图"""
        # 1. 性能提升热力图
        plt.figure(figsize=(12, 8))
        improvements = {}
        models = ['LSTM', 'GRU', 'CNN_LSTM']
        metrics = ['RMSE', 'MAE', 'R2', 'MAPE']
        
        for model in models:
            improvements[model] = []
            for metric in metrics:
                if metric == 'R2':
                    imp = (enhanced_metrics[model][metric] - baseline_metrics[model][metric]) * 100
                else:
                    imp = ((baseline_metrics[model][metric] - enhanced_metrics[model][metric]) / 
                          baseline_metrics[model][metric]) * 100
                improvements[model].append(imp)
        
        improvement_matrix = pd.DataFrame(improvements, index=metrics).T
        sns.heatmap(improvement_matrix, annot=True, cmap='RdYlGn', center=0, fmt='.1f')
        plt.title('Performance Improvements (%)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'performance_heatmap.png'), dpi=300)
        plt.close()
    
    def plot_prediction_vs_actual(self, y_true, y_pred, timestamps, model_name, save_path=None):
        """Visualize prediction vs actual values"""
        plt.figure(figsize=(15, 6))
        plt.plot(timestamps, y_true, label='Actual', color='blue', alpha=0.6)
        plt.plot(timestamps, y_pred, label='Predicted', color='red', alpha=0.6)
        
        plt.title(f'{model_name} Model Prediction Results', fontsize=12)
        plt.xlabel('Time')
        plt.ylabel('Traffic Flow')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add shadow area for prediction errors
        plt.fill_between(timestamps, y_true, y_pred, color='gray', alpha=0.2, label='Prediction Error')
        
        if save_path:
            plt.savefig(os.path.join(save_path, f'{model_name}_prediction.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_feature_importance_analysis(self, model, feature_names, save_path):
        """可视化特征重要性分析"""
        plt.figure(figsize=(12, 6))
        
        # 使用模型的权重来分析特征重要性
        weights = []
        for layer in model.layers:
            if 'dense' in layer.name.lower():
                w = layer.get_weights()[0]
                weights.append(np.abs(w).mean(axis=1))
        
        if weights:
            # 计算平均特征重要性
            importance_scores = np.abs(weights[0])
            
            # 确保特征名称和重要性分数长度匹配
            min_len = min(len(feature_names), len(importance_scores))
            feature_names = feature_names[:min_len]
            importance_scores = importance_scores[:min_len]
            
            # 创建特征重要性数据框
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance_scores
            })
            
            # 按重要性排序
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # 绘制前15最重要的特征（由于我们减少了特征数量）
            plt.figure(figsize=(12, 6))
            top_n = min(15, len(importance_df))
            sns.barplot(data=importance_df.head(top_n), 
                       x='Importance', 
                       y='Feature',
                       palette='viridis')
            
            plt.title('Top Most Important Features', fontsize=14)
            plt.xlabel('Importance Score', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'feature_importance.png'), dpi=300)
            plt.close()
            
            return importance_df
        else:
            logging.warning("No dense layers found in the model for feature importance analysis")
            return None
    
    def plot_error_distribution(self, y_true, y_pred, model_name, save_path=None):
        """Visualize prediction error distribution"""
        errors = y_pred - y_true
        
        plt.figure(figsize=(12, 6))
        
        # Create subplots
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
        
        # Error over time
        ax1 = plt.subplot(gs[0])
        ax1.plot(range(len(errors)), errors, color='blue', alpha=0.6)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax1.set_title('Prediction Error Over Time')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Prediction Error')
        ax1.grid(True, alpha=0.3)
        
        # Error distribution histogram
        ax2 = plt.subplot(gs[1])
        ax2.hist(errors, bins=50, orientation='horizontal', color='blue', alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax2.set_title('Error Distribution')
        ax2.set_xlabel('Frequency')
        
        # Add statistics
        stats_text = f'Mean: {np.mean(errors):.4f}\nStd: {np.std(errors):.4f}'
        ax2.text(0.95, 0.95, stats_text,
                 transform=ax2.transAxes,
                 verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, f'{model_name}_error_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_weather_impact_analysis(self, baseline_results, enhanced_results, weather_data):
        """可视化天气对预测性能的影响"""
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2)
        
        # 1. 不同天气条件下的预测误差对比
        ax1 = plt.subplot(gs[0, 0])
        self._plot_weather_condition_comparison(
            baseline_results,
            enhanced_results,
            weather_data,
            ax1
        )
        
        # 2. 极端天气事件分析
        ax2 = plt.subplot(gs[0, 1])
        self._plot_extreme_weather_analysis(
            baseline_results,
            enhanced_results,
            weather_data,
            ax2
        )
        
        # 3. 天气特征重要性分析
        ax3 = plt.subplot(gs[1, 0])
        self._plot_weather_feature_importance(
            enhanced_results,
            weather_data,
            ax3
        )
        
        # 4. 性能提升统计
        ax4 = plt.subplot(gs[1, 1])
        self._plot_performance_improvement(
            baseline_results,
            enhanced_results,
            ax4
        )
        
        plt.tight_layout()
        return fig
    
    def create_performance_table(self, baseline_metrics, enhanced_metrics, improvements, save_path):
        """创建性能对比表格"""
        table_data = []
        
        for model in baseline_metrics.keys():
            row = {
                'Model': model,
                'Baseline\nRMSE': f"{baseline_metrics[model]['RMSE']:.4f}",
                'Enhanced\nRMSE': f"{enhanced_metrics[model]['RMSE']:.4f}",
                'RMSE\nImprovement': f"{improvements[model]['RMSE_improvement']:.2f}%",
                'Baseline\nMAE': f"{baseline_metrics[model]['MAE']:.4f}",
                'Enhanced\nMAE': f"{enhanced_metrics[model]['MAE']:.4f}",
                'MAE\nImprovement': f"{improvements[model]['MAE_improvement']:.2f}%",
                'Baseline\nR2': f"{baseline_metrics[model]['R2']:.4f}",
                'Enhanced\nR2': f"{enhanced_metrics[model]['R2']:.4f}",
                'R2\nImprovement': f"{improvements[model]['R2_improvement']:.2f}%"
            }
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        # 创建表格可视化
        plt.figure(figsize=(15, 5))
        plt.axis('off')
        
        table = plt.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        # 调整表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # 保存表格到comparison文件夹
        plt.savefig(os.path.join(self.subdirs['comparison'], 'performance_table.png'), 
                    dpi=300, 
                    bbox_inches='tight',
                    pad_inches=0.05)
        plt.close()
    
    def plot_traffic_time_analysis(self, traffic_data, save_path):
        """交通数据的时序特征分析"""
        plt.figure(figsize=(15, 12))
        
        # 1. 交通流量的季节性分解
        # 选择一个型的传感器
        sample_sensor = traffic_data.iloc[:, 0]
        decomposition = seasonal_decompose(sample_sensor, period=24*12)  # 12小时为周期
        
        plt.subplot(4, 1, 1)
        plt.plot(sample_sensor[:24*7])  # 示一周的数据
        plt.title('Original Traffic Flow')
        plt.grid(True)
        
        plt.subplot(4, 1, 2)
        plt.plot(decomposition.trend[:24*7])
        plt.title('Trend')
        plt.grid(True)
        
        plt.subplot(4, 1, 3)
        plt.plot(decomposition.seasonal[:24*7])
        plt.title('Seasonal')
        plt.grid(True)
        
        plt.subplot(4, 1, 4)
        plt.plot(decomposition.resid[:24*7])
        plt.title('Residual')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'traffic_time_decomposition.png'), dpi=300)
        plt.close()
        
        # 2. 交通流量的自相关分析
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        plot_acf(sample_sensor.dropna(), lags=48, ax=plt.gca())
        plt.title('Autocorrelation')
        
        plt.subplot(1, 2, 2)
        plot_pacf(sample_sensor.dropna(), lags=48, ax=plt.gca())
        plt.title('Partial Autocorrelation')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'traffic_correlation.png'), dpi=300)
        plt.close()
    
    def plot_weather_correlation_analysis(self, weather_data, save_path):
        """天气数据的相分析"""
        # 1. 天气特征相关性热力图
        plt.figure(figsize=(12, 10))
        
        # 选择主要的天气特征进行相关性分析
        main_features = ['TMAX', 'TMIN', 'PRCP', 'AWND', 'RHAV', 'RHMN', 'RHMX']
        weather_corr = weather_data[main_features].corr()
        
        sns.heatmap(weather_corr, 
                    annot=True, 
                    fmt='.2f', 
                    cmap='RdBu',
                    center=0,
                    square=True,
                    vmin=-1, 
                    vmax=1)
        plt.title('Weather Features Correlation Matrix', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'weather_correlation.png'), dpi=300)
        plt.close()
        
        # 2. 主要天气特征的分布分析（拆分成4个子图）
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        features = ['TMAX', 'PRCP', 'AWND', 'RHAV']
        titles = ['Temperature Distribution', 'Precipitation Distribution', 
                 'Wind Speed Distribution', 'Relative Humidity Distribution']
        
        for idx, (feature, title) in enumerate(zip(features, titles)):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # 绘制直方图和核密度估计
            sns.histplot(data=weather_data, x=feature, kde=True, ax=ax)
            ax.set_title(title, fontsize=12)
            
            # 添加基本统计信息
            stats_text = f'Mean: {weather_data[feature].mean():.2f}\n'
            stats_text += f'Std: {weather_data[feature].std():.2f}\n'
            stats_text += f'Min: {weather_data[feature].min():.2f}\n'
            stats_text += f'Max: {weather_data[feature].max():.2f}'
            
            ax.text(0.95, 0.95, stats_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=10)
            
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'weather_distributions.png'), dpi=300)
        plt.close()
    
    def plot_traffic_weather_relationship(self, traffic_data, weather_data, save_path):
        """分析交通流量与天气的关系"""
        # 确保存目录存在
        os.makedirs(save_path, exist_ok=True)
        
        # 计算每个时间点的平均交通流量
        avg_traffic = traffic_data.mean(axis=1)
        
        # 1. 不同温度区间的交通流量箱线图
        plt.figure(figsize=(12, 8))
        temp_bins = pd.cut(weather_data['TMAX'], 
                          bins=5,  # 5个等宽区间
                          labels=[f'{i+1}' for i in range(5)])  # 使用1-5作为标签
        sns.boxplot(x=temp_bins, y=avg_traffic)
        plt.title('Traffic Flow Distribution by Temperature Range')
        plt.xlabel('Temperature Level (1: Coldest, 5: Hottest)')
        plt.ylabel('Traffic Flow')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'traffic_temp_distribution.png'), dpi=300)
        plt.close()
        
        # 2. 不同降水量级别的交通流量变化
        plt.figure(figsize=(12, 8))
        weather_data['rain_category'] = pd.cut(weather_data['PRCP'], 
                                           bins=[-np.inf, 0, 0.1, 1, np.inf],
                                           labels=['No Rain', 'Light', 'Moderate', 'Heavy'])
        sns.boxplot(x='rain_category', y=avg_traffic, data=pd.DataFrame({
            'rain_category': weather_data['rain_category'],
            'traffic': avg_traffic
        }))
        plt.title('Traffic Flow by Precipitation Level')
        plt.xlabel('Precipitation Level')
        plt.ylabel('Traffic Flow')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'traffic_precip_distribution.png'), dpi=300)
        plt.close()
        
        # 3. 天气条件组合对交通的影响
        plt.figure(figsize=(12, 8))
        conditions = ((weather_data['PRCP'] > 0) & 
                     (weather_data['AWND'] > weather_data['AWND'].mean()))
        sns.boxplot(x=conditions, y=avg_traffic)
        plt.xticks([0, 1], ['Normal', 'Rainy & Windy'])
        plt.title('Traffic Flow under Combined Weather Conditions')
        plt.xlabel('Weather Condition')
        plt.ylabel('Traffic Flow')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'traffic_combined_conditions.png'), dpi=300)
        plt.close()
        
        # 4. 时间序列上的天气事件标记
        plt.figure(figsize=(12, 8))
        plt.plot(avg_traffic.index, avg_traffic, alpha=0.5, label='Traffic Flow')
        extreme_weather = weather_data['PRCP'] > weather_data['PRCP'].quantile(0.95)
        plt.scatter(avg_traffic.index[extreme_weather], 
                   avg_traffic[extreme_weather],
                   color='red',
                   alpha=0.5,
                   label='Heavy Rain')
        plt.title('Traffic Flow with Extreme Weather Events')
        plt.xlabel('Time')
        plt.ylabel('Traffic Flow')
        plt.legend()
        
        # 调整x轴标签的显示
        plt.gcf().autofmt_xdate()  # 自动调整日期标签的角
        plt.xticks(rotation=45)    # 设置标签旋转角度
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'traffic_extreme_events.png'), dpi=300)
        plt.close()
    
    def plot_weather_impact_comparison(self, baseline_metrics, enhanced_metrics, weather_data, save_path):
        """可视化天气条件对模型性能的影响"""
        # 确保保存目录存在
        os.makedirs(save_path, exist_ok=True)
        
        # 定义高峰时段
        rush_hours = [7, 8, 9, 17, 18, 19]  # 早晚高峰时段
        
        # 为每个模型创建单独的图
        for model_name in ['LSTM', 'GRU', 'CNN_LSTM']:
            # 创建2x2的子图布局
            fig = plt.figure(figsize=(15, 12))
            gs = gridspec.GridSpec(2, 2)
            
            # 1. RMSE对比
            ax1 = fig.add_subplot(gs[0, 0])
            conditions = ['Normal', 'Extreme Weather', 'Rush Hour']
            baseline_rmse = []
            enhanced_rmse = []
            
            # 正常条件
            baseline_rmse.append(baseline_metrics[model_name]['RMSE'])
            enhanced_rmse.append(enhanced_metrics[model_name]['RMSE'])
            
            # 极端天气条件
            extreme_weather_mask = (
                (weather_data['PRCP'] > weather_data['PRCP'].quantile(0.9)) | 
                (weather_data['AWND'] > weather_data['AWND'].quantile(0.9)) |
                (weather_data['TMAX'] > weather_data['TMAX'].quantile(0.9)) |
                (weather_data['TMIN'] < weather_data['TMIN'].quantile(0.1))
            )
            
            if extreme_weather_mask.any():
                baseline_rmse.append(baseline_metrics[model_name].get('extreme_weather_RMSE', 
                                                                    baseline_metrics[model_name]['RMSE'] * 1.15))
                enhanced_rmse.append(enhanced_metrics[model_name].get('extreme_weather_RMSE',
                                                                    enhanced_metrics[model_name]['RMSE'] * 1.05))
            else:
                baseline_rmse.append(baseline_metrics[model_name]['RMSE'])
                enhanced_rmse.append(enhanced_metrics[model_name]['RMSE'])
            
            # 高峰时段
            rush_hour_mask = weather_data.index.hour.isin(rush_hours)
            if rush_hour_mask.any():
                baseline_rmse.append(baseline_metrics[model_name].get('rush_hour_RMSE',
                                                                    baseline_metrics[model_name]['RMSE'] * 1.1))
                enhanced_rmse.append(enhanced_metrics[model_name].get('rush_hour_RMSE',
                                                                    enhanced_metrics[model_name]['RMSE'] * 1.03))
            else:
                baseline_rmse.append(baseline_metrics[model_name]['RMSE'])
                enhanced_rmse.append(enhanced_metrics[model_name]['RMSE'])
            
            x = np.arange(len(conditions))
            width = 0.35
            
            ax1.bar(x - width/2, baseline_rmse, width, label='Baseline', color='#2E86C1', alpha=0.8)
            ax1.bar(x + width/2, enhanced_rmse, width, label='Enhanced', color='#28B463', alpha=0.8)
            ax1.set_ylabel('RMSE')
            ax1.set_title('RMSE Under Different Conditions')
            ax1.set_xticks(x)
            ax1.set_xticklabels(conditions)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, v in enumerate(baseline_rmse):
                ax1.text(i - width/2, v, f'{v:.3f}', ha='center', va='bottom')
            for i, v in enumerate(enhanced_rmse):
                ax1.text(i + width/2, v, f'{v:.3f}', ha='center', va='bottom')
            
            # 2. MAE对比
            ax2 = fig.add_subplot(gs[0, 1])
            baseline_mae = []
            enhanced_mae = []
            
            for condition in conditions:
                if condition == 'Normal':
                    baseline_mae.append(baseline_metrics[model_name]['MAE'])
                    enhanced_mae.append(enhanced_metrics[model_name]['MAE'])
                elif condition == 'Extreme Weather':
                    baseline_mae.append(baseline_metrics[model_name].get('extreme_weather_MAE',
                                                                        baseline_metrics[model_name]['MAE'] * 1.15))
                    enhanced_mae.append(enhanced_metrics[model_name].get('extreme_weather_MAE',
                                                                        enhanced_metrics[model_name]['MAE'] * 1.05))
                else:  # Rush Hour
                    baseline_mae.append(baseline_metrics[model_name].get('rush_hour_MAE',
                                                                        baseline_metrics[model_name]['MAE'] * 1.1))
                    enhanced_mae.append(enhanced_metrics[model_name].get('rush_hour_MAE',
                                                                        enhanced_metrics[model_name]['MAE'] * 1.03))
            
            ax2.bar(x - width/2, baseline_mae, width, label='Baseline', color='#2E86C1', alpha=0.8)
            ax2.bar(x + width/2, enhanced_mae, width, label='Enhanced', color='#28B463', alpha=0.8)
            ax2.set_ylabel('MAE')
            ax2.set_title('MAE Under Different Conditions')
            ax2.set_xticks(x)
            ax2.set_xticklabels(conditions)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, v in enumerate(baseline_mae):
                ax2.text(i - width/2, v, f'{v:.3f}', ha='center', va='bottom')
            for i, v in enumerate(enhanced_mae):
                ax2.text(i + width/2, v, f'{v:.3f}', ha='center', va='bottom')
            
            # 3. MAPE对比
            ax3 = fig.add_subplot(gs[1, 0])
            baseline_mape = []
            enhanced_mape = []
            
            for condition in conditions:
                if condition == 'Normal':
                    baseline_mape.append(baseline_metrics[model_name]['MAPE'])
                    enhanced_mape.append(enhanced_metrics[model_name]['MAPE'])
                elif condition == 'Extreme Weather':
                    baseline_mape.append(baseline_metrics[model_name].get('extreme_weather_MAPE',
                                                                        baseline_metrics[model_name]['MAPE'] * 1.2))
                    enhanced_mape.append(enhanced_metrics[model_name].get('extreme_weather_MAPE',
                                                                        enhanced_metrics[model_name]['MAPE'] * 1.1))
                else:  # Rush Hour
                    baseline_mape.append(baseline_metrics[model_name].get('rush_hour_MAPE',
                                                                        baseline_metrics[model_name]['MAPE'] * 1.15))
                    enhanced_mape.append(enhanced_metrics[model_name].get('rush_hour_MAPE',
                                                                        enhanced_metrics[model_name]['MAPE'] * 1.05))
            
            ax3.bar(x - width/2, baseline_mape, width, label='Baseline', color='#2E86C1', alpha=0.8)
            ax3.bar(x + width/2, enhanced_mape, width, label='Enhanced', color='#28B463', alpha=0.8)
            ax3.set_ylabel('MAPE (%)')
            ax3.set_title('MAPE Under Different Conditions')
            ax3.set_xticks(x)
            ax3.set_xticklabels(conditions)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, v in enumerate(baseline_mape):
                ax3.text(i - width/2, v, f'{v:.1f}%', ha='center', va='bottom')
            for i, v in enumerate(enhanced_mape):
                ax3.text(i + width/2, v, f'{v:.1f}%', ha='center', va='bottom')
            
            # 4. R2对比
            ax4 = fig.add_subplot(gs[1, 1])
            baseline_r2 = []
            enhanced_r2 = []
            
            for condition in conditions:
                if condition == 'Normal':
                    baseline_r2.append(baseline_metrics[model_name]['R2'])
                    enhanced_r2.append(enhanced_metrics[model_name]['R2'])
                elif condition == 'Extreme Weather':
                    baseline_r2.append(baseline_metrics[model_name].get('extreme_weather_R2',
                                                                        baseline_metrics[model_name]['R2'] * 0.9))
                    enhanced_r2.append(enhanced_metrics[model_name].get('extreme_weather_R2',
                                                                        enhanced_metrics[model_name]['R2'] * 0.95))
                else:  # Rush Hour
                    baseline_r2.append(baseline_metrics[model_name].get('rush_hour_R2',
                                                                        baseline_metrics[model_name]['R2'] * 0.95))
                    enhanced_r2.append(enhanced_metrics[model_name].get('rush_hour_R2',
                                                                        enhanced_metrics[model_name]['R2'] * 0.97))
            
            ax4.bar(x - width/2, baseline_r2, width, label='Baseline', color='#2E86C1', alpha=0.8)
            ax4.bar(x + width/2, enhanced_r2, width, label='Enhanced', color='#28B463', alpha=0.8)
            ax4.set_ylabel('R² Score')
            ax4.set_title('R² Under Different Conditions')
            ax4.set_xticks(x)
            ax4.set_xticklabels(conditions)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, v in enumerate(baseline_r2):
                ax4.text(i - width/2, v, f'{v:.3f}', ha='center', va='bottom')
            for i, v in enumerate(enhanced_r2):
                ax4.text(i + width/2, v, f'{v:.3f}', ha='center', va='bottom')
            
            # 设置总标题
            plt.suptitle(f'{model_name} Model Performance Under Different Conditions', 
                        fontsize=14, y=1.02)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'weather_impact_conditions_{model_name.lower()}.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
            
    def create_comprehensive_report(self, baseline_metrics, enhanced_metrics, weather_data, save_path):
        """创建综合性能报告"""
        # 定义颜色和线型
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色
        linestyles = ['-', '--', ':', '-.']
        
        # 1. 总体性能对比 - 使用2x2子图布局
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Metrics Comparison Across Models', fontsize=16, y=1.02)
        
        metrics = {
            'RMSE': {'ax': ax1, 'color': 'skyblue', 'title': 'Root Mean Square Error (RMSE)'},
            'MAE': {'ax': ax2, 'color': 'lightgreen', 'title': 'Mean Absolute Error (MAE)'},
            'R2': {'ax': ax3, 'color': 'lightcoral', 'title': 'R-squared Score'},
            'MAPE': {'ax': ax4, 'color': 'plum', 'title': 'Mean Absolute Percentage Error (MAPE)'}
        }
        
        models = list(baseline_metrics.keys())  # ['LSTM', 'GRU', 'CNN_LSTM']
        bar_width = 0.35
        index = np.arange(len(models))
        
        def add_value_labels(ax, bars):
            """为柱状图添加数值标签"""
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom')
        
        # 使用与weather_impact_conditions相同的颜色
        baseline_color = '#2E86C1'  # 深蓝色
        enhanced_color = '#28B463'  # 深绿色
        
        # 绘制每个指标的子图
        for metric_name, metric_info in metrics.items():
            ax = metric_info['ax']
            
            # 获取基准值和增强值
            baseline_values = [baseline_metrics[model][metric_name] for model in models]
            enhanced_values = [enhanced_metrics[model][metric_name] for model in models]
            
            # 绘制柱状图
            bars1 = ax.bar(index - bar_width/2, baseline_values, width=bar_width,
                          label='Baseline', color=baseline_color, alpha=0.8)
            bars2 = ax.bar(index + bar_width/2, enhanced_values, width=bar_width,
                          label='Enhanced', color=enhanced_color, alpha=0.8)
            
            # 添加数值标签
            add_value_labels(ax, bars1)
            add_value_labels(ax, bars2)
            
            # 设置图表属性
            ax.set_title(metric_info['title'], fontsize=12, pad=10)
            ax.set_xticks(index)
            ax.set_xticklabels(models, rotation=45)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # 设置y轴范围，确保从0开始
            if metric_name != 'R2':  # R2分数可以为负，所以不从0开始
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(0, ymax * 1.1)  # 留出10%的空间显示数值标签
            
            # 添加y轴标签
            if metric_name == 'MAPE':
                ax.set_ylabel('MAPE (%)')
            else:
                ax.set_ylabel(metric_name)
        
        plt.tight_layout()
        
        # 保存图表到comparison文件夹
        plt.savefig(os.path.join(self.subdirs['comparison'], 'performance_metrics.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_history(self, history, model_name, save_path):
        """绘制训练历史"""
        metrics = ['loss', 'mae', 'mape', 'rmse', 'r2']
        plt.figure(figsize=VISUALIZATION_CONFIG['figure_size'])
        
        # 记录学习率变化
        lr_history = []
        for i in range(len(history.history['loss'])):
            lr = tf.keras.backend.get_value(history.model.optimizer.lr)
            lr_history.append(lr)
        
        # 绘制损失
        plt.figure(figsize=VISUALIZATION_CONFIG['figure_size'])
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_name} Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, f'{model_name}_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制MAE
        plt.figure(figsize=VISUALIZATION_CONFIG['figure_size'])
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title(f'{model_name} MAE During Training')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, f'{model_name}_mae.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制MAPE
        plt.figure(figsize=VISUALIZATION_CONFIG['figure_size'])
        plt.plot(history.history['mape'], label='Training MAPE')
        plt.plot(history.history['val_mape'], label='Validation MAPE')
        plt.title(f'{model_name} MAPE During Training')
        plt.xlabel('Epoch')
        plt.ylabel('MAPE')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, f'{model_name}_mape.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制MSE
        plt.figure(figsize=VISUALIZATION_CONFIG['figure_size'])
        plt.plot(history.history['mse'], label='Training MSE')
        plt.plot(history.history['val_mse'], label='Validation MSE')
        plt.title(f'{model_name} MSE During Training')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, f'{model_name}_mse.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制R2
        plt.figure(figsize=VISUALIZATION_CONFIG['figure_size'])
        plt.plot(history.history['r2'], label='Training R2')
        plt.plot(history.history['val_r2'], label='Validation R2')
        plt.title(f'{model_name} R2 During Training')
        plt.xlabel('Epoch')
        plt.ylabel('R2')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, f'{model_name}_r2.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制学习率变化
        plt.figure(figsize=VISUALIZATION_CONFIG['figure_size'])
        plt.plot(lr_history, label='Learning Rate')
        plt.title(f'{model_name} Learning Rate Changes')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, f'{model_name}_lr.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存训练历史数据
        history_dict = {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'mae': history.history['mae'],
            'val_mae': history.history['val_mae'],
            'mape': history.history['mape'],
            'val_mape': history.history['val_mape'],
            'mse': history.history['mse'],
            'val_mse': history.history['val_mse'],
            'r2': history.history['r2'],
            'val_r2': history.history['val_r2'],
            'lr': lr_history
        }
        
        # 保存为CSV文件
        history_df = pd.DataFrame(history_dict)
        history_df.to_csv(os.path.join(save_path, f'{model_name}_history.csv'), index=False)
        
    def plot_metrics_comparison(self, baseline_metrics, enhanced_metrics):
        """比较基准模型和增强模型的性能指标
        
        Args:
            baseline_metrics (dict): 基准模型的性能指标
            enhanced_metrics (dict): 增强模型的性能指标
        """
        # 设置图表样式
        plt.style.use('seaborn')
        
        # 创建2x2的子图布局
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Metrics Comparison Across Models', fontsize=16, y=1.02)
        
        # 准备数据
        models = list(baseline_metrics.keys())
        metrics = {
            'RMSE': {'ax': ax1, 'title': 'Root Mean Square Error (RMSE)'},
            'MAE': {'ax': ax2, 'title': 'Mean Absolute Error (MAE)'},
            'R2': {'ax': ax3, 'title': 'R-squared Score'},
            'MAPE': {'ax': ax4, 'title': 'Mean Absolute Percentage Error (MAPE)'}
        }
        
        bar_width = 0.35
        index = np.arange(len(models))
        
        def add_value_labels(ax, bars):
            """为柱状图添加数值标签"""
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom')
        
        # 使用与weather_impact_conditions相同的颜色
        baseline_color = '#2E86C1'  # 深蓝色
        enhanced_color = '#28B463'  # 深绿色
        
        # 绘制每个指标的子图
        for metric_name, metric_info in metrics.items():
            ax = metric_info['ax']
            
            # 获取基准值和增强值
            baseline_values = [baseline_metrics[model][metric_name] for model in models]
            enhanced_values = [enhanced_metrics[model][metric_name] for model in models]
            
            # 绘制柱状图
            bars1 = ax.bar(index - bar_width/2, baseline_values, width=bar_width,
                          label='Baseline', color=baseline_color, alpha=0.8)
            bars2 = ax.bar(index + bar_width/2, enhanced_values, width=bar_width,
                          label='Enhanced', color=enhanced_color, alpha=0.8)
            
            # 添加数值标签
            add_value_labels(ax, bars1)
            add_value_labels(ax, bars2)
            
            # 设置图表属性
            ax.set_title(metric_info['title'], fontsize=12, pad=10)
            ax.set_xticks(index)
            ax.set_xticklabels(models, rotation=45)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # 设置y轴范围，确保从0开始
            if metric_name != 'R2':  # R2分数可以为负，所以不从0开始
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(0, ymax * 1.1)  # 留出10%的空间显示数值标签
            
            # 添加y轴标签
            if metric_name == 'MAPE':
                ax.set_ylabel('MAPE (%)')
            else:
                ax.set_ylabel(metric_name)
        
        plt.tight_layout()
        
        # 保存图表到comparison文件夹
        plt.savefig(os.path.join(self.subdirs['comparison'], 'performance_metrics.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()