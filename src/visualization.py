import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config import VISUALIZATION_CONFIG
import os
import matplotlib.gridspec as gridspec

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
        
    def plot_traffic_patterns(self, traffic_data, save_path=None):
        """可视化交通流量模式"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size)
        
        # 每日交通流量模式
        hourly_pattern = traffic_data.groupby(traffic_data.index.hour).mean().mean(axis=1)
        ax1.plot(hourly_pattern.index, hourly_pattern.values)
        ax1.set_title('Daily Traffic Pattern')
        ax1.set_xlabel('Hour')
        ax1.set_ylabel('Average Traffic Flow')
        
        # 每周交通流量模式
        weekly_pattern = traffic_data.groupby(traffic_data.index.dayofweek).mean().mean(axis=1)
        ax2.plot(range(7), weekly_pattern.values)
        ax2.set_xticks(range(7))
        ax2.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        ax2.set_title('Weekly Traffic Pattern')
        ax2.set_ylabel('Average Traffic Flow')
        
        # 交通流量热力图
        sample_data = traffic_data.iloc[:24, :10]
        sns.heatmap(sample_data.T, ax=ax3, cmap='YlOrRd')
        ax3.set_title('Traffic Flow Heatmap')
        ax3.set_xlabel('Hour')
        ax3.set_ylabel('Sensor ID')
        
        # 交通流量分布
        sns.histplot(traffic_data.values.flatten(), ax=ax4, bins=50)
        ax4.set_title('Traffic Flow Distribution')
        ax4.set_xlabel('Traffic Flow')
        ax4.set_ylabel('Count')
        
        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, 'traffic_patterns.png'), dpi=self.dpi)
            plt.close()
        return fig
    
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
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size)
        
        # 训练损失
        for model_name, history in history_dict.items():
            ax1.plot(history.history['loss'], label=f'{model_name}_train')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # 验证损失
        for model_name, history in history_dict.items():
            ax2.plot(history.history['val_loss'], label=f'{model_name}_val')
        ax2.set_title('Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        # 训练指标
        for model_name, history in history_dict.items():
            for metric in ['mae', 'mse']:
                if metric in history.history:
                    ax3.plot(history.history[metric], label=f'{model_name}_{metric}')
        ax3.set_title('Training Metrics')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Value')
        ax3.legend()
        
        # 学习率变化
        for model_name, history in history_dict.items():
            if 'lr' in history.history:
                ax4.plot(history.history['lr'], label=f'{model_name}_lr')
        ax4.set_title('Learning Rate Changes')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.legend()
        
        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, 'model_performance.png'), dpi=self.dpi)
            plt.close()
        return fig
    
    def plot_prediction_comparison(self, y_true, predictions_dict, save_path=None):
        """可视化不同模型的预测结果对比"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size)
        
        # 预测值与实际值对比
        ax1.plot(y_true[:100], 'k-', label='Actual', alpha=0.7)
        for model_name, pred in predictions_dict.items():
            ax1.plot(pred[:100], '--', label=f'{model_name}_pred', alpha=0.7)
        ax1.set_title('Prediction vs Actual')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Value')
        ax1.legend()
        
        # 误差分布
        for model_name, pred in predictions_dict.items():
            errors = y_true - pred
            sns.histplot(errors, ax=ax2, label=f'{model_name}_error', alpha=0.5)
        ax2.set_title('Error Distribution')
        ax2.set_xlabel('Error')
        ax2.set_ylabel('Count')
        ax2.legend()
        
        # 预测散点图
        for model_name, pred in predictions_dict.items():
            ax3.scatter(y_true, pred, alpha=0.5, label=model_name)
        ax3.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
        ax3.set_title('Prediction Scatter Plot')
        ax3.set_xlabel('Actual')
        ax3.set_ylabel('Predicted')
        ax3.legend()
        
        # 累积误差
        for model_name, pred in predictions_dict.items():
            cum_error = np.cumsum(np.abs(y_true - pred))
            ax4.plot(cum_error, label=model_name)
        ax4.set_title('Cumulative Error')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Cumulative Error')
        ax4.legend()
        
        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, 'prediction_comparison.png'), dpi=self.dpi)
            plt.close()
        return fig
    
    def plot_model_comparison(self, metrics_dict, save_path=None):
        """可视化不同模型的性能指标对比"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size)
        
        metrics = ['rmse', 'mae', 'r2', 'mape']
        positions = [(ax1, 'RMSE'), (ax2, 'MAE'), (ax3, 'R²'), (ax4, 'MAPE')]
        
        for ax, metric in zip(positions, metrics):
            values = [metrics_dict[model][metric] for model in metrics_dict]
            models = list(metrics_dict.keys())
            ax[0].bar(models, values)
            ax[0].set_title(f'{metric.upper()} Comparison')
            ax[0].set_xticklabels(models, rotation=45)
            ax[0].set_ylabel(metric.upper())
        
        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, 'model_comparison.png'), dpi=self.dpi)
            plt.close()
        return fig
    
    def plot_model_improvements(self, baseline_metrics, enhanced_metrics, save_path=None):
        """Plot model performance improvements comparison"""
        metrics = ['rmse', 'mae', 'r2', 'mape']
        models = list(baseline_metrics.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            baseline_values = [baseline_metrics[model][metric] for model in models]
            enhanced_values = [enhanced_metrics[model][metric] for model in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            # Plot bars
            bars1 = axes[idx].bar(x - width/2, baseline_values, width, label='Baseline', color='skyblue', alpha=0.8)
            bars2 = axes[idx].bar(x + width/2, enhanced_values, width, label='Enhanced', color='lightcoral', alpha=0.8)
            
            # Add value labels
            def add_value_labels(bars, offset=0):
                for bar in bars:
                    height = bar.get_height()
                    value = height
                    # Format values differently for different metrics
                    if metric == 'mape':
                        value_text = f'{value:.1f}'
                    elif metric == 'r2':
                        value_text = f'{value:.3f}'
                    else:
                        value_text = f'{value:.4f}'
                    
                    # Adjust label position based on value
                    if metric == 'mape':
                        y_pos = height + offset
                        va = 'bottom'
                    else:
                        if height < 0:
                            y_pos = height - offset
                            va = 'top'
                        else:
                            y_pos = height + offset
                            va = 'bottom'
                    
                    axes[idx].text(
                        bar.get_x() + bar.get_width()/2,
                        y_pos,
                        value_text,
                        ha='center',
                        va=va,
                        fontsize=9,
                        rotation=0
                    )
            
            # Add labels for baseline and enhanced models
            max_value = max(max(baseline_values), max(enhanced_values))
            offset = max_value * 0.02
            add_value_labels(bars1, offset)
            add_value_labels(bars2, offset)
            
            # Set titles and labels
            axes[idx].set_title(f'{metric.upper()} Comparison', fontsize=12, pad=20)
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(models, rotation=45)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            
            # Adjust y-axis limits
            if metric == 'mape':
                axes[idx].set_ylim(0, max_value * 1.25)
            else:
                current_ymin, current_ymax = axes[idx].get_ylim()
                axes[idx].set_ylim(current_ymin, current_ymax * 1.15)
            
            # Add improvement percentage labels
            for i in range(len(models)):
                baseline = baseline_values[i]
                enhanced = enhanced_values[i]
                if metric == 'r2':
                    improvement = (enhanced - baseline) / abs(baseline) * 100
                else:
                    improvement = (baseline - enhanced) / baseline * 100
                
                # Adjust improvement label position
                max_height = max(baseline_values[i], enhanced_values[i])
                if metric == 'mape':
                    y_pos = max_height * 1.15
                else:
                    y_pos = max_height * 1.08
                
                # Use unicode arrows instead of Chinese characters
                axes[idx].text(
                    i, 
                    y_pos,
                    f'↑{improvement:.1f}%' if improvement > 0 else f'↓{abs(improvement):.1f}%',
                    ha='center',
                    va='bottom',
                    color='green' if improvement > 0 else 'red',
                    fontsize=10,
                    fontweight='bold'
                )
        
        plt.suptitle('Model Performance Comparison (Baseline vs Enhanced)', fontsize=14, y=0.95)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, 'model_improvements.png'), dpi=300, bbox_inches='tight')
            plt.close()
        return fig
    
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
    
    def plot_feature_importance(self, feature_importance, feature_names, save_path=None):
        """Visualize feature importance"""
        # Sort feature importance
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        
        plt.figure(figsize=(10, 6))
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, np.array(feature_names)[sorted_idx])
        plt.xlabel('Feature Importance')
        plt.title('Impact of Weather Features on Traffic Flow')
        
        # Add value labels
        for i, v in enumerate(feature_importance[sorted_idx]):
            plt.text(v, i, f'{v:.4f}', va='center')
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
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
    
    def create_presentation_summary(self, baseline_metrics, enhanced_metrics, save_path=None):
        """Create comprehensive summary for presentation"""
        plt.figure(figsize=(15, 10))
        
        # Create 3x2 subplot layout
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.5])
        
        # 1. Performance improvement overview
        ax1 = plt.subplot(gs[0, :])
        models = list(baseline_metrics.keys())
        metrics = ['RMSE', 'MAE', 'R2', 'MAPE']
        improvements = []
        
        for metric in metrics:
            metric_improvements = []
            for model in models:
                baseline = baseline_metrics[model][metric.lower()]
                enhanced = enhanced_metrics[model][metric.lower()]
                if metric == 'r2':
                    imp = (enhanced - baseline) / abs(baseline) * 100
                else:
                    imp = (baseline - enhanced) / baseline * 100
                metric_improvements.append(imp)
            improvements.append(metric_improvements)
        
        im = ax1.imshow(improvements, cmap='RdYlGn', aspect='auto')
        ax1.set_xticks(range(len(models)))
        ax1.set_yticks(range(len(metrics)))
        ax1.set_xticklabels(models)
        ax1.set_yticklabels(metrics)
        plt.colorbar(im, ax=ax1, label='Improvement (%)')
        
        # Add value labels
        for i in range(len(metrics)):
            for j in range(len(models)):
                text = ax1.text(j, i, f'{improvements[i][j]:.1f}%',
                              ha="center", va="center", color="black")
        
        ax1.set_title('Model Performance Improvement Heatmap')
        
        # 2. Key findings
        ax2 = plt.subplot(gs[1, :])
        ax2.axis('off')
        findings = [
            "1. CNN-LSTM model shows best performance across all metrics",
            "2. All models show significant improvement with weather features",
            "3. MAPE shows most notable improvement, indicating more stable predictions",
            "4. Room for improvement in extreme weather conditions"
        ]
        
        ax2.text(0.05, 0.8, "Key Findings:", fontsize=12, fontweight='bold')
        for i, finding in enumerate(findings):
            ax2.text(0.05, 0.6-i*0.2, finding, fontsize=10)
        
        # 3. Recommendations
        ax3 = plt.subplot(gs[2, :])
        ax3.axis('off')
        recommendations = [
            "• Recommend using CNN-LSTM model for practical applications",
            "• Consider collecting more extreme weather data to improve robustness",
            "• Consider ensemble learning methods for further accuracy improvement"
        ]
        
        ax3.text(0.05, 0.8, "Recommendations:", fontsize=12, fontweight='bold')
        for i, rec in enumerate(recommendations):
            ax3.text(0.05, 0.6-i*0.2, rec, fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'presentation_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()