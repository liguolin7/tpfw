from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from config import RESULTS_DIR, RANDOM_SEED
import logging
import shap
from scipy import stats
from sklearn.inspection import permutation_importance

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model performance and return results"""
    # Handle NaN values
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    # Avoid division by zero
    epsilon = 1e-10
    y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    
    # Calculate metrics
    results = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true_safe - y_pred) / y_true_safe)) * 100
    }
    
    # Log results
    logging.info(f"\n{model_name} Model Evaluation Results:")
    for metric, value in results.items():
        logging.info(f"{metric}: {value:.4f}")
    
    return results

def plot_predictions(y_true, y_pred, model_name, results_dir):
    """Plot prediction comparison"""
    plt.figure(figsize=(15, 6))
    plt.plot(y_true.reset_index(drop=True), label='Actual', linewidth=2)
    plt.plot(y_pred, label='Predicted', linewidth=2, linestyle='--')
    plt.title(f'{model_name} Prediction Results', fontsize=14)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    
    if results_dir:
        figures_dir = os.path.join(results_dir, 'figures')
        if y_pred is not None and y_true is not None:
            os.makedirs(figures_dir, exist_ok=True)
            save_path = os.path.join(figures_dir, f'{model_name}_predictions.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_residuals(y_true, y_pred, model_name):
    """绘制残差分析图"""
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 残差散点图
    ax1.scatter(y_pred, residuals, alpha=0.5, color='blue')
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted Values', fontsize=12)
    ax1.set_ylabel('Residuals', fontsize=12)
    ax1.set_title('Residuals vs Predicted', fontsize=14)
    ax1.grid(True)
    
    # 残差分布图
    sns.histplot(residuals, kde=True, ax=ax2, color='blue')
    ax2.axvline(x=0, color='r', linestyle='--')
    ax2.set_xlabel('Residuals', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Residual Distribution', fontsize=14)
    ax2.grid(True)
    
    plt.suptitle(f'{model_name} Residual Analysis', fontsize=16, y=1.05)
    plt.tight_layout()
    
    save_path = os.path.join(RESULTS_DIR, 'figures', f'{model_name}_residuals.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 记录残差统计信息
    residuals_stats = {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'skewness': pd.Series(residuals).skew(),
        'kurtosis': pd.Series(residuals).kurtosis()
    }
    
    logging.info(f"\n{model_name} Residual Statistics:")
    for stat, value in residuals_stats.items():
        logging.info(f"{stat}: {value:.4f}")

def plot_feature_importance(model, X_test, y_test, feature_names, model_name, results_dir):
    """使用 Permutation Importance 进行特征重要性分析"""
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=RANDOM_SEED, n_jobs=-1)
    importance = result.importances_mean
    
    # 绘制特征重要性
    indices = np.argsort(importance)[::-1]
    plt.figure(figsize=(12, 6))
    plt.title(f'{model_name} 特征重要性（Permutation Importance）', fontsize=16)
    plt.bar(range(len(feature_names)), importance[indices], align='center')
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel('特征')
    plt.ylabel('重要性')
    plt.tight_layout()
    save_path = os.path.join(results_dir, f'{model_name}_feature_importance.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

def find_best_model(improvements):
    """根据性能提升找出最佳模型
    
    参数:
        improvements: 性能提升分析结果
    """
    model_scores = {}
    
    # 计算个模型的综合得分
    for model, metrics in improvements.items():
        # 根据RMSE和MAE的降低程度以及R2的提升程度计算得分
        rmse_score = metrics['rmse_improvement']
        mae_score = metrics['mae_improvement']
        r2_score = metrics['r2_improvement']
        
        # 综合得分（可以根据需要调整权重）
        model_scores[model] = (rmse_score + mae_score + r2_score) / 3
    
    # 找出得分最高的模型
    best_model = max(model_scores.items(), key=lambda x: x[1])
    
    logging.info("\n最佳模型分析结果:")
    logging.info(f"最佳模型: {best_model[0]}")
    logging.info(f"平均性能提升: {best_model[1]:.2f}%")
    
    return best_model[0]
    
def plot_model_comparison(baseline_results, enhanced_results, results_dir, metrics=['rmse', 'mae', 'r2', 'mape']):
    """绘制基础模型和增强模型的性能对比图"""
    # 确保目录存在
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
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
        plt.ylabel(metric.upper())
        plt.title(f'{metric.upper()} Comparison')
        plt.xticks(x, models)
        plt.legend()
        
        save_path = os.path.join(figures_dir, f'comparison_{metric}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def plot_prediction_distribution(y_true, y_pred, model_name):
    """绘制预测值分布图"""
    plt.figure(figsize=(15, 5))
    
    # 预测值与实际值的散点图
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction vs Actual')
    plt.grid(True)
    
    # 预测误差的分布图
    plt.subplot(1, 2, 2)
    residuals = y_pred - y_true
    sns.histplot(residuals, kde=True)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    plt.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'figures', f'{model_name}_prediction_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_series_decomposition(data, model_name):
    """绘制时间序列分解图"""
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # 执行时间序列分解
    decomposition = seasonal_decompose(data, period=24*7)  # 假设数据是每小时一个点，周期为一周
    
    plt.figure(figsize=(15, 12))
    
    # 原始数据
    plt.subplot(411)
    plt.plot(data)
    plt.title('Original Time Series')
    plt.grid(True)
    
    # 趋势
    plt.subplot(412)
    plt.plot(decomposition.trend)
    plt.title('Trend')
    plt.grid(True)
    
    # 季节性
    plt.subplot(413)
    plt.plot(decomposition.seasonal)
    plt.title('Seasonal')
    plt.grid(True)
    
    # 残差
    plt.subplot(414)
    plt.plot(decomposition.resid)
    plt.title('Residual')
    plt.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'figures', f'{model_name}_time_series_decomposition.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def analyze_weather_impact(model, X_test, y_test, feature_names):
    """分析天气特征对预测的影响
    
    参数:
        model: 训练好的模型
        X_test: 测试集特征
        y_test: 测试集标签
        feature_names: 特征名列表
    """
    # 执行排列重要性分析
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        random_state=42
    )
    
    # 提取天气相关特征的重要性
    weather_features = [f for f in feature_names if f.startswith(('temp_', 'humidity_', 'precip_', 'wind_'))]
    weather_importance = {
        feature: {
            'importance_mean': result.importances_mean[feature_names.index(feature)],
            'importance_std': result.importances_std[feature_names.index(feature)]
        }
        for feature in weather_features
    }
    
    # 绘制天气特征重要性图
    plt.figure(figsize=(12, 6))
    features = list(weather_importance.keys())
    means = [weather_importance[f]['importance_mean'] for f in features]
    stds = [weather_importance[f]['importance_std'] for f in features]
    
    plt.barh(range(len(features)), means, xerr=stds)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Feature Importance (Mean Decrease in Error)')
    plt.title('Weather Features Impact Analysis')
    
    save_path = os.path.join(RESULTS_DIR, 'figures', 'weather_impact_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return weather_importance

def detailed_model_analysis(y_true, y_pred, model_name):
    """详细的模型性能分析"""
    residuals = y_pred - y_true
    stats_results = {
        'mean_error': np.mean(residuals),
        'std_error': np.std(residuals),
        'skewness': stats.skew(residuals),
        'kurtosis': stats.kurtosis(residuals)
    }
    
    return stats_results

def shap_analysis(model, X_sample, model_name, results_dir):
    """Analyze feature importance using perturbation method"""
    try:
        base_predictions = model.predict(X_sample)
        n_features = X_sample.shape[2]
        importances = np.zeros(n_features)
        
        for i in range(n_features):
            X_perturbed = X_sample.copy()
            X_perturbed[:, :, i] = np.random.permutation(X_perturbed[:, :, i])
            perturbed_predictions = model.predict(X_perturbed)
            importances[i] = np.mean((base_predictions - perturbed_predictions) ** 2)
        
        plt.figure(figsize=(10, 6))
        feature_indices = np.argsort(importances)[::-1]
        plt.bar(range(len(importances)), importances[feature_indices])
        plt.title(f'{model_name} Feature Importance Analysis')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance Score')
        plt.xticks(range(len(importances)), feature_indices)
        plt.tight_layout()
        
        figures_dir = os.path.join(results_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        save_path = os.path.join(figures_dir, f'{model_name}_feature_importance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logging.error(f"Feature importance analysis error: {str(e)}")
        raise

def plot_training_history(history, model_name, results_dir):
    """绘制模型训练历史"""
    # 确保目录存在
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Training History - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # MAE曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f'{model_name} Training History - MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(figures_dir, f'{model_name}_training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_weather_features(df, weather_features, target='avg_speed'):
    """分析天气特征与目标变量的关系"""
    # 确保目录存在
    figures_dir = os.path.join(RESULTS_DIR, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # 相关性分析
    plt.figure(figsize=(12, 8))
    corr = df[weather_features + [target]].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Weather Features Correlation with Traffic Speed')
    save_path = os.path.join(figures_dir, 'weather_correlation.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_eda_visualizations(traffic_data, weather_data, results_dir):
    """创建数据探索分析可视化"""
    plt.style.use('seaborn')
    os.makedirs(os.path.join(results_dir, 'figures'), exist_ok=True)
    
    # 1. 交通流量时间序列图
    plt.figure(figsize=(15, 6))
    daily_traffic = traffic_data.mean(axis=1).resample('D').mean()
    plt.plot(daily_traffic.index, daily_traffic.values)
    plt.title('Daily Traffic Flow Pattern', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average Traffic Flow', fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'figures/traffic_pattern.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 相关性热力图
    plt.figure(figsize=(12, 8))
    combined_data = pd.concat([
        traffic_data.mean(axis=1).to_frame('traffic_flow'),
        weather_data
    ], axis=1)
    sns.heatmap(combined_data.corr(), annot=True, cmap='RdBu_r', center=0)
    plt.title('Weather-Traffic Correlation Heatmap', fontsize=14)
    plt.savefig(os.path.join(results_dir, 'figures/correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 天气变量分布图
    weather_features = weather_data.columns[:3]  # 选择前三个天气特征
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, feature in enumerate(weather_features):
        sns.histplot(weather_data[feature], ax=axes[i], kde=True)
        axes[i].set_title(f'{feature.capitalize()} Distribution', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'figures/weather_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_comparison_radar(baseline_results, enhanced_results, results_dir):
    """创建模型性能对比雷达图"""
    metrics = ['rmse', 'mae', 'r2', 'mape']
    models = ['LSTM', 'GRU', 'CNN_LSTM']
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for model in models:
        values = []
        for metric in metrics:
            baseline = baseline_results[model][metric]
            enhanced = enhanced_results[model][metric]
            # 对于 r2，较大值更好；对于其他指标，较小值更好
            ratio = enhanced / baseline if metric != 'r2' else baseline / enhanced
            values.append(ratio)
            
        values = np.concatenate((values, [values[0]]))
        angles_plot = np.concatenate((angles, [angles[0]]))
        
        ax.plot(angles_plot, values, '-o', linewidth=2, label=model)
        ax.fill(angles_plot, values, alpha=0.25)
    
    ax.set_xticks(angles)
    ax.set_xticklabels(metrics)
    ax.set_title('Model Performance Comparison\n(Enhanced/Baseline)', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    
    save_path = os.path.join(results_dir, 'figures/model_comparison_radar.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_comparison(y_true, y_pred_baseline, y_pred_enhanced, model_name, results_dir):
    """绘制基准模型增强模型预测结果对比"""
    plt.figure(figsize=(15, 6))
    
    # 选择前200个数据点以便清晰展示
    n_points = 200
    x = np.arange(n_points)
    
    plt.plot(x, y_true[:n_points], label='Actual', linewidth=2)
    plt.plot(x, y_pred_baseline[:n_points], '--', label='Baseline', linewidth=2)
    plt.plot(x, y_pred_enhanced[:n_points], '--', label='Enhanced', linewidth=2)
    
    plt.title(f'{model_name} Prediction Comparison', fontsize=14)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Traffic Flow', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    save_path = os.path.join(results_dir, 'figures', f'{model_name}_prediction_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_distribution(y_true, y_pred_baseline, y_pred_enhanced, model_name, results_dir):
    """绘制预测误差分布对比"""
    plt.figure(figsize=(12, 6))
    
    errors_baseline = y_true - y_pred_baseline
    errors_enhanced = y_true - y_pred_enhanced
    
    plt.hist(errors_baseline, bins=50, alpha=0.5, label='Baseline Errors', density=True)
    plt.hist(errors_enhanced, bins=50, alpha=0.5, label='Enhanced Errors', density=True)
    
    plt.title(f'{model_name} Prediction Error Distribution', fontsize=14)
    plt.xlabel('Prediction Error', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    save_path = os.path.join(results_dir, 'figures', f'{model_name}_error_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, feature_names, results_dir):
    """绘制特征重要性分析图"""
    plt.figure(figsize=(12, 6))
    
    # 获取特征重要性分数
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.title('Feature Importance Analysis', fontsize=14)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance Score', fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(results_dir, 'figures', 'feature_importance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_seasonal_analysis(traffic_data, weather_data, results_dir):
    """绘制交通流量的季节性分析图"""
    # 确保目录存在
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # 按小时分析
    plt.subplot(2, 1, 1)
    hourly_pattern = traffic_data.groupby(traffic_data.index.hour).mean()
    plt.plot(hourly_pattern.index, hourly_pattern.values, marker='o')
    plt.title('Average Traffic Flow by Hour', fontsize=14)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Average Traffic Flow', fontsize=12)
    plt.grid(True)
    
    # 按星期分析
    plt.subplot(2, 1, 2)
    weekly_pattern = traffic_data.groupby(traffic_data.index.dayofweek).mean()
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    plt.plot(weekly_pattern.index, weekly_pattern.values, marker='o')
    plt.xticks(range(7), days, rotation=45)
    plt.title('Average Traffic Flow by Day of Week', fontsize=14)
    plt.xlabel('Day of Week', fontsize=12)
    plt.ylabel('Average Traffic Flow', fontsize=12)
    plt.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(figures_dir, 'seasonal_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_feature_importance(model, X_train, y_train, feature_names, results_dir):
    """分析特征重要性"""
    plt.figure(figsize=(15, 8))
    
    # 使用排列重要性
    perm_importance = permutation_importance(
        model, X_train, y_train,
        n_repeats=10,
        random_state=42
    )
    
    # 获取特征重要性分数
    importances = pd.DataFrame(
        {'feature': feature_names,
         'importance': perm_importance.importances_mean}
    )
    importances = importances.sort_values('importance', ascending=False)
    
    # 绘制特征重要性条形图
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importances.head(20), x='importance', y='feature')
    plt.title('Top 20 Most Important Features', fontsize=14)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(results_dir, 'figures', 'feature_importance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return importances
