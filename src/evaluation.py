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
    
    # Save plot
    figures_dir = os.path.join(results_dir, 'figures')
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
    
    # 计算每个模型的综合得分
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
    """绘制训练和验证集的损失和 MAE 曲线"""
    epochs = range(1, len(history.history['loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'ro-', label='Validation Loss')
    plt.title(f'{model_name} Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制 MAE 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['mae'], 'bo-', label='Training MAE')
    plt.plot(epochs, history.history['val_mae'], 'ro-', label='Validation MAE')
    plt.title(f'{model_name} MAE Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, f'{model_name}_learning_curve.png')
    plt.savefig(save_path, dpi=300)
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