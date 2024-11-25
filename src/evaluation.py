from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from config import RESULTS_DIR
import logging
import shap
from scipy import stats
from sklearn.inspection import permutation_importance

def evaluate_model(y_true, y_pred, model_name):
    """评估模型性能并返回结果"""
    results = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred)
    }
    
    # 只输出一次评估结果
    logging.info(f"\n{model_name} 模型评估结果:")
    for metric, value in results.items():
        logging.info(f"{metric}: {value:.4f}")
    
    return results

def plot_predictions(y_true, y_pred, model_name):
    """绘制预测结果对比图"""
    plt.figure(figsize=(15, 8))
    plt.plot(y_true.index[-100:], y_true[-100:], label='Actual', linewidth=2)
    plt.plot(y_true.index[-100:], y_pred[-100:], label='Predicted', linewidth=2, linestyle='--')
    
    plt.title(f'{model_name} Prediction Results (Last 100 Points)', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Speed', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    save_path = os.path.join(RESULTS_DIR, 'figures', f'{model_name}_predictions.png')
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

def plot_feature_importance(model, feature_names, model_name):
    """绘制特征重要性图（仅适用于随机森林模型）
    
    参数:
        model: 训练好的模型
        feature_names: 特征名称列表
        model_name: 模型名称
    """
    if not hasattr(model, 'feature_importances_'):
        logging.warning(f"{model_name} does not support feature importance analysis")
        return
        
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    importances = importances.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importances.head(10), x='importance', y='feature')
    plt.title(f'{model_name} Top 10 Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    
    save_path = os.path.join(RESULTS_DIR, 'figures', f'{model_name}_feature_importance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
    
def plot_model_comparison(baseline_results, enhanced_results, metrics=['rmse', 'mae', 'r2', 'mape']):
    """绘制基础模型和增强模型的性能对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        models = list(baseline_results.keys())
        baseline_values = [baseline_results[model][metric] for model in models]
        enhanced_values = [enhanced_results[model][metric] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        # 添加百分比改进标签
        improvements = [(e - b) / b * 100 if metric != 'r2' else (abs(e) - abs(b)) / abs(b) * 100
                       for b, e in zip(baseline_values, enhanced_values)]
        
        bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', color='skyblue')
        bars2 = ax.bar(x + width/2, enhanced_values, width, label='Enhanced', color='lightgreen')
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom')
        
        # 添加改进百分比
        for idx, imp in enumerate(improvements):
            ax.text(x[idx], max(baseline_values[idx], enhanced_values[idx]),
                   f'{imp:+.2f}%',
                   ha='center', va='bottom', color='red')
        
        ax.set_title(f'{metric.upper()} Comparison', fontsize=14)
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'figures', 'model_comparison.png')
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

def shap_analysis(model, X_test, feature_names, model_name):
    """改进的SHAP值分析"""
    # 根据模型类型选择合适的解释器
    if model_name == 'LinearRegression':
        explainer = shap.LinearExplainer(
            model, 
            shap.sample(X_test, 100),
            feature_names=feature_names
        )
    elif model_name == 'RandomForest':
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation='interventional',
            feature_names=feature_names
        )
    else:
        return None
    
    # 计算SHAP值
    shap_values = explainer.shap_values(X_test)
    
    # 创建多个SHAP可视化
    plt.figure(figsize=(15, 10))
    
    # 1. 特征重要性总结图
    plt.subplot(211)
    shap.summary_plot(
        shap_values, 
        X_test,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.title(f'{model_name} SHAP Feature Importance')
    
    # 2. 特征影响图
    plt.subplot(212)
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        show=False
    )
    plt.title(f'{model_name} SHAP Feature Impact')
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'figures', f'{model_name}_shap_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return shap_values    

def plot_training_history(history, model_name):
    """绘制模型训练历史
    
    Args:
        history: 模型训练历史对象
        model_name: 模型名称
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title(f'{model_name} Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # MAE曲线
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title(f'{model_name} Training MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'figures', f'{model_name}_training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def analyze_weather_features(df, weather_features, target='avg_speed'):
    """分析天气特征与目标变量的关系
    
    Args:
        df: 包含天气特征的数据框
        weather_features: 天气特征列表
        target: 目标变量名称
    """
    # 相关性分析
    plt.figure(figsize=(12, 8))
    corr = df[weather_features + [target]].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Weather Features Correlation with Traffic Speed')
    save_path = os.path.join(RESULTS_DIR, 'figures', 'weather_correlation.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 天气特征箱线图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, feature in enumerate(weather_features[:4]):
        sns.boxplot(data=df, y=feature, ax=axes[idx])
        axes[idx].set_title(f'Distribution of {feature}')
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'figures', 'weather_distributions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 散点图矩阵
    sns.pairplot(df[weather_features + [target]], diag_kind='kde')
    save_path = os.path.join(RESULTS_DIR, 'figures', 'weather_pairplot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    