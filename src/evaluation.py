from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from config import RESULTS_DIR
import logging

def evaluate_model(y_true, y_pred, model_name):
    """评估模型性能并返回结果"""
    results = {
        'test_rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'test_mae': mean_absolute_error(y_true, y_pred),
        'test_r2': r2_score(y_true, y_pred)
    }
    
    # 记录评估结果
    logging.info(f"\n{model_name} Evaluation Results:")
    logging.info(f"MAE: {results['test_mae']:.4f}")
    logging.info(f"RMSE: {results['test_rmse']:.4f}")
    logging.info(f"R2: {results['test_r2']:.4f}")
    
    return results

def plot_predictions(y_true, y_pred, model_name):
    """Plot prediction results comparison"""
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
    """Plot residual analysis"""
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Residual scatter plot
    ax1.scatter(y_pred, residuals, alpha=0.5, color='blue')
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted Values', fontsize=12)
    ax1.set_ylabel('Residuals', fontsize=12)
    ax1.set_title('Residuals vs Predicted', fontsize=14)
    ax1.grid(True)
    
    # Residual distribution
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
    
    # Record residual statistics
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
    """Plot feature importance (only applicable to random forest models)
    
    Args:
        model: Trained model
        feature_names: List of feature names
        model_name: Model name
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
    
    Args:
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
    