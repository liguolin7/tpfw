from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(y_true, y_pred, model_name):
    """评估模型性能"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} 评估结果:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2}

def plot_predictions(y_true, y_pred, model_name):
    """绘制预测结果"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='实际值')
    plt.plot(y_pred, label='预测值')
    plt.title(f'{model_name} 预测结果')
    plt.legend()
    plt.savefig(f'results/figures/{model_name}_predictions.png')
    plt.close() 