import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from config import *
import os

def train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
    """训练和评估多个模型
    
    Args:
        X_train, X_val, X_test: 特征数据
        y_train, y_val, y_test: 目标变量
        feature_names: 特征名列表
    Returns:
        评估结果字典
    """
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)
    }
    
    results = []
    
    for name, model in models.items():
        logging.info(f"训练{name}模型...")
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 在验证集上评估
        val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_mae = mean_absolute_error(y_val, val_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        # 在测试集上评估
        test_pred = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mae = mean_absolute_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 保存特征重要性
        feature_importance.to_csv(
            os.path.join(RESULTS_DIR, 'feature_importance', f'{name}_feature_importance.csv')
        )
        
        results.append({
            'model': name,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_r2': val_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2
        })
        
        logging.info(f"{name}模型评估完成")
    
    return results 