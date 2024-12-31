from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from .config import RESULTS_DIR, RANDOM_SEED
import logging
import shap
from scipy import stats
from sklearn.inspection import permutation_importance

def evaluate_model(y_true, y_pred, model_name, model=None, feature_names=None, history=None):
    """Evaluate model performance"""
    results = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        'errors': y_pred - y_true  # Save prediction errors
    }
    
    # Save training history
    if history is not None:
        results['history'] = history
    
    # Calculate feature importance
    if model is not None and feature_names is not None:
        try:
            # Calculate feature importance using model weights
            feature_importance = {}
            for layer in model.layers:
                if 'dense' in layer.name.lower():
                    weights = layer.get_weights()[0]
                    importance = np.abs(weights).mean(axis=1)
                    # Ensure feature names and importance scores have the same length
                    min_len = min(len(feature_names), len(importance))
                    for i in range(min_len):
                        feature_importance[feature_names[i]] = float(importance[i])
                    break  # Only use the weights of the first dense layer
            
            if feature_importance:
                results['feature_importance'] = feature_importance
        except Exception as e:
            logging.warning(f"Error occurred when calculating feature importance: {str(e)}")
    
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
    
    # Residual distribution plot
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

def plot_feature_importance(model, X_test, y_test, feature_names, model_name, results_dir):
    """Use Permutation Importance for feature importance analysis"""
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=RANDOM_SEED, n_jobs=-1)
    importance = result.importances_mean
    
    # Plot feature importance
    indices = np.argsort(importance)[::-1]
    plt.figure(figsize=(12, 6))
    plt.title(f'{model_name} Feature Importance (Permutation Importance)', fontsize=16)
    plt.bar(range(len(feature_names)), importance[indices], align='center')
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    save_path = os.path.join(results_dir, f'{model_name}_feature_importance.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

def find_best_model(improvements):
    """Find the best model based on performance improvement"""
    model_scores = {}
    
    # Calculate the overall score of each model
    for model, metrics in improvements.items():
        # Calculate the score based on the decrease in RMSE and MAE and the increase in R2
        rmse_score = metrics['RMSE_improvement']
        mae_score = metrics['MAE_improvement']
        r2_score = metrics['R2_improvement']
        
        # Overall score (weights can be adjusted as needed)
        model_scores[model] = (rmse_score + mae_score + r2_score) / 3
    
    # Find the model with the highest score
    best_model = max(model_scores.items(), key=lambda x: x[1])
    
    logging.info("\nBest Model Analysis Results:")
    logging.info(f"Best Model: {best_model[0]}")
    logging.info(f"Average Performance Improvement: {best_model[1]:.2f}%")
    
    return best_model[0]
    
def plot_model_comparison(baseline_results, enhanced_results, results_dir, metrics=['rmse', 'mae', 'r2', 'mape']):
    """Plot the performance comparison of the baseline model and the enhanced model"""
    # Ensure the directory exists
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
    """Plot the prediction distribution"""
    plt.figure(figsize=(15, 5))
    
    # Scatter plot of predictions vs actual values
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction vs Actual')
    plt.grid(True)
    
    # Prediction error distribution plot
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
    """Plot the time series decomposition"""
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Perform time series decomposition
    decomposition = seasonal_decompose(data, period=24*7)  # Assuming the data is one point per hour, with a period of
    
    plt.figure(figsize=(15, 12))
    
    # Original data
    plt.subplot(411)
    plt.plot(data)
    plt.title('Original Time Series')
    plt.grid(True)
    
    # Trend
    plt.subplot(412)
    plt.plot(decomposition.trend)
    plt.title('Trend')
    plt.grid(True)
    
    # Seasonality
    plt.subplot(413)
    plt.plot(decomposition.seasonal)
    plt.title('Seasonal')
    plt.grid(True)
    
    # Residuals
    plt.subplot(414)
    plt.plot(decomposition.resid)
    plt.title('Residual')
    plt.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'figures', f'{model_name}_time_series_decomposition.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def analyze_weather_impact(baseline_results, enhanced_results, weather_data):
    """Analyze the impact of weather factors"""
    impact_analysis = {
        'overall_improvement': {},
        'weather_condition_impact': {},
        'feature_importance': {}
    }
    
    # 1. Calculate the overall performance improvement
    for metric in ['rmse', 'mae', 'mape']:
        improvement = (
            baseline_results[metric] - enhanced_results[metric]
        ) / baseline_results[metric] * 100
        impact_analysis['overall_improvement'][metric] = improvement
    
    # 2. Analyze the impact under different weather conditions
    weather_conditions = weather_data['condition'].unique()
    for condition in weather_conditions:
        condition_mask = weather_data['condition'] == condition
        condition_impact = calculate_condition_impact(
            baseline_results,
            enhanced_results,
            condition_mask
        )
        impact_analysis['weather_condition_impact'][condition] = condition_impact
    
    # 3. Feature importance analysis
    weather_features = [
        'temperature', 'precipitation', 'wind_speed', 'humidity'
    ]
    impact_analysis['feature_importance'] = analyze_feature_importance(
        enhanced_results,
        weather_data[weather_features]
    )
    
    return impact_analysis

def detailed_model_analysis(y_true, y_pred, model_name):
    """Detailed model performance analysis"""
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
    """Plot the model training history"""
    # Ensure the directory exists
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Training History - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # MAE curve
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
    """Analyze the relationship between weather features and the target variable"""
    # Ensure the directory exists
    figures_dir = os.path.join(RESULTS_DIR, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Correlation analysis
    plt.figure(figsize=(12, 8))
    corr = df[weather_features + [target]].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Weather Features Correlation with Traffic Speed')
    save_path = os.path.join(figures_dir, 'weather_correlation.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_eda_visualizations(traffic_data, weather_data, results_dir):
    """Create data exploration analysis visualizations"""
    plt.style.use('seaborn')
    os.makedirs(os.path.join(results_dir, 'figures'), exist_ok=True)
    
    # 1. Traffic flow time series plot
    plt.figure(figsize=(15, 6))
    daily_traffic = traffic_data.mean(axis=1).resample('D').mean()
    plt.plot(daily_traffic.index, daily_traffic.values)
    plt.title('Daily Traffic Flow Pattern', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average Traffic Flow', fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'figures/traffic_pattern.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation heatmap
    plt.figure(figsize=(12, 8))
    combined_data = pd.concat([
        traffic_data.mean(axis=1).to_frame('traffic_flow'),
        weather_data
    ], axis=1)
    sns.heatmap(combined_data.corr(), annot=True, cmap='RdBu_r', center=0)
    plt.title('Weather-Traffic Correlation Heatmap', fontsize=14)
    plt.savefig(os.path.join(results_dir, 'figures/correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Weather variable distribution plot
    weather_features = weather_data.columns[:3]  # Select the first three weather features
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, feature in enumerate(weather_features):
        sns.histplot(weather_data[feature], ax=axes[i], kde=True)
        axes[i].set_title(f'{feature.capitalize()} Distribution', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'figures/weather_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_comparison_radar(baseline_results, enhanced_results, results_dir):
    """Create a radar chart for model performance comparison"""
    metrics = ['rmse', 'mae', 'r2', 'mape']
    models = ['LSTM', 'GRU', 'CNN_LSTM']
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for model in models:
        values = []
        for metric in metrics:
            baseline = baseline_results[model][metric]
            enhanced = enhanced_results[model][metric]
            # For r2, a larger value is better; for other metrics, a smaller value is better
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
    """Plot the comparison of baseline and enhanced model prediction results"""
    plt.figure(figsize=(15, 6))
    
    # Select the first 200 data points for clear display
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
    """Plot the comparison of prediction error distributions"""
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
    """Plot the feature importance analysis chart"""
    plt.figure(figsize=(12, 6))
    
    # Get feature importance scores
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
    """Plot the seasonal analysis chart of traffic flow"""
    # Ensure the directory exists
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # Hourly analysis
    plt.subplot(2, 1, 1)
    hourly_pattern = traffic_data.groupby(traffic_data.index.hour).mean()
    plt.plot(hourly_pattern.index, hourly_pattern.values, marker='o')
    plt.title('Average Traffic Flow by Hour', fontsize=14)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Average Traffic Flow', fontsize=12)
    plt.grid(True)
    
    # Weekly analysis
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
    """Analyze feature importance"""
    plt.figure(figsize=(15, 8))
    
    # Using permutation importance
    perm_importance = permutation_importance(
        model, X_train, y_train,
        n_repeats=10,
        random_state=42
    )
    
    # Get feature importance scores
    importances = pd.DataFrame(
        {'feature': feature_names,
         'importance': perm_importance.importances_mean}
    )
    importances = importances.sort_values('importance', ascending=False)
    
    # Plot feature importance bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importances.head(20), x='importance', y='feature')
    plt.title('Top 20 Most Important Features', fontsize=14)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    # Save the image
    save_path = os.path.join(results_dir, 'figures', 'feature_importance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return importances

def evaluate_models(predictions, y_true):
    """Evaluate model performance"""
    results = {}
    
    for model_name, y_pred in predictions.items():
        # Calculate evaluation metrics
        results[model_name] = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        # Print evaluation results
        logging.info(f"\n{model_name} Model Evaluation Results:")
        for metric, value in results[model_name].items():
            logging.info(f"{metric}: {value:.4f}")
    
    return results

def calculate_improvements(baseline_results, enhanced_results):
    """Calculate performance improvements"""
    improvements = {}
    
    for model_name in baseline_results.keys():
        improvements[model_name] = {}
        metrics = ['RMSE', 'MAE', 'R2', 'MAPE']  # Use uppercase metric names
        
        for metric in metrics:
            baseline = baseline_results[model_name][metric]
            enhanced = enhanced_results[model_name][metric]
            
            # For R2, the improvement calculation is different
            if metric == 'R2':
                improvement = (enhanced - baseline) * 100
            else:
                improvement = (baseline - enhanced) / baseline * 100
                
            improvements[model_name][f'{metric}_improvement'] = improvement  # Keep uppercase
    
    return improvements
