import os
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from data_loader import load_traffic_data, load_weather_data
from data_processor import DataProcessor
from models import BaselineModels, EnhancedModels
from evaluation import evaluate_model, evaluate_models, calculate_improvements
from visualization import DataVisualizer
from config import (
    RESULTS_DIR, 
    TRAIN_RATIO, 
    VAL_RATIO, 
    TEST_RATIO,
    DATA_CONFIG,
    MODEL_CONFIG,
    TRAINING_CONFIG
)

# 设置环境变量以禁用警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 禁止 TensorFlow 日志
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # 只使用第一个 GPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用 oneDNN 优化
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE'] = '0'

# 禁用所有 Python 警告
import warnings
warnings.filterwarnings('ignore')

# 设置 TensorFlow 日志级别
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# 禁用 TensorFlow 的执行器警告
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# 禁用 NumPy 的警告
np.seterr(all='ignore')

# 禁用 Pandas 的警告
pd.options.mode.chained_assignment = None

# 设置 GPU 内存分配策略
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            # 启用内存增长
            tf.config.experimental.set_memory_growth(device, True)
            
            # 禁用 TensorFlow 32
            tf.config.experimental.enable_tensor_float_32_execution(False)
            
            # 设置优化器选项
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            
            # 设置计算精度
            tf.keras.backend.set_floatx('float32')
            
            # 配置 GPU 选项
            tf.config.optimizer.set_jit(False)  # 禁用 XLA
            tf.config.optimizer.set_experimental_options({
                'layout_optimizer': True,
                'constant_folding': True,
                'shape_optimization': True,
                'remapping': True,
                'arithmetic_optimization': True,
                'dependency_optimization': True,
                'loop_optimization': True,
                'function_optimization': True,
                'debug_stripper': True,
            })
    except RuntimeError as e:
        print(e)

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)

def setup_directories():
    """创建必要的目录结构"""
    directories = [
        os.path.join(RESULTS_DIR, 'figures'),
        os.path.join(RESULTS_DIR, 'figures', 'traffic'),
        os.path.join(RESULTS_DIR, 'figures', 'weather'),
        os.path.join(RESULTS_DIR, 'figures', 'models'),
        os.path.join(RESULTS_DIR, 'figures', 'comparison'),
        os.path.join(RESULTS_DIR, 'figures', 'analysis')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def run_experiment():
    """运行实验并返回结果"""
    try:
        data_processor = DataProcessor()
        
        # 加载数据
        traffic_data = load_traffic_data()
        weather_data = load_weather_data()
        
        # 准备基准模型数据
        X_train, y_train, X_val, y_val, X_test, y_test = data_processor.prepare_sequences(
            traffic_data=traffic_data,
            sequence_length=DATA_CONFIG['sequence_length']
        )
        
        # 准备增强模型数据（包��天气特征）
        X_train_enhanced, y_train_enhanced, X_val_enhanced, y_val_enhanced, X_test_enhanced, y_test_enhanced = \
            data_processor.prepare_sequences(
                traffic_data=traffic_data,
                weather_data=weather_data,
                sequence_length=DATA_CONFIG['sequence_length']
            )
        
        # 生成特征名称
        feature_names = []
        # 生成基本特征名称
        for i in range(DATA_CONFIG['sequence_length']):
            feature_names.extend([
                f'traffic_t-{i}',
                f'temp_t-{i}',
                f'precip_t-{i}',
                f'wind_t-{i}',
                f'humidity_t-{i}'
            ])

        # 添加额外特征名称
        feature_names.extend([
            'hour_sin',
            'hour_cos',
            'day_sin',
            'day_cos',
            'is_weekend',
            'is_holiday'
        ])

        # 确保特征名称列表长度与输入维度匹配
        input_shape = X_train_enhanced.shape[-1]
        if len(feature_names) < input_shape:
            feature_names.extend([f'feature_{i}' for i in range(len(feature_names), input_shape)])
        elif len(feature_names) > input_shape:
            feature_names = feature_names[:input_shape]
        
        # 对增强模型的训练数据���行数据增强
        X_train_enhanced, y_train_enhanced = data_processor.augment_weather_data(
            X_train_enhanced, y_train_enhanced
        )
        
        # 存储实验结果
        baseline_metrics = {}
        enhanced_metrics = {}
        baseline_predictions = {}
        enhanced_predictions = {}
        
        # 训练和评估基准模型
        baseline_models = BaselineModels()
        for model_name in ['LSTM', 'GRU', 'CNN_LSTM']:
            model, history = baseline_models.train_model(
                model_name, X_train, y_train, X_val, y_val
            )
            predictions = model.predict(X_test)
            metrics = evaluate_model(y_test, predictions.flatten(), model_name)
            baseline_metrics[model_name] = metrics
            baseline_predictions[model_name] = predictions.flatten()
        
        # 训练和评估增强模型
        enhanced_models = EnhancedModels()
        for model_name in ['LSTM', 'GRU', 'CNN_LSTM']:
            model, history = enhanced_models.train_model(
                model_name, X_train_enhanced, y_train_enhanced, 
                X_val_enhanced, y_val_enhanced
            )
            predictions = model.predict(X_test_enhanced)
            metrics = evaluate_model(
                y_test, 
                predictions.flatten(), 
                model_name,
                model=enhanced_models.models[model_name],
                feature_names=feature_names
            )
            enhanced_metrics[model_name] = metrics
            enhanced_predictions[model_name] = predictions.flatten()
        
        # 计算性能提升
        improvements = calculate_improvements(baseline_metrics, enhanced_metrics)
        
        # 获取时间戳
        test_timestamps = traffic_data.index[-len(y_test):]
        
        return (baseline_metrics, enhanced_metrics, improvements,
                y_test, test_timestamps, baseline_predictions,
                enhanced_predictions, enhanced_models)
        
    except Exception as e:
        logging.error(f"实验运行出错: {str(e)}")
        raise

def main():
    try:
        # 创建必要的目录
        setup_directories()
        
        visualizer = DataVisualizer()
        
        # 运行实验取结果
        (baseline_metrics, enhanced_metrics, improvements,
         y_test, test_timestamps, baseline_predictions,
         enhanced_predictions, enhanced_models) = run_experiment()
        
        # 加载原始数据用于基础可视化
        traffic_data = load_traffic_data()
        weather_data = load_weather_data()
        
        # 生成交通数据可视化
        visualizer.plot_traffic_patterns(
            traffic_data=traffic_data,
            save_path=os.path.join(RESULTS_DIR, 'figures', 'traffic')
        )
        
        # 生成天气数据可视化
        visualizer.plot_weather_analysis(
            weather_data=weather_data,
            save_path=os.path.join(RESULTS_DIR, 'figures', 'weather')
        )
        
        # 生成更多数据分析可视化
        visualizer.plot_traffic_time_analysis(
            traffic_data=traffic_data,
            save_path=os.path.join(RESULTS_DIR, 'figures', 'traffic')
        )

        visualizer.plot_weather_correlation_analysis(
            weather_data=weather_data,
            save_path=os.path.join(RESULTS_DIR, 'figures', 'weather')
        )

        visualizer.plot_traffic_weather_relationship(
            traffic_data=traffic_data,
            weather_data=weather_data,
            save_path=os.path.join(RESULTS_DIR, 'figures', 'analysis')
        )
        
        # 1. 为每个模型生成对比图
        for model_name in ['LSTM', 'GRU', 'CNN_LSTM']:
            # 基础预测结果可视化
            visualizer.plot_prediction_vs_actual(
                y_true=y_test[:100],
                y_pred=baseline_predictions[model_name][:100],
                timestamps=test_timestamps[:100],
                model_name=f'{model_name}_baseline',
                save_path=os.path.join(RESULTS_DIR, 'figures', 'models')
            )
            
            # 增强预测结果可视化
            visualizer.plot_prediction_vs_actual(
                y_true=y_test[:100],
                y_pred=enhanced_predictions[model_name][:100],
                timestamps=test_timestamps[:100],
                model_name=f'{model_name}_enhanced',
                save_path=os.path.join(RESULTS_DIR, 'figures', 'models')
            )
            
            # 预测误差分布
            visualizer.plot_error_distribution(
                y_true=y_test,
                y_pred=baseline_predictions[model_name],
                model_name=f'{model_name}_baseline',
                save_path=os.path.join(RESULTS_DIR, 'figures', 'models')
            )
            
            visualizer.plot_error_distribution(
                y_true=y_test,
                y_pred=enhanced_predictions[model_name],
                model_name=f'{model_name}_enhanced',
                save_path=os.path.join(RESULTS_DIR, 'figures', 'models')
            )
        
        # 2. 创建所有模型的预测对比
        predictions_dict = {
            'LSTM_baseline': baseline_predictions['LSTM'],
            'LSTM_enhanced': enhanced_predictions['LSTM'],
            'GRU_baseline': baseline_predictions['GRU'],
            'GRU_enhanced': enhanced_predictions['GRU'],
            'CNN_LSTM_baseline': baseline_predictions['CNN_LSTM'],
            'CNN_LSTM_enhanced': enhanced_predictions['CNN_LSTM']
        }
        visualizer.plot_prediction_comparison(
            y_true=y_test,
            predictions_dict=predictions_dict,
            save_path=os.path.join(RESULTS_DIR, 'figures', 'comparison')
        )
        
        # 3. 创建模型改进对比图
        visualizer.plot_model_improvements(
            baseline_metrics=baseline_metrics,
            enhanced_metrics=enhanced_metrics,
            save_path=os.path.join(RESULTS_DIR, 'figures', 'comparison')
        )
        
        # 4. 创建总体性能对比表格
        visualizer.create_performance_table(
            baseline_metrics=baseline_metrics,
            enhanced_metrics=enhanced_metrics,
            improvements=improvements,
            save_path=os.path.join(RESULTS_DIR, 'figures')
        )
        
        # 5. 创建性能提升热力图
        visualizer.create_presentation_summary(
            baseline_metrics=baseline_metrics,
            enhanced_metrics=enhanced_metrics,
            save_path=os.path.join(RESULTS_DIR, 'figures')
        )
        
        # 生成天气影响分析可视化
        visualizer.plot_weather_impact_comparison(
            baseline_metrics=baseline_metrics,
            enhanced_metrics=enhanced_metrics,
            weather_data=weather_data,
            save_path=os.path.join(RESULTS_DIR, 'figures', 'analysis')
        )

        # 获取特征名称和生成���征重要性分析
        input_shape = enhanced_models.models['LSTM'].input_shape[-1]
        feature_names = []

        # 生成基本特征名称
        for i in range(DATA_CONFIG['sequence_length']):
            feature_names.extend([
                f'traffic_t-{i}',
                f'temp_t-{i}',
                f'precip_t-{i}',
                f'wind_t-{i}',
                f'humidity_t-{i}'
            ])

        # 添加额外特征名称
        feature_names.extend([
            'hour_sin',
            'hour_cos',
            'day_sin',
            'day_cos',
            'is_weekend',
            'is_holiday'
        ])

        # 确保特征名称列表长度与输入维度匹配
        if len(feature_names) < input_shape:
            feature_names.extend([f'feature_{i}' for i in range(len(feature_names), input_shape)])
        elif len(feature_names) > input_shape:
            feature_names = feature_names[:input_shape]

        # 生成特征重要性分析
        try:
            importance_df = visualizer.plot_feature_importance_analysis(
                model=enhanced_models.models['LSTM'],
                feature_names=feature_names,
                save_path=os.path.join(RESULTS_DIR, 'figures', 'analysis')
            )
            if importance_df is not None:
                logging.info(f"Top 5 most important features:\n{importance_df.head()}")
        except Exception as e:
            logging.error(f"Error in feature importance analysis: {str(e)}")

        # 创建综合报告
        visualizer.create_comprehensive_report(
            baseline_metrics=baseline_metrics,
            enhanced_metrics=enhanced_metrics,
            weather_data=weather_data,
            save_path=os.path.join(RESULTS_DIR, 'figures')
        )
        
        logging.info("所有模型的对比图和性能分析保存")
        
    except Exception as e:
        logging.error(f"实验过程中出现错误: {str(e)}")
        raise

if __name__ == '__main__':
    main()