import os
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import importlib
import sys

# 确保每次都重新加载配置
if 'src.config' in sys.modules:
    importlib.reload(sys.modules['src.config'])
    # 重新加载相关模块
    if 'src.models' in sys.modules:
        importlib.reload(sys.modules['src.models'])

from . import config

from .data_loader import load_traffic_data, load_weather_data
from .data_processor import DataProcessor
from .models import BaselineModels, EnhancedModels
from .evaluation import evaluate_model, evaluate_models, calculate_improvements
from .visualization import DataVisualizer

# 设置环境变量以禁用所有 TensorFlow 和 CUDA 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # 只使用第一个 GPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用 oneDNN 优化
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE'] = '0'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '0'
os.environ['TF_ENABLE_RESOURCE_VARIABLES'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_DISABLE_CONTROL_FLOW_V2'] = '1'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'
os.environ['TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM'] = '1'

# 禁用所有 Python 警告
import warnings
warnings.filterwarnings('ignore')

# 配置 TensorFlow 日志级别
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# 禁用 TensorFlow 执行器警告
tf.debugging.disable_traceback_filtering()

def setup_logging():
    """配置日志设置"""
    # 创建logs目录
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'experiment_{timestamp}.log')
    
    # 配置日志格式
    log_format = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 配置文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(log_format)
    file_handler.setLevel(logging.INFO)
    
    # 配置控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    console_handler.setLevel(logging.INFO)
    
    # 创建自定义的日志过滤器
    class TensorFlowFilter(logging.Filter):
        def filter(self, record):
            return not any(msg in str(record.getMessage()).lower() for msg in [
                'executing op',
                'gradient',
                'executor',
                'custom operations',
                'numa node',
                'tf-trt',
                'tensorflow',
                'cuda',
                'gpu',
                'warning',
                'warn'
            ])
    
    # 获取根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    root_logger.handlers.clear()
    
    # 添加处理器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # 添加过滤器到两个处理器
    tf_filter = TensorFlowFilter()
    file_handler.addFilter(tf_filter)
    console_handler.addFilter(tf_filter)
    
    # 配置TensorFlow日志
    tf_logger = logging.getLogger('tensorflow')
    tf_logger.addFilter(tf_filter)
    tf_logger.setLevel(logging.ERROR)
    
    # 禁用 NumPy 警告
    np.seterr(all='ignore')
    
    # 禁用 Pandas 警告
    pd.options.mode.chained_assignment = None
    
    # 记录实验开始信息和配置信息
    logging.info("-"*30)
    logging.info("实验开始")
    logging.info("-"*30)
    
    # 记录系统信息
    logging.info("系统信息：")
    logging.info(f"Python版本: {sys.version}")
    logging.info(f"TensorFlow版本: {tf.__version__}")
    logging.info(f"NumPy版本: {np.__version__}")
    logging.info(f"Pandas版本: {pd.__version__}")
    
    # 记录GPU信息
    if tf.config.list_physical_devices('GPU'):
        for gpu in tf.config.list_physical_devices('GPU'):
            logging.info(f"找到GPU设备: {gpu}")
    else:
        logging.info("未找到GPU设备，使用CPU进行训练")
    
    logging.info("-"*50)
    
    return log_file  # 返回日志文件路径，以便后续使用

def setup_directories():
    """创建必要的目录结构"""
    directories = [
        os.path.join(config.RESULTS_DIR, 'figures'),
        os.path.join(config.RESULTS_DIR, 'figures', 'traffic'),
        os.path.join(config.RESULTS_DIR, 'figures', 'weather'),
        os.path.join(config.RESULTS_DIR, 'figures', 'models'),
        os.path.join(config.RESULTS_DIR, 'figures', 'comparison'),
        os.path.join(config.RESULTS_DIR, 'figures', 'analysis'),
        os.path.join(config.RESULTS_DIR, 'figures', 'training')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def run_experiment():
    """运行实验并返回结果"""
    try:
        # 获取最新配置
        training_config = config.get_training_config()
        model_config = config.get_model_config()
        
        # 添加详细的配置验证日志
        logging.info("初始配置验证:")
        logging.info(f"训练配置: epochs={training_config['epochs']}, batch_size={training_config['batch_size']}")
        logging.info(f"回调函数: {[type(cb).__name__ for cb in training_config['callbacks']]}")
        logging.info(f"优化器配置: {model_config}")
        
        data_processor = DataProcessor()
        visualizer = DataVisualizer()
        
        # 加载数据
        logging.info("开始加载数据...")
        traffic_data = load_traffic_data()
        weather_data = load_weather_data()
        logging.info(f"交通数据形状: {traffic_data.shape}, 时间范围: {traffic_data.index[0]} 到 {traffic_data.index[-1]}")
        logging.info(f"天气数据形状: {weather_data.shape}, 时间范围: {weather_data.index[0]} 到 {weather_data.index[-1]}")
        
        # 准备基准模型数据
        logging.info("准备基准模型数据...")
        X_train, y_train, X_val, y_val, X_test, y_test = data_processor.prepare_sequences(
            traffic_data=traffic_data,
            sequence_length=config.DATA_CONFIG['sequence_length']
        )
        logging.info(f"基准模型数据准备完成:")
        logging.info(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
        
        # 准备增强模型数据
        logging.info("准备增强模型数据(包含天气特征)...")
        X_train_enhanced, y_train_enhanced, X_val_enhanced, y_val_enhanced, X_test_enhanced, y_test_enhanced = \
            data_processor.prepare_sequences(
                traffic_data=traffic_data,
                weather_data=weather_data,
                sequence_length=config.DATA_CONFIG['sequence_length']
            )
        logging.info(f"增强模型数据准备完成:")
        logging.info(f"训练集: {X_train_enhanced.shape}, 验证集: {X_val_enhanced.shape}, 测试集: {X_test_enhanced.shape}")
        
        # 训练基准模型
        logging.info("开始训练基准模型...")
        baseline_models = BaselineModels()
        baseline_metrics = {}
        baseline_predictions = {}
        
        for model_name in ['LSTM', 'GRU', 'CNN_LSTM']:
            logging.info(f"\n开始训练基准{model_name}模型...")
            logging.info(f"配置信息: batch_size={training_config['batch_size']}, epochs={training_config['epochs']}")
            
            model, history = baseline_models.train_model(
                model_name, X_train, y_train, X_val, y_val
            )
            
            predictions = model.predict(
                X_test,
                batch_size=int(training_config['batch_size']),
                verbose=training_config['verbose']
            )
            metrics = evaluate_model(
                y_test, 
                predictions.flatten(), 
                model_name,
                model=baseline_models.models[model_name],
                feature_names=None,
                history=history
            )
            
            logging.info(f"{model_name}基准模型评估结果:")
            logging.info(f"实际训练轮数: {len(history.history['loss'])}")
            logging.info(f"RMSE: {metrics['RMSE']:.4f}")
            logging.info(f"MAE: {metrics['MAE']:.4f}")
            logging.info(f"R2: {metrics['R2']:.4f}")
            logging.info(f"MAPE: {metrics['MAPE']:.4f}")
            
            baseline_metrics[model_name] = metrics
            baseline_predictions[model_name] = predictions.flatten()
            
            visualizer.plot_training_history(
                history=history,
                model_name=f"{model_name}_baseline",
                save_path=os.path.join(config.RESULTS_DIR, 'figures', 'training')
            )
        
        # 训练增强模型
        logging.info("\n开始训练增强模型...")
        enhanced_models = EnhancedModels()
        enhanced_metrics = {}
        enhanced_predictions = {}
        
        # 生成特征名称
        feature_names = []
        # 生成基本特征名称
        for i in range(config.DATA_CONFIG['sequence_length']):
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
        
        for model_name in ['LSTM', 'GRU', 'CNN_LSTM']:
            logging.info(f"\n开始训练增强{model_name}模型(包含天气特征)...")
            logging.info(f"配置信息: batch_size={training_config['batch_size']}, epochs={training_config['epochs']}")
            
            model, history = enhanced_models.train_model(
                model_name, X_train_enhanced, y_train_enhanced, 
                X_val_enhanced, y_val_enhanced
            )
            
            predictions = model.predict(
                X_test_enhanced,
                batch_size=int(training_config['batch_size']),
                verbose=training_config['verbose']
            )
            metrics = evaluate_model(
                y_test, 
                predictions.flatten(), 
                model_name,
                model=enhanced_models.models[model_name],
                feature_names=feature_names,
                history=history
            )
            
            logging.info(f"{model_name}增强模型评估结果:")
            logging.info(f"实际训练轮数: {len(history.history['loss'])}")
            logging.info(f"RMSE: {metrics['RMSE']:.4f}")
            logging.info(f"MAE: {metrics['MAE']:.4f}")
            logging.info(f"R2: {metrics['R2']:.4f}")
            logging.info(f"MAPE: {metrics['MAPE']:.4f}")
            
            enhanced_metrics[model_name] = metrics
            enhanced_predictions[model_name] = predictions.flatten()
            
            visualizer.plot_training_history(
                history=history,
                model_name=f"{model_name}_enhanced",
                save_path=os.path.join(config.RESULTS_DIR, 'figures', 'training')
            )
        
        # 计算性能提升
        improvements = calculate_improvements(baseline_metrics, enhanced_metrics)
        logging.info("\n模型性能提升:")
        for model in improvements:
            logging.info(f"\n{model}模型改进:")
            for metric, value in improvements[model].items():
                logging.info(f"{metric}: {value:.2f}%")
        
        # 获取时间戳
        test_timestamps = traffic_data.index[-len(y_test):]
        
        return (baseline_metrics, enhanced_metrics, improvements,
                y_test, test_timestamps, baseline_predictions,
                enhanced_predictions, enhanced_models)
        
    except Exception as e:
        logging.error(f"实验运行过程中出现错误: {str(e)}")
        raise e

def main():
    """主函数"""
    try:
        # 确保随机种子被正确设置
        config.set_global_random_seed()
        
        # 设置日志
        log_file = setup_logging()
        
        # 记录实验配置
        logging.info("实验配置：")
        logging.info(f"随机种子: {config.RANDOM_SEED}")
        logging.info(f"数据集划分比例: 训练集={config.TRAIN_RATIO}, 验证集={config.VAL_RATIO}, 测试集={config.TEST_RATIO}")
        logging.info(f"序列长度: {config.DATA_CONFIG['sequence_length']}")
        logging.info(f"预测步长: {config.DATA_CONFIG['prediction_horizon']}")
        
        # 创建必要的目录
        setup_directories()
        
        # 加载数据
        logging.info("\n开始加载数据...")
        traffic_data = load_traffic_data()
        weather_data = load_weather_data()
        
        # 记录数据信息
        logging.info("\n数据集信息：")
        logging.info(f"交通数据形状: {traffic_data.shape}")
        logging.info(f"交通数据时间范围: {traffic_data.index[0]} 到 {traffic_data.index[-1]}")
        logging.info(f"天气数据形状: {weather_data.shape}")
        logging.info(f"天气数据时间范围: {weather_data.index[0]} 到 {weather_data.index[-1]}")
        
        # 获取最新配置
        training_config = config.get_training_config()
        model_config = config.get_model_config()
        
        # 记录训练配置
        logging.info("\n训练配置：")
        logging.info(f"批次大小: {training_config['batch_size']}")
        logging.info(f"训练轮数: {training_config['epochs']}")
        logging.info(f"优化器配置: {model_config}")
        
        visualizer = DataVisualizer()
        
        # 运行实验
        logging.info("\n开始运行实验...")
        start_time = datetime.now()
        
        (baseline_metrics, enhanced_metrics, improvements,
         y_test, test_timestamps, baseline_predictions,
         enhanced_predictions, enhanced_models) = run_experiment()
        
        # 记录实验时间
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"\n实验完成，总耗时: {duration}")
        
        # 记录实验结果
        logging.info("\n实验结果：")
        for model_name in baseline_metrics.keys():
            logging.info(f"\n{model_name}模型:")
            logging.info("基准模型性能：")
            logging.info(f"RMSE: {baseline_metrics[model_name]['RMSE']:.4f}")
            logging.info(f"MAE: {baseline_metrics[model_name]['MAE']:.4f}")
            logging.info(f"R2: {baseline_metrics[model_name]['R2']:.4f}")
            logging.info(f"MAPE: {baseline_metrics[model_name]['MAPE']:.4f}")
            
            logging.info("\n增强模型性能：")
            logging.info(f"RMSE: {enhanced_metrics[model_name]['RMSE']:.4f}")
            logging.info(f"MAE: {enhanced_metrics[model_name]['MAE']:.4f}")
            logging.info(f"R2: {enhanced_metrics[model_name]['R2']:.4f}")
            logging.info(f"MAPE: {enhanced_metrics[model_name]['MAPE']:.4f}")
            
            logging.info("\n性能提升：")
            for metric, value in improvements[model_name].items():
                logging.info(f"{metric}: {value:.2f}%")
        
        # 生成可视化
        logging.info("\n开始生成可视化结果...")
        
        # 生成交通数据可视化
        visualizer.plot_traffic_patterns(
            traffic_data=traffic_data,
            save_path=visualizer.subdirs['traffic']
        )
        
        # 生成天气数据可视化
        visualizer.plot_weather_analysis(
            weather_data=weather_data,
            save_path=visualizer.subdirs['weather']
        )
        
        # 生成更多数据分析可视化
        visualizer.plot_traffic_time_analysis(
            traffic_data=traffic_data,
            save_path=visualizer.subdirs['traffic']
        )
        
        visualizer.plot_weather_correlation_analysis(
            weather_data=weather_data,
            save_path=visualizer.subdirs['weather']
        )
        
        visualizer.plot_traffic_weather_relationship(
            traffic_data=traffic_data,
            weather_data=weather_data,
            save_path=visualizer.subdirs['analysis']
        )
        
        # 生成模型性能对比可视化
        visualizer.plot_metrics_comparison(
            baseline_metrics=baseline_metrics,
            enhanced_metrics=enhanced_metrics
        )
        
        # 生成天气影响分析可视化
        visualizer.plot_weather_impact_comparison(
            baseline_metrics=baseline_metrics,
            enhanced_metrics=enhanced_metrics,
            weather_data=weather_data,
            save_path=visualizer.subdirs['comparison']
        )
        
        # 创建性能对比表格
        visualizer.create_performance_table(
            baseline_metrics=baseline_metrics,
            enhanced_metrics=enhanced_metrics,
            improvements=improvements,
            save_path=visualizer.subdirs['comparison']
        )
        
        # 为每个模型生成详细的性能分析
        for model_name in ['LSTM', 'GRU', 'CNN_LSTM']:
            # 基准模型预测可视化
            visualizer.plot_prediction_vs_actual(
                y_true=y_test,
                y_pred=baseline_predictions[model_name],
                timestamps=test_timestamps,
                model_name=f'{model_name}_baseline',
                save_path=visualizer.subdirs['models']
            )
            
            # 增强模型预测可视化
            visualizer.plot_prediction_vs_actual(
                y_true=y_test,
                y_pred=enhanced_predictions[model_name],
                timestamps=test_timestamps,
                model_name=f'{model_name}_enhanced',
                save_path=visualizer.subdirs['models']
            )
            
            # 误差分布分析
            visualizer.plot_error_distribution(
                y_true=y_test,
                y_pred=baseline_predictions[model_name],
                model_name=f'{model_name}_baseline',
                save_path=visualizer.subdirs['models']
            )
            
            visualizer.plot_error_distribution(
                y_true=y_test,
                y_pred=enhanced_predictions[model_name],
                model_name=f'{model_name}_enhanced',
                save_path=visualizer.subdirs['models']
            )
        
        logging.info("可视化结果生成完成")
        
        # 保存实验配置和结果摘要
        summary_file = os.path.join(os.path.dirname(log_file), f'summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("实验配置和结果摘要\n")
            f.write("="*50 + "\n\n")
            
            f.write("实验配置：\n")
            f.write(f"随机种子: {config.RANDOM_SEED}\n")
            f.write(f"数据集划分比例: 训练集={config.TRAIN_RATIO}, 验证集={config.VAL_RATIO}, 测试集={config.TEST_RATIO}\n")
            f.write(f"序列长度: {config.DATA_CONFIG['sequence_length']}\n")
            f.write(f"预测步长: {config.DATA_CONFIG['prediction_horizon']}\n\n")
            
            f.write("实验结果：\n")
            for model_name in baseline_metrics.keys():
                f.write(f"\n{model_name}模型性能提升：\n")
                for metric, value in improvements[model_name].items():
                    f.write(f"{metric}: {value:.2f}%\n")
            
            f.write(f"\n实验总耗时: {duration}\n")
        
        logging.info(f"\n实验配置和结果摘要已保存至: {summary_file}")
        logging.info("\n" + "-"*30)
        logging.info("实验全部完成")
        logging.info("-"*30)
        
    except Exception as e:
        logging.error(f"\n实验过程中出现错误: {str(e)}")
        logging.error("详细错误信息:", exc_info=True)
        raise

if __name__ == '__main__':
    main()