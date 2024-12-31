import os
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import importlib
import sys

# Ensure to reload the configuration every time
if 'src.config' in sys.modules:
    importlib.reload(sys.modules['src.config'])
    # Reload related modules
    if 'src.models' in sys.modules:
        importlib.reload(sys.modules['src.models'])

from . import config

from .data_loader import load_traffic_data, load_weather_data
from .data_processor import DataProcessor
from .models import BaselineModels, EnhancedModels
from .evaluation import evaluate_model, evaluate_models, calculate_improvements
from .visualization import DataVisualizer

# Set environment variables to disable all TensorFlow and CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # Use only the first GPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization
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

# Disable all Python warnings
import warnings
warnings.filterwarnings('ignore')

# Configure TensorFlow log level
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# Disable TensorFlow executor warnings
tf.debugging.disable_traceback_filtering()

def setup_logging():
    """Configure logging settings"""
    # Create logs directory
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log file name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'experiment_{timestamp}.log')
    
    # Configure log format
    log_format = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(log_format)
    file_handler.setLevel(logging.INFO)
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    console_handler.setLevel(logging.INFO)
    
    # Create custom log filter
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
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Add filters to both handlers
    tf_filter = TensorFlowFilter()
    file_handler.addFilter(tf_filter)
    console_handler.addFilter(tf_filter)
    
    # Configure TensorFlow log
    tf_logger = logging.getLogger('tensorflow')
    tf_logger.addFilter(tf_filter)
    tf_logger.setLevel(logging.ERROR)
    
    # Disable NumPy warnings
    np.seterr(all='ignore')
    
    # Disable Pandas warnings
    pd.options.mode.chained_assignment = None
    
    # Log experiment start information and configuration
    logging.info("-"*30)
    logging.info("Experiment start")
    logging.info("-"*30)
    
    # Log system information
    logging.info("System information:")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"TensorFlow version: {tf.__version__}")
    logging.info(f"NumPy version: {np.__version__}")
    logging.info(f"Pandas version: {pd.__version__}")
    
    # Log GPU information
    if tf.config.list_physical_devices('GPU'):
        for gpu in tf.config.list_physical_devices('GPU'):
            logging.info(f"Found GPU device: {gpu}")
    else:
        logging.info("No GPU device found, using CPU for training")
    
    logging.info("-"*50)
    
    return log_file  # Return log file path for later use

def setup_directories():
    """Create necessary directory structure"""
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
    """Run the experiment and return the results"""
    try:
        # Get the latest configuration
        training_config = config.get_training_config()
        model_config = config.get_model_config()
        
        # Add detailed configuration validation log
        logging.info("Initial configuration validation:")
        logging.info(f"Training configuration: epochs={training_config['epochs']}, batch_size={training_config['batch_size']}")
        logging.info(f"Callbacks: {[type(cb).__name__ for cb in training_config['callbacks']]}")
        logging.info(f"Optimizer configuration: {model_config}")
        
        data_processor = DataProcessor()
        visualizer = DataVisualizer()
        
        # Load data
        logging.info("Start loading data...")
        traffic_data = load_traffic_data()
        weather_data = load_weather_data()
        logging.info(f"Traffic data shape: {traffic_data.shape}, time range: {traffic_data.index[0]} to {traffic_data.index[-1]}")
        logging.info(f"Weather data shape: {weather_data.shape}, time range: {weather_data.index[0]} to {weather_data.index[-1]}")
        
        # Prepare baseline model data
        logging.info("Prepare baseline model data...")
        X_train, y_train, X_val, y_val, X_test, y_test = data_processor.prepare_sequences(
            traffic_data=traffic_data,
            sequence_length=config.DATA_CONFIG['sequence_length']
        )
        logging.info(f"Baseline model data prepared:")
        logging.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
        
        # Prepare enhanced model data
        logging.info("Prepare enhanced model data (including weather features)...")
        X_train_enhanced, y_train_enhanced, X_val_enhanced, y_val_enhanced, X_test_enhanced, y_test_enhanced = \
            data_processor.prepare_sequences(
                traffic_data=traffic_data,
                weather_data=weather_data,
                sequence_length=config.DATA_CONFIG['sequence_length']
            )
        logging.info(f"Enhanced model data prepared:")
        logging.info(f"Training set: {X_train_enhanced.shape}, Validation set: {X_val_enhanced.shape}, Test set: {X_test_enhanced.shape}")
        
        # Train baseline model
        logging.info("Start training baseline model...")
        baseline_models = BaselineModels()
        baseline_metrics = {}
        baseline_predictions = {}
        
        for model_name in ['LSTM', 'GRU', 'CNN_LSTM']:
            logging.info(f"\nStart training baseline {model_name} model...")
            logging.info(f"Configuration: batch_size={training_config['batch_size']}, epochs={training_config['epochs']}")
            
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
            
            logging.info(f"{model_name} baseline model evaluation result:")
            logging.info(f"Actual training rounds: {len(history.history['loss'])}")
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
        
        # Train enhanced model
        logging.info("\nStart training enhanced model...")
        enhanced_models = EnhancedModels()
        enhanced_metrics = {}
        enhanced_predictions = {}
        
        # Generate feature names
        feature_names = []
        # Generate basic feature names
        for i in range(config.DATA_CONFIG['sequence_length']):
            feature_names.extend([
                f'traffic_t-{i}',
                f'temp_t-{i}',
                f'precip_t-{i}',
                f'wind_t-{i}',
                f'humidity_t-{i}'
            ])
        
        # Add extra feature names
        feature_names.extend([
            'hour_sin',
            'hour_cos',
            'day_sin',
            'day_cos',
            'is_weekend',
            'is_holiday'
        ])
        
        # Ensure feature names list length matches input dimension
        input_shape = X_train_enhanced.shape[-1]
        if len(feature_names) < input_shape:
            feature_names.extend([f'feature_{i}' for i in range(len(feature_names), input_shape)])
        elif len(feature_names) > input_shape:
            feature_names = feature_names[:input_shape]
        
        for model_name in ['LSTM', 'GRU', 'CNN_LSTM']:
            logging.info(f"\nStart training enhanced {model_name} model (including weather features)...")
            logging.info(f"Configuration: batch_size={training_config['batch_size']}, epochs={training_config['epochs']}")
            
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
            
            logging.info(f"{model_name} enhanced model evaluation result:")
            logging.info(f"Actual training rounds: {len(history.history['loss'])}")
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
        
        # Calculate performance improvement
        improvements = calculate_improvements(baseline_metrics, enhanced_metrics)
        logging.info("\nModel performance improvement:")
        for model in improvements:
            logging.info(f"\n{model} model improvement:")
            for metric, value in improvements[model].items():
                logging.info(f"{metric}: {value:.2f}%")
        
        # Get timestamp
        test_timestamps = traffic_data.index[-len(y_test):]
        
        return (baseline_metrics, enhanced_metrics, improvements,
                y_test, test_timestamps, baseline_predictions,
                enhanced_predictions, enhanced_models)
        
    except Exception as e:
        logging.error(f"Error occurred during experiment: {str(e)}")
        raise e

def main():
    """Main function"""
    try:
        # Ensure random seed is correctly set
        config.set_global_random_seed()
        
        # Set up logging
        log_file = setup_logging()
        
        # Log experiment configuration
        logging.info("Experiment configuration:")
        logging.info(f"Random seed: {config.RANDOM_SEED}")
        logging.info(f"Data set split ratio: Training set={config.TRAIN_RATIO}, Validation set={config.VAL_RATIO}, Test set={config.TEST_RATIO}")
        logging.info(f"Sequence length: {config.DATA_CONFIG['sequence_length']}")
        logging.info(f"Prediction horizon: {config.DATA_CONFIG['prediction_horizon']}")
        
        # Create necessary directories
        setup_directories()
        
        # Load data
        logging.info("\nStart loading data...")
        traffic_data = load_traffic_data()
        weather_data = load_weather_data()
        
        # Log data information
        logging.info("\nData set information:")
        logging.info(f"Traffic data shape: {traffic_data.shape}")
        logging.info(f"Traffic data time range: {traffic_data.index[0]} to {traffic_data.index[-1]}")
        logging.info(f"Weather data shape: {weather_data.shape}")
        logging.info(f"Weather data time range: {weather_data.index[0]} to {weather_data.index[-1]}")
        
        # Get the latest configuration
        training_config = config.get_training_config()
        model_config = config.get_model_config()
        
        # Log training configuration
        logging.info("\nTraining configuration:")
        logging.info(f"Batch size: {training_config['batch_size']}")
        logging.info(f"Training rounds: {training_config['epochs']}")
        logging.info(f"Optimizer configuration: {model_config}")
        
        visualizer = DataVisualizer()
        
        # Run the experiment
        logging.info("\nStart running the experiment...")
        start_time = datetime.now()
        
        (baseline_metrics, enhanced_metrics, improvements,
         y_test, test_timestamps, baseline_predictions,
         enhanced_predictions, enhanced_models) = run_experiment()
        
        # Log experiment duration
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"\nExperiment completed, total duration: {duration}")
        
        # Log experiment results
        logging.info("\nExperiment results:")
        for model_name in baseline_metrics.keys():
            logging.info(f"\n{model_name} model:")
            logging.info("Baseline model performance:")
            logging.info(f"RMSE: {baseline_metrics[model_name]['RMSE']:.4f}")
            logging.info(f"MAE: {baseline_metrics[model_name]['MAE']:.4f}")
            logging.info(f"R2: {baseline_metrics[model_name]['R2']:.4f}")
            logging.info(f"MAPE: {baseline_metrics[model_name]['MAPE']:.4f}")
            
            logging.info("\nEnhanced model performance:")
            logging.info(f"RMSE: {enhanced_metrics[model_name]['RMSE']:.4f}")
            logging.info(f"MAE: {enhanced_metrics[model_name]['MAE']:.4f}")
            logging.info(f"R2: {enhanced_metrics[model_name]['R2']:.4f}")
            logging.info(f"MAPE: {enhanced_metrics[model_name]['MAPE']:.4f}")
            
            logging.info("\nPerformance improvement:")
            for metric, value in improvements[model_name].items():
                logging.info(f"{metric}: {value:.2f}%")
        
        # Generate visualizations
        logging.info("\nStart generating visualization results...")
        
        # Generate traffic data visualization
        visualizer.plot_traffic_patterns(
            traffic_data=traffic_data,
            save_path=visualizer.subdirs['traffic']
        )
        
        # Generate weather data visualization
        visualizer.plot_weather_analysis(
            weather_data=weather_data,
            save_path=visualizer.subdirs['weather']
        )
        
        # Generate more data analysis visualizations
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
        
        # Generate model performance comparison visualization
        visualizer.plot_metrics_comparison(
            baseline_metrics=baseline_metrics,
            enhanced_metrics=enhanced_metrics
        )
        
        # Generate weather impact analysis visualization
        visualizer.plot_weather_impact_comparison(
            baseline_metrics=baseline_metrics,
            enhanced_metrics=enhanced_metrics,
            weather_data=weather_data,
            save_path=visualizer.subdirs['comparison']
        )
        
        # Create performance comparison table
        visualizer.create_performance_table(
            baseline_metrics=baseline_metrics,
            enhanced_metrics=enhanced_metrics,
            improvements=improvements,
            save_path=visualizer.subdirs['comparison']
        )
        
        # Generate detailed performance analysis for each model
        for model_name in ['LSTM', 'GRU', 'CNN_LSTM']:
            # Baseline model prediction visualization
            visualizer.plot_prediction_vs_actual(
                y_true=y_test,
                y_pred=baseline_predictions[model_name],
                timestamps=test_timestamps,
                model_name=f'{model_name}_baseline',
                save_path=visualizer.subdirs['models']
            )
            
            # Enhanced model prediction visualization
            visualizer.plot_prediction_vs_actual(
                y_true=y_test,
                y_pred=enhanced_predictions[model_name],
                timestamps=test_timestamps,
                model_name=f'{model_name}_enhanced',
                save_path=visualizer.subdirs['models']
            )
            
            # Error distribution analysis
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
        
        logging.info("Visualization results generated successfully")
        
        # Save experiment configuration and summary
        summary_file = os.path.join(os.path.dirname(log_file), f'summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Experiment Configuration and Summary\n")
            f.write("="*50 + "\n\n")
            
            f.write("Experiment Configuration：\n")
            f.write(f"Random Seed: {config.RANDOM_SEED}\n")
            f.write(f"Dataset Split Ratio: Training Set={config.TRAIN_RATIO}, Validation Set={config.VAL_RATIO}, Test Set={config.TEST_RATIO}\n")
            f.write(f"Sequence Length: {config.DATA_CONFIG['sequence_length']}\n")
            f.write(f"Prediction Horizon: {config.DATA_CONFIG['prediction_horizon']}\n\n")
            
            f.write("Experiment Results：\n")
            for model_name in baseline_metrics.keys():
                f.write(f"\n{model_name} Model Performance Improvement：\n")
                for metric, value in improvements[model_name].items():
                    f.write(f"{metric}: {value:.2f}%\n")
            
            f.write(f"\nTotal Experiment Duration: {duration}\n")
        
        logging.info(f"\nExperiment Configuration and Summary saved to: {summary_file}")
        logging.info("\n" + "-"*30)
        logging.info("Experiment Completed")
        logging.info("-"*30)
        
    except Exception as e:
        logging.error(f"\nError occurred during the experiment: {str(e)}")
        logging.error("Detailed error information:", exc_info=True)
        raise

if __name__ == '__main__':
    main()