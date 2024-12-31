# Traffic Flow Prediction Enhanced with Weather Data

This project implements a traffic flow prediction system that incorporates weather data to improve prediction accuracy. The project utilizes deep learning models (LSTM, GRU, CNN-LSTM) for traffic flow prediction and validates the performance improvement through comparative experiments with weather data integration.

## Features

- Multi-Model Support: Implements LSTM, GRU, and CNN-LSTM deep learning models with performance comparison capabilities
- Weather Data Integration: Incorporates multiple weather features including temperature, precipitation, wind speed, etc., to enhance prediction accuracy
- Automated Experiments: Supports automated comparative experiments between baseline and enhanced models, ensuring experimental reproducibility
- Visualization Analysis: Provides rich data visualization and model performance analysis tools for intuitive result presentation
- Experiment Logging: Automatically records experimental processes and results, supporting reproducible research
- GPU Acceleration: Supports GPU-accelerated training, optimized for RTX 4090 GPU, with mixed-precision training

## System Requirements

- Python 3.8+
- CUDA 11.0+ (for GPU usage)
- 32GB+ RAM recommended
- NVIDIA RTX GPU (recommended)
- Ubuntu 22.04 or other Linux distributions (recommended)
- Anaconda or Miniconda (for environment management)

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/liguolin7/tfpw.git
cd tfpw
```

2. Install dependencies (two methods):

### Method 1: Using conda environment file (recommended)

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate tp
```

### Method 2: Manual environment setup

```bash
# Create environment
conda create -n tp python=3.8
conda activate tp

# Install dependencies
pip install -r requirements.txt
```

3. Run experiments:
```bash
# Run complete experiment
python -m src.main

# Or run with custom configuration
python -m src.main --config configs/custom_config.py
```

> Note:
> 1. The METR-LA traffic dataset and NOAA weather dataset are included in the `data/raw/` directory, no additional download required.
> 2. Before first run, check GPU availability using the `nvidia-smi` command.
> 3. Full dataset training may take considerable time; testing with a smaller dataset is recommended for initial setup.

## Project Structure

```
tfpw/
├── data/
│   ├── raw/                # Raw data
│   │   ├── traffic/       # Traffic data (METR-LA dataset)
│   │   └── weather/       # Weather data (NOAA dataset)
│   └── processed/         # Processed data
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration file (model parameters, data processing parameters, etc.)
│   ├── data_loader.py     # Data loading module
│   ├── data_processor.py  # Data processing module (preprocessing, feature engineering)
│   ├── models.py          # Model definitions (LSTM, GRU, CNN-LSTM)
│   ├── evaluation.py      # Evaluation module (performance metrics calculation)
│   ├── visualization.py   # Visualization module
│   └── main.py           # Main program (experiment workflow control)
├── results/
│   └── figures/          # Visualization results
│       ├── analysis/     # Data analysis figures
│       │   ├── traffic_combined_conditions.png    # Traffic flow under combined weather conditions
│       │   ├── traffic_precip_distribution.png    # Precipitation impact on traffic
│       │   ├── traffic_temp_distribution.png      # Temperature impact on traffic
│       │   └── traffic_extreme_events.png         # Extreme weather event analysis
│       ├── comparison/   # Model comparison results
│       │   ├── performance_metrics.png            # Performance metrics comparison
│       │   ├── performance_table.png              # Performance data table
│       │   └── weather_impact_conditions_*.png    # Model performance under different weather conditions
│       ├── models/       # Model prediction results
│       │   ├── *_baseline_prediction.png          # Baseline model predictions
│       │   ├── *_enhanced_prediction.png          # Enhanced model predictions
│       │   ├── *_baseline_error_distribution.png  # Baseline model error distribution
│       │   └── *_enhanced_error_distribution.png  # Enhanced model error distribution
│       ├── traffic/      # Traffic data analysis
│       │   ├── traffic_patterns.png               # Traffic flow patterns
│       │   ├── traffic_heatmap.png                # Traffic flow heatmap
│       │   ├── traffic_time_decomposition.png     # Time series decomposition
│       │   └── traffic_correlation.png            # Correlation analysis
│       ├── weather/      # Weather data analysis
│       │   ├── weather_patterns.png               # Weather patterns analysis
│       │   ├── weather_correlation.png            # Weather feature correlation
│       │   ├── weather_distributions.png          # Weather feature distributions
│       │   └── wind_analysis.png                  # Wind direction and speed analysis
│       └── training/     # Training process visualization
│           ├── *_loss.png                         # Loss function curves
│           ├── *_mae.png                          # MAE curves
│           ├── *_mape.png                         # MAPE curves
│           ├── *_mse.png                          # MSE curves
│           ├── *_r2.png                          # R² curves
│           └── *_lr.png                          # Learning rate curves
├── logs/                 # Experiment logs
│   ├── experiment_*.log  # Detailed experiment logs
│   └── summary_*.txt     # Experiment result summaries
├── requirements.txt      # Package dependencies
└── README.md            # Project documentation
```

> Note: `*` represents model name (LSTM, GRU, CNN-LSTM) or timestamp

## Script Descriptions

### Core Modules

- `src/__init__.py`: Package initialization file
  - Exports public interfaces of all submodules
  - Provides project version information and basic description

- `src/config.py`: Configuration management module
  - Defines all configurable parameters (data paths, model parameters, training parameters, etc.)
  - Provides parameter reading and updating interfaces
  - Contains random seed setting functionality for experiment reproducibility

- `src/data_loader.py`: Data loading module
  - Handles loading of raw traffic and weather data
  - Manages data loading exceptions
  - Provides basic data information logging

- `src/data_processor.py`: Data processing module
  - Implements data cleaning and preprocessing
  - Executes feature engineering and data transformation
  - Generates training, validation, and test sets

- `src/models.py`: Model definition module
  - Implements baseline models (LSTM, GRU, CNN-LSTM)
  - Implements enhanced models (with weather features)
  - Contains core training and evaluation logic
  - Provides GPU acceleration and mixed-precision training support

- `src/evaluation.py`: Evaluation module
  - Implements multiple evaluation metrics (RMSE, MAE, MAPE, R²)
  - Provides model performance analysis functionality
  - Generates evaluation reports and performance comparisons

- `src/visualization.py`: Visualization module
  - Generates data analysis charts
  - Creates model performance comparison visualizations
  - Plots prediction results and error analysis
  - Provides weather impact analysis charts

- `src/main.py`: Main program module
  - Coordinates workflow between modules
  - Controls experiment execution process
  - Manages logging and result saving

### Auxiliary Scripts

- `run.py`: Project launch script
  - Provides command-line interface
  - Ensures correct configuration loading
  - Handles module reloading logic

- `setup.py`: Project installation configuration script
  - Defines project metadata
  - Specifies project dependencies
  - Configures installation options

## Usage

1. Data Information:
   - METR-LA traffic dataset included (in `data/raw/traffic/` directory)
   - NOAA weather dataset included (in `data/raw/weather/` directory)
   - Datasets are preprocessed and aligned, ready for use

2. Configure Experiment Parameters:
   - Set data processing and model training parameters in `src/config.py`
   - Adjust model structure, training parameters, and evaluation metrics as needed

3. Run Experiments:
```bash
# Activate conda environment
conda activate tp

# Run main program
python -m src.main
```

## Core Functionalities

1. Data Processing
   - Traffic data preprocessing and cleaning
   - Weather data feature engineering
   - Data alignment and merging
   - Sequence data preparation

2. Model Training
   - Supports LSTM, GRU, CNN-LSTM models
   - Automated training pipeline
   - Early stopping and learning rate adjustment
   - Model saving and loading

3. Performance Evaluation
   - Multiple metrics evaluation (RMSE, MAE, R², MAPE)
   - Baseline vs. enhanced model comparison
   - Performance analysis under different weather conditions

4. Visualization Analysis
   - Traffic flow pattern analysis
   - Weather impact visualization
   - Prediction result comparison
   - Error distribution analysis

## Experimental Results

With weather feature integration, model performance improved significantly:

- GRU Model:
  - RMSE improvement: 53.78%
  - MAE improvement: 57.79%
  - R² improvement: 98.48%
  - MAPE improvement: 44.04%

- CNN-LSTM Model:
  - RMSE improvement: 41.74%
  - MAE improvement: 40.36%
  - R² improvement: 67.15%
  - MAPE improvement: 13.75%

- LSTM Model:
  - RMSE improvement: 10.48%
  - MAE improvement: 22.69%
  - R² improvement: 20.27%
  - MAPE improvement: 67.22%

## Notes and Common Issues

1. Data-Related:
   - Traffic data must include timestamps and sensor readings
   - Weather data must include basic meteorological elements
   - Consider memory usage with large datasets

2. Hardware-Related:
   - GPU usage recommended for training
   - Ensure CUDA version matches environment.yml
   - Verify GPU availability with nvidia-smi
   - GPU driver updates may be required

3. Performance-Related:
   - For memory issues:
     * Reduce batch size
     * Decrease sequence length
     * Use data generators
   - For training speed:
     * Enable mixed-precision training
     * Adjust worker count
     * Optimize preprocessing pipeline

4. Experiment Tips:
   - Test with small dataset initially
   - Adjust parameters via configuration file
   - Save important experiment configurations

## Performance Optimization Tips

1. Data Processing Optimization:
   - Use memory-mapped files for large datasets
   - Implement data preloading and caching
   - Optimize data augmentation strategies

2. Training Optimization:
   - Use gradient accumulation to reduce memory usage
   - Implement model checkpointing
   - Enable early stopping to prevent overfitting

3. Inference Optimization:
   - Model quantization
   - Batch inference
   - Model pruning

## Citation

If you use this project's code or methods, please cite:

```bibtex
@misc{tfpw2024,
  author = {Liguo Lin},
  title = {Traffic Flow Prediction with Weather},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/liguolin7/tfpw}
}
```

## Maintainer

- Author: Liguo Lin
- Email: liguo.lin@connect.hkust-gz.edu.cn
- GitHub: [@liguolin7](https://github.com/liguolin7)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Thanks to the following open-source projects:
- TensorFlow
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn 