# 基于天气数据增强的交通流量预测系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-CUDA%2012.x-brightgreen.svg)](https://developer.nvidia.com/cuda-toolkit)

本项目实现了一个基于深度学习的交通流量预测系统，通过整合天气数据显著提升预测精度。系统支持多种深度学习模型（LSTM、GRU、CNN-LSTM），并提供完整的对比实验框架来验证天气数据集成的效果。

## ✨ 核心特性

- **🚀 多模型支持**：实现LSTM、GRU、CNN-LSTM三种深度学习模型，支持性能对比分析
- **🌤️ 天气数据集成**：融合温度、降水、风速、湿度等多维天气特征，显著提升预测精度
- **⚡ GPU加速训练**：支持NVIDIA RTX GPU混合精度训练，优化训练效率
- **🔬 自动化实验**：提供基线模型与增强模型的自动化对比实验框架
- **📊 丰富可视化**：包含数据分析、模型性能、训练过程等多维度可视化工具
- **📝 实验记录**：自动记录实验过程和结果，支持可重现研究

## 🎯 性能提升

通过天气特征集成，模型性能获得显著提升：

| 模型 | RMSE提升 | MAE提升 | R²提升 | MAPE提升 |
|------|----------|---------|--------|----------|
| **GRU** | **53.78%** | **57.79%** | **98.48%** | **44.04%** |
| **CNN-LSTM** | **41.74%** | **40.36%** | **67.15%** | **13.75%** |
| **LSTM** | **10.48%** | **22.69%** | **20.27%** | **67.22%** |

## 🛠️ 系统要求

### 硬件要求
- **内存**：32GB+ RAM（推荐）
- **GPU**：NVIDIA RTX系列GPU（推荐，支持CUDA 12.x）
- **存储**：至少2GB可用空间

### 软件环境
- **操作系统**：Ubuntu 22.04+ / macOS / Windows 10+
- **Python**：3.8+
- **CUDA**：12.0+（GPU训练）
- **包管理**：Anaconda/Miniconda（推荐）

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/liguolin7/tpfw.git
cd tpfw
```

### 2. 环境配置

#### 方法一：使用conda环境文件（推荐）
```bash
# 创建并激活conda环境
conda env create -f environment.yml
conda activate tp
```

#### 方法二：手动环境配置
```bash
# 创建Python环境
conda create -n tp python=3.8
conda activate tp

# 安装依赖包
pip install -r requirements.txt
```

### 3. 验证GPU环境（可选）
```bash
# 检查GPU可用性
nvidia-smi

# 验证TensorFlow GPU支持
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

### 4. 运行实验
```bash
# 运行完整实验
python -m src.main

# 或使用启动脚本
python run.py
```

> **注意**：
> - 项目已包含METR-LA交通数据集和NOAA天气数据集，无需额外下载
> - 首次运行建议使用小数据集测试环境配置
> - 完整数据集训练可能需要较长时间，请确保GPU可用

## 📁 项目结构

```
tpfw/
├── 📂 data/                    # 数据目录
│   ├── 📂 raw/                # 原始数据
│   │   ├── 📂 traffic/        # 交通数据（METR-LA数据集，69MB）
│   │   └── 📂 weather/        # 天气数据（NOAA数据集，8.5MB）
│   └── 📂 processed/          # 处理后数据
├── 📂 src/                     # 源代码
│   ├── 📄 config.py           # 配置管理（8.1KB，241行）
│   ├── 📄 data_loader.py      # 数据加载（1.4KB，38行）
│   ├── 📄 data_processor.py   # 数据处理（14KB，356行）
│   ├── 📄 models.py           # 模型定义（24KB，569行）
│   ├── 📄 evaluation.py       # 评估模块（23KB，611行）
│   ├── 📄 visualization.py    # 可视化（52KB，1218行）
│   └── 📄 main.py            # 主程序（21KB，538行）
├── 📂 results/                 # 实验结果
│   └── 📂 figures/            # 可视化图表
│       ├── 📂 analysis/       # 数据分析图表
│       ├── 📂 comparison/     # 模型对比结果
│       ├── 📂 models/         # 模型预测结果
│       ├── 📂 traffic/        # 交通数据分析
│       ├── 📂 weather/        # 天气数据分析
│       └── 📂 training/       # 训练过程可视化
├── 📂 logs/                    # 实验日志
├── 📄 requirements.txt         # Python依赖（106个包）
├── 📄 environment.yml          # Conda环境配置
└── 📄 README.md               # 项目文档
```

## 🔧 核心模块详解

### 配置管理 (`config.py`)
- **全局配置中心**：统一管理训练参数、模型配置、数据处理参数
- **自定义损失函数**：组合RMSE、MAE、MAPE、R²的复合损失函数
- **随机种子控制**：确保实验可重现性（RANDOM_SEED=42）
- **GPU优化配置**：混合精度训练、内存增长设置

### 数据处理 (`data_processor.py`)
- **智能数据清洗**：异常值检测、缺失值处理、数据对齐
- **特征工程**：时间特征、滞后特征、移动平均、天气复合特征
- **序列数据准备**：支持可配置的序列长度（默认24）和预测窗口（默认3）
- **数据标准化**：StandardScaler标准化，支持训练/验证/测试集分割

### 模型架构 (`models.py`)
#### 基线模型（BaselineModels）
- **LSTM**：双层LSTM + BatchNormalization + Dropout
- **GRU**：双向GRU + BatchNormalization + Dropout
- **CNN-LSTM**：多尺度CNN + LSTM + 池化层

#### 增强模型（EnhancedModels）
- **特征分离处理**：交通特征（207维）与天气特征分支处理
- **注意力机制**：MultiHeadAttention用于天气特征和时间序列
- **残差连接**：全局平均池化 + 特征融合
- **深度网络结构**：多层Dense + Dropout + 正则化

### 评估系统 (`evaluation.py`)
- **多指标评估**：RMSE、MAE、R²、MAPE四种核心指标
- **性能对比分析**：基线vs增强模型自动对比
- **统计显著性检验**：确保性能提升的统计意义
- **详细评估报告**：生成完整的模型性能分析

### 可视化系统 (`visualization.py`)
- **数据分析可视化**：交通流量模式、天气数据分布、相关性分析
- **模型性能可视化**：训练曲线、预测结果、误差分布
- **对比分析图表**：基线vs增强模型性能对比
- **交互式图表**：支持多种图表格式和自定义样式

## 🔬 实验设计

### 数据集
- **交通数据**：METR-LA数据集，包含207个传感器的5分钟间隔交通流量数据
- **天气数据**：NOAA天气数据，包含温度、降水、风速、湿度等多维特征
- **时间范围**：2012年3月开始的连续时间序列数据
- **数据规模**：交通数据69MB，天气数据8.5MB

### 实验流程
1. **数据预处理**：清洗、特征工程、序列化
2. **基线模型训练**：仅使用交通数据的标准模型
3. **增强模型训练**：集成天气特征的改进模型
4. **性能评估**：多指标对比分析
5. **结果可视化**：生成完整的分析报告

### 训练策略
- **批次大小**：64（可配置）
- **训练轮数**：100轮（带早停机制）
- **学习率调度**：指数衰减 + 自适应调整
- **正则化**：Dropout + L2正则化 + BatchNormalization
- **优化器**：Adam优化器（学习率1e-4到5e-5）

## 📊 实验结果分析

### 性能提升详情
天气数据集成为所有模型带来显著性能提升：

**GRU模型表现最佳**：
- RMSE从原来的基线降低53.78%
- MAE改善57.79%
- R²提升98.48%，接近完美拟合
- MAPE降低44.04%

**CNN-LSTM模型稳定提升**：
- 在所有指标上均有30-70%的改善
- 特别适合捕捉时空模式

**LSTM模型基础改善**：
- 虽然提升相对较小，但在MAPE上有67.22%的显著改善

### 可视化输出
实验生成丰富的可视化结果：
- **数据分析图表**：交通流量模式、天气影响分析
- **模型对比图表**：性能指标对比、预测精度分析
- **训练过程图表**：损失函数曲线、学习率变化
- **预测结果图表**：实际vs预测对比、误差分布分析

## ⚙️ 配置说明

### 主要配置参数
```python
# 数据处理配置
TRAIN_RATIO = 0.7      # 训练集比例
VAL_RATIO = 0.15       # 验证集比例
TEST_RATIO = 0.15      # 测试集比例
SEQUENCE_LENGTH = 24   # 输入序列长度
PREDICTION_HORIZON = 3 # 预测窗口

# 训练配置
BATCH_SIZE = 64        # 批次大小
EPOCHS = 100          # 训练轮数
LEARNING_RATE = 1e-4  # 学习率
RANDOM_SEED = 42      # 随机种子
```

### 模型配置
每个模型都有独立的配置参数，支持：
- 网络层数和神经元数量调整
- Dropout和正则化强度设置
- 优化器和学习率策略配置
- 损失函数权重调整

## 🚨 常见问题与解决方案

### 环境相关
**Q: GPU内存不足怎么办？**
```bash
# 减少批次大小
BATCH_SIZE = 32  # 或更小

# 减少序列长度
SEQUENCE_LENGTH = 12  # 或更小

# 启用梯度累积
# 在config.py中调整相关参数
```

**Q: CUDA版本不匹配？**
```bash
# 检查CUDA版本
nvcc --version

# 重新安装对应版本的TensorFlow
pip install tensorflow==2.12.0
```

### 训练相关
**Q: 训练速度太慢？**
- 确保GPU可用：`nvidia-smi`
- 启用混合精度训练（已默认开启）
- 调整数据加载器的worker数量
- 使用更小的数据集进行测试

**Q: 模型不收敛？**
- 检查学习率设置（可能过大或过小）
- 调整批次大小
- 增加训练轮数
- 检查数据预处理是否正确

### 数据相关
**Q: 内存不足？**
- 使用数据生成器而非一次性加载
- 减少特征维度
- 分批处理数据

## 🔧 性能优化建议

### 数据处理优化
- **内存映射文件**：对于大型数据集使用内存映射
- **数据预加载**：实现数据缓存和预加载机制
- **并行处理**：使用多进程进行数据预处理

### 训练优化
- **梯度累积**：在内存受限时使用梯度累积
- **模型检查点**：定期保存模型状态
- **早停机制**：防止过拟合，节省训练时间

### 推理优化
- **模型量化**：减少模型大小和推理时间
- **批量推理**：提高推理吞吐量
- **模型剪枝**：移除不重要的连接

## 📚 扩展开发

### 添加新模型
1. 在`models.py`中继承`BaselineModels`或`EnhancedModels`
2. 实现`build_model`方法
3. 在`config.py`中添加模型配置
4. 更新`main.py`中的模型列表

### 添加新特征
1. 在`data_processor.py`中扩展`preprocess_weather_features`方法
2. 更新特征选择配置
3. 调整模型输入维度
4. 重新训练和评估

### 自定义可视化
1. 在`visualization.py`中添加新的绘图函数
2. 更新`main.py`中的可视化调用
3. 配置图表样式和保存路径

## 📖 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@misc{tpfw2024,
  author = {Liguo Lin},
  title = {Traffic Flow Prediction Enhanced with Weather Data},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/liguolin7/tpfw},
  note = {基于深度学习的交通流量预测系统，通过天气数据集成实现显著性能提升}
}
```

## 👥 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. **Fork项目**并创建特性分支
2. **编写代码**并添加相应测试
3. **确保代码质量**：运行测试和代码检查
4. **提交Pull Request**并详细描述更改

### 代码规范
- 遵循PEP 8 Python代码规范
- 添加详细的文档字符串
- 编写单元测试
- 保持代码简洁和可读性

## 📄 许可证

本项目采用MIT许可证。详见[LICENSE](LICENSE)文件。

## 🙏 致谢

感谢以下开源项目的支持：
- [TensorFlow](https://tensorflow.org/) - 深度学习框架
- [scikit-learn](https://scikit-learn.org/) - 机器学习工具库
- [pandas](https://pandas.pydata.org/) - 数据处理库
- [matplotlib](https://matplotlib.org/) & [seaborn](https://seaborn.pydata.org/) - 数据可视化
- [METR-LA](http://pems.dot.ca.gov/) - 交通数据集
- [NOAA](https://www.noaa.gov/) - 天气数据集

## 📞 联系方式

- **作者**：Liguo Lin
- **邮箱**：liguo.lin@connect.hkust-gz.edu.cn
- **GitHub**：[@liguolin7](https://github.com/liguolin7)
- **项目主页**：[https://github.com/liguolin7/tpfw](https://github.com/liguolin7/tpfw)

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给我们一个Star！⭐**

[🚀 快速开始](#-快速开始) • [📊 查看结果](#-实验结果分析) • [🔧 配置说明](#️-配置说明) • [❓ 常见问题](#-常见问题与解决方案)

</div> 