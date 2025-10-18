# TCC - Mateus Marques Pinto - Análise de Dados de Intercâmbio Nacional de Energia Elétrica

## Project Overview

This is an academic project (Trabalho de Conclusão de Curso - TCC) focused on analyzing national electricity interchange data from the Brazilian National Electric System Operator (ONS - Operador Nacional do Sistema Elétrico). The project aims to understand patterns in electricity interchange between different subsystems and develop predictive models for load forecasting.

The project is structured around three main components:
1. Data collection and preparation (`data/` directory)
2. Exploratory data analysis (`eda/` directory) 
3. Machine learning models (`models/` directory)

## Project Architecture

### Data Source
- **Primary Data Source**: ONS (Operador Nacional do Sistema Elétrico) - Intercâmbio Nacional dataset
- **URL**: https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/intercambio_nacional_ho/
- **Years Covered**: 2022-2025 (with focus on 2023-2025 in the main dataset)

### Directory Structure
- `data/` - Contains data collection and preparation notebooks
- `eda/` - Contains exploratory data analysis and visualization utilities
- `models/` - Contains machine learning model implementations (TCN for time series forecasting)
- `.venv/` - Python virtual environment
- `README.md` - Project documentation

## Key Technologies Used

- **Python 3.x** - Programming language
- **Jupyter Notebook** - Development environment
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **PyTorch** - Deep learning framework (for TCN model)
- **Plotly** - Interactive visualizations
- **Matplotlib/Seaborn** - Static visualizations
- **Scikit-learn** - Machine learning utilities

## Data Processing Pipeline

### Data Collection (`data/ONS-data.ipynb`)
- Downloads historical interchange data from ONS S3 bucket
- Supports multiple years (2022, 2023, 2024, 2025)
- Stacks multiple years into a single consolidated CSV file
- Prepares data for downstream analysis

### Exploratory Data Analysis (`eda/EDA.ipynb` and `eda/utils_eda.py`)
- Implements functions for time series analysis
- Functions for daily and weekly pattern analysis
- Time series visualization with gap detection and interpolation
- Data quality analysis tools
- Calendar feature engineering utilities

### Model Implementation (`models/TCN.ipynb`)
- Implements a Temporal Convolutional Network (TCN) for time series forecasting
- Uses PyTorch for deep learning implementation
- Includes residual blocks with dilated convolutions
- Implements causal convolutions to preserve temporal order
- Implements temporal data splitting for training/validation/test sets
- Comprehensive model evaluation and visualization

## Key Features

### EDA Utilities (`eda/utils_eda.py`)
- `plot_daily_pattern()` - Visualizes daily patterns in energy interchange
- `plot_weekly_pattern()` - Visualizes weekly patterns in energy interchange  
- `plot_timeseries_analysis()` - Complete time series analysis with gap detection
- `plot_simple_timeseries()` - Basic time series plotting
- `analyze_data_quality()` - Quality assessment of time series data
- `add_calendar_features()` - Adds temporal features to datasets

### Model Architecture
- TCN with dilated convolutions for long-term dependencies
- Residual connections to improve gradient flow
- Causal convolutions to maintain temporal order
- Adaptive pooling and linear output layer
- Proper initialization using Kaiming normal distribution

## Building and Running

### Prerequisites
- Python 3.8+ 
- Required packages listed in virtual environment (`.venv/`)

### Setup Instructions
1. Clone the repository
2. Navigate to the project directory
3. Activate the virtual environment: `.venv/Scripts/activate` (Windows) or `.venv/bin/activate` (Linux/Mac)
4. Install required packages if not already installed in the virtual environment

### Running the Project
1. Execute `data/ONS-data.ipynb` to download and prepare the data
2. Execute `eda/EDA.ipynb` for exploratory data analysis
3. Execute `models/TCN.ipynb` to train and evaluate the TCN model

## Development Conventions

- Code follows a modular structure with reusable functions
- Visualization functions are centralized in `eda/utils_eda.py`
- Model components are implemented with PyTorch best practices
- Time series data is handled with proper temporal splitting to avoid data leakage
- Calendar features include cyclical encoding for temporal patterns

## Project Status

- ✅ Data collection and preparation
- ✅ Exploratory data analysis
- ✅ TCN model implementation
- 🔄 Model development and refinement (in progress)
- ⏳ Results analysis (pending)
- ⏳ Final documentation (pending)

## Key Findings

The project analyzes electricity interchange between Brazil's main subsystems:
- NORTE (North)
- NORDESTE (Northeast) 
- SUDESTE/CENTRO-OESTE (Southeast/Central-West)
- SUL (South)

The analysis reveals:
- Distinct daily and weekly patterns in energy interchange
- Seasonal variations in power flow between subsystems
- Different load profiles during weekdays vs weekends
- Significant variations during different hours of the day

The TCN model is designed to capture these temporal patterns for accurate forecasting.