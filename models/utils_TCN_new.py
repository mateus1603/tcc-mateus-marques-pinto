"""
Utility functions for TCN model implementation using Darts.

This module provides helper functions for data preprocessing, model evaluation,
and visualization for Temporal Convolutional Networks (TCN) using the Darts package.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.metrics import mape, mae, rmse, mse


def calculate_receptive_field(kernel_size: int, num_layers: int, dilation_base: int = 2) -> int:
    """
    Calculate the receptive field of a TCN model.
    
    The receptive field (RF) determines how far back in time the model can "see".
    For TCN, RF grows exponentially with the number of layers due to dilated convolutions.
    
    Formula: RF = 1 + 2 * (kernel_size - 1) * sum(dilation_base^i for i in range(num_layers))
    
    Args:
        kernel_size: Size of the convolutional kernel
        num_layers: Number of residual blocks in the TCN
        dilation_base: Base for exponential dilation growth (typically 2)
    
    Returns:
        The receptive field size in time steps
    
    Example:
        >>> calculate_receptive_field(kernel_size=3, num_layers=4, dilation_base=2)
        31  # Can see 31 time steps into the past
    """
    dilation_sum = sum(dilation_base ** i for i in range(num_layers))
    rf = 1 + 2 * (kernel_size - 1) * dilation_sum
    return rf


def load_and_prepare_data(
    csv_path: str,
    timestamp_col: str = 'din_instante',
    value_col: str = 'val_intercambiomwmed',
    subsystem_origin: Optional[str] = None,
    subsystem_destination: Optional[str] = None,
    sep: str = ';'
) -> Tuple[TimeSeries, pd.DataFrame]:
    """
    Load energy interchange data and convert to Darts TimeSeries format.
    
    Args:
        csv_path: Path to the CSV file
        timestamp_col: Name of the timestamp column
        value_col: Name of the value column
        subsystem_origin: Filter by origin subsystem (e.g., 'SE', 'N', 'NE', 'S')
        subsystem_destination: Filter by destination subsystem
        sep: CSV separator
    
    Returns:
        Tuple of (TimeSeries object, original DataFrame)
    """
    # Load data
    df = pd.read_csv(csv_path, sep=sep)
    
    # Convert timestamp to datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Filter by subsystems if specified
    if subsystem_origin is not None:
        df = df[df['id_subsistema_origem'] == subsystem_origin]
    if subsystem_destination is not None:
        df = df[df['id_subsistema_destino'] == subsystem_destination]
    
    # Sort by timestamp
    df = df.sort_values(timestamp_col)
    
    # Remove duplicates, keeping the first occurrence
    df = df.drop_duplicates(subset=[timestamp_col], keep='first')
    
    # Create TimeSeries object
    ts = TimeSeries.from_dataframe(
        df,
        time_col=timestamp_col,
        value_cols=value_col,
        freq='H'  # Hourly frequency
    )
    
    return ts, df


def create_temporal_covariates(
    ts: TimeSeries,
    add_hour: bool = True,
    add_day_of_week: bool = True,
    add_month: bool = True,
    add_year: bool = False
) -> TimeSeries:
    """
    Create temporal covariates from datetime attributes.
    
    These are deterministic features that are known for all future time steps,
    making them suitable for forecasting with past_covariates.
    
    Args:
        ts: Input TimeSeries object
        add_hour: Add hour of day (0-23) as one-hot encoded features
        add_day_of_week: Add day of week (0-6) as one-hot encoded features
        add_month: Add month (1-12) as one-hot encoded features
        add_year: Add year as a cyclic feature
    
    Returns:
        TimeSeries object containing all selected temporal features
    """
    covariates_list = []
    
    if add_hour:
        hour_cov = datetime_attribute_timeseries(
            ts, attribute='hour', one_hot=True
        )
        covariates_list.append(hour_cov)
    
    if add_day_of_week:
        dow_cov = datetime_attribute_timeseries(
            ts, attribute='dayofweek', one_hot=True
        )
        covariates_list.append(dow_cov)
    
    if add_month:
        month_cov = datetime_attribute_timeseries(
            ts, attribute='month', one_hot=True
        )
        covariates_list.append(month_cov)
    
    if add_year:
        year_cov = datetime_attribute_timeseries(
            ts, attribute='year', one_hot=False, cyclic=True
        )
        covariates_list.append(year_cov)
    
    # Stack all covariates horizontally
    if len(covariates_list) > 1:
        covariates = covariates_list[0].stack(covariates_list[1])
        for cov in covariates_list[2:]:
            covariates = covariates.stack(cov)
    else:
        covariates = covariates_list[0]
    
    return covariates


def prepare_scalers(
    train: TimeSeries,
    val: TimeSeries,
    test: TimeSeries,
    covariates: Optional[TimeSeries] = None
) -> Tuple[Scaler, Optional[Scaler], TimeSeries, TimeSeries, TimeSeries, Optional[TimeSeries]]:
    """
    Create and fit scalers for time series data, preventing data leakage.
    
    Important: Scalers are fit only on training data and then applied to validation/test.
    This prevents information from validation/test sets from leaking into the model.
    
    Args:
        train: Training TimeSeries
        val: Validation TimeSeries
        test: Test TimeSeries
        covariates: Optional covariates TimeSeries
    
    Returns:
        Tuple of (target_scaler, covariate_scaler, train_scaled, val_scaled, 
                  test_scaled, covariates_scaled)
    """
    # Scale target series
    target_scaler = Scaler()
    train_scaled = target_scaler.fit_transform(train)
    val_scaled = target_scaler.transform(val)
    test_scaled = target_scaler.transform(test)
    
    # Scale covariates if provided (excluding one-hot encoded features)
    covariate_scaler = None
    covariates_scaled = None
    if covariates is not None:
        # For one-hot encoded covariates, we don't need scaling
        # But if we have continuous covariates, we should scale them
        # Since datetime attributes are one-hot or cyclic, we skip scaling
        covariates_scaled = covariates
    
    return target_scaler, covariate_scaler, train_scaled, val_scaled, test_scaled, covariates_scaled


def evaluate_model(
    model,
    series: TimeSeries,
    scaler: Scaler,
    n: int,
    past_covariates: Optional[TimeSeries] = None,
    stride: int = 1,
    metric_names: List[str] = None
) -> dict:
    """
    Evaluate model performance on a series using multiple metrics.
    
    Args:
        model: Trained Darts model
        series: TimeSeries to evaluate on (should be scaled)
        scaler: Scaler used for inverse transformation
        n: Forecast horizon
        past_covariates: Optional past covariates
        stride: Stride for generating forecasts
        metric_names: List of metrics to compute ['mae', 'rmse', 'mape', 'mse']
    
    Returns:
        Dictionary of metric names and values
    """
    if metric_names is None:
        metric_names = ['mae', 'rmse', 'mape']
    
    # Generate predictions
    predictions = model.historical_forecasts(
        series=series,
        past_covariates=past_covariates,
        forecast_horizon=n,
        stride=stride,
        retrain=False,
        verbose=False
    )
    
    # Inverse transform to original scale
    predictions_original = scaler.inverse_transform(predictions)
    series_original = scaler.inverse_transform(series)
    
    # Calculate metrics
    metrics = {}
    metric_funcs = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'mse': mse
    }
    
    for metric_name in metric_names:
        if metric_name in metric_funcs:
            metric_value = metric_funcs[metric_name](series_original, predictions_original)
            metrics[metric_name] = metric_value
    
    return metrics


def plot_predictions(
    actual: TimeSeries,
    predicted: TimeSeries,
    title: str = "TCN Predictions vs Actual",
    num_points: int = 168,
    figsize: Tuple[int, int] = (15, 6)
):
    """
    Plot actual vs predicted values for visual comparison.
    
    Args:
        actual: Actual TimeSeries values (in original scale)
        predicted: Predicted TimeSeries values (in original scale)
        title: Plot title
        num_points: Number of points to display (default 168 = 1 week for hourly data)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Limit to num_points for readability
    actual_subset = actual[-num_points:]
    predicted_subset = predicted[-num_points:]
    
    # Plot
    actual_subset.plot(label='Actual', linewidth=2)
    predicted_subset.plot(label='Predicted', linewidth=2, linestyle='--')
    
    plt.title(title, fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Energy Interchange (MWmed)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_model_summary(
    model,
    input_chunk_length: int,
    output_chunk_length: int,
    kernel_size: int,
    num_layers: int,
    num_filters: int,
    dilation_base: int = 2
):
    """
    Print a summary of the TCN model configuration.
    
    Args:
        model: TCN model instance
        input_chunk_length: Input sequence length
        output_chunk_length: Output sequence length
        kernel_size: Kernel size
        num_layers: Number of layers
        num_filters: Number of filters
        dilation_base: Dilation base
    """
    rf = calculate_receptive_field(kernel_size, num_layers, dilation_base)
    
    print("=" * 60)
    print("TCN Model Configuration")
    print("=" * 60)
    print(f"Input Chunk Length:      {input_chunk_length}")
    print(f"Output Chunk Length:     {output_chunk_length}")
    print(f"Kernel Size:             {kernel_size}")
    print(f"Number of Layers:        {num_layers}")
    print(f"Number of Filters:       {num_filters}")
    print(f"Dilation Base:           {dilation_base}")
    print(f"Receptive Field:         {rf}")
    print(f"")
    if input_chunk_length < rf:
        print(f"⚠️  WARNING: input_chunk_length ({input_chunk_length}) < RF ({rf})")
        print(f"   The model cannot utilize its full receptive field!")
    else:
        print(f"✓  input_chunk_length >= RF: Model can use full history")
    print("=" * 60)
