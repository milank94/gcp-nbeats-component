from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Create a function to plot time series data
def plot_time_series(
    timesteps: np.array,
    values: np.array,
    format: str = '.',
    start: int = 0,
    end: int = None,
    label: str = None
):
    """Plots timesteps (a series of points in time) against values (a series of values across timesteps).

    Args:
        timesteps (np.array): array of timestep values
        values (np.array): array of values across time
        format (str, optional): style of plot. Defaults to '.'.
        start (int, optional): where to start the plot (setting a value will index from start of timesteps & values).
        Defaults to 0.
        end (int, optional): where to end the plot (similar to start but for the end). Defaults to None.
        label (str, optional): label to show on plot about values. Defaults to None.
    """
    # Plot the series
    plt.plot(timesteps[start:end], values[start:end], format, label=label)
    plt.xlabel('Time')
    plt.ylabel('TSLA Price')
    if label:
        plt.legend(fontsize=14)  # make label bigger
    plt.grid(True)


# Create function to label windowed data
def _get_labelled_windows(
    input_array: np.array,
    horizon: int
) -> Tuple[np.array, np.array]:
    """Create labels for windowed dataset.

    E.g. if horizon == 1
    Input: [0, 1, 2, 3, 4, 5, 6, 7] -> Output: ([0, 1, 2, 3, 4, 5, 6], [7])

    Args:
        input_array (np.array): Input array to generate label for.
        horizon (int): Horizon of labels.

    Returns:
        Tuple[np.array, np.array]: Array of window values, array of labels
    """
    return input_array[:, :-horizon], input_array[:, -horizon:]


# Create function to view NumPy arrays as windows
def make_windows(
    input_array: np.array,
    window_size: int,
    horizon: int
) -> Tuple[np.array, np.array]:
    """Turns a 1D array into a 2D array of sequential labelled windows of window_size with horizon size labels.

    Args:
        input_array (np.array): _description_
        window_size (int): Window size of prediction data.
        horizon (int): Horizon of labels.

    Returns:
        Tuple[np.array, np.array]: Array of window values, array of labels
    """
    # 1. Create a window of specific window_size (add the horizon on the end for labelling later)
    window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)

    # 2. Create a 2D array of multiple window steps (minus 1 to account for 0 indexing)
    window_indexes = window_step + np.expand_dims(np.arange(len(input_array)-(window_size+horizon-1)), axis=0).T

    # 3. Index on the target array (a time series) with 2D array of multiple window steps
    windowed_array = input_array[window_indexes]

    # 4. Get the labelled windows
    windows, labels = _get_labelled_windows(input_array=windowed_array, horizon=horizon)

    return windows, labels


# Make the train/test splits
def make_train_test_splits(
    windows: np.array,
    labels: np.array,
    test_split: float = 0.2
) -> Tuple[np.array, np.array, np.array, np.array]:
    """Splits matching pairs of windows and labels into train and test splits.

    Args:
        windows (np.array): Array of windowed data.
        labels (np.array): Array of data labels.
        test_split (float, optional): Test size as a percentage. Defaults to 0.2.

    Returns:
        Tuple[np.array, np.array, np.array, np.array]: _description_
    """
    split_size = int(len(windows) * (1 - test_split))  # this will default to 80% train / 20% test

    train_windows, train_labels = windows[:split_size], labels[:split_size]
    test_windows, test_labels = windows[split_size:], labels[split_size:]

    return train_windows, test_windows, train_labels, test_labels


# Make predictions with model
def make_preds(
    model: tf.keras.Model,
    input_data: tf.data.Dataset
) -> tf.Tensor:
    """Uses model to make predictions on input_data.

    Args:
        model (tf.keras.Model): Trained TensorFlow model.
        input_data (tf.data.Dataset): Input data to make predictions on.

    Returns:
        tf.Tensor: Model predictions.
    """
    forecast = model.predict(input_data)
    return tf.squeeze(forecast)  # return 1D array of predictions
