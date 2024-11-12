import os
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pandas as pd
import matplotlib.pyplot as plt

def plot_all_features_separate_subplots(headers, timestamps, original, augmented):
    num_features = original.shape[1]  # Get the number of features (columns)

    # Create a figure with subplots, one for each feature
    fig, axes = plt.subplots(num_features, 1, figsize=(10, num_features * 2), sharex=True)

    # Convert timestamps to datetime if they are not already
    timestamps = pd.to_datetime(timestamps)

    # If there's only one feature, 'axes' won't be a list, so make it a list for consistency
    if num_features == 1:
        axes = [axes]

    for feature in range(num_features):
        axes[feature].plot(timestamps, original[:, feature], label=f'Original ' + headers[feature + 1])
        axes[feature].plot(timestamps, augmented[:, feature], linestyle='dashed', label=f'Augmented ' + headers[feature + 1])
        axes[feature].set_title(f'' + headers[feature + 1] + ' (Original vs Augmented)')
        axes[feature].legend()

        # Set the date format for the x-axis
        axes[feature].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # Show a part of the x-axis timestamps
        # Custom range
        axes[feature].set_xticks(timestamps[::len(timestamps) // 10])  # Show 10 ticks


    plt.xlabel(headers[0])
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust the layout to prevent overlapping titles/labels

    # Save plot as PNG (change to 'jpeg' if needed)
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    save_path = f'plots/MW_{timestamp}.png'
    plt.savefig(save_path)

    plt.show()




def plot_component(original_path, generated_path):

    # Load the original component
    original_component = pd.read_csv(original_path, index_col=0)

    # Load the generated component
    generated_component = pd.read_csv(generated_path, index_col=0)

    # Create a plot
    plt.figure(figsize=(12, 6))

    # Plot original component
    plt.plot(original_component.index, original_component['trend'],
             label='Original trend', color='blue')

    # Plot generated component
    plt.plot(generated_component.index, generated_component['trend'],
             label='Generated trend', color='orange')

    # Adding titles and labels
    plt.title('Original vs Generated ' + 'trend')
    plt.xlabel('Index')
    plt.ylabel('Trend Value')
    plt.legend()
    plt.grid()

    # Show the plot
    plt.show()


import matplotlib.pyplot as plt
import numpy as np


def plot_seasonalities(original_seasonality_path, generated_seasonality_path):
    # Load original and generated seasonalities from CSV
    original_seasonality = np.loadtxt(original_seasonality_path, delimiter=",", skiprows=1, usecols=1)
    generated_seasonality = np.loadtxt(generated_seasonality_path, delimiter=",", skiprows=1, usecols=1)

    # Repeat the generated seasonality to match the length of the original (1462 days)
    num_repeats = len(original_seasonality) // len(generated_seasonality)
    generated_seasonality_repeated = np.tile(generated_seasonality, num_repeats)

    # Handle the case where the original length (1462) is not exactly divisible by 365
    if len(generated_seasonality_repeated) < len(original_seasonality):
        generated_seasonality_repeated = np.concatenate((generated_seasonality_repeated, generated_seasonality[:len(
            original_seasonality) % len(generated_seasonality)]))

    # Create the time axis (days)
    days = np.arange(len(original_seasonality))

    # Plot both original and generated seasonalities
    plt.figure(figsize=(12, 6))
    plt.plot(days, original_seasonality, label='Original Seasonality', color='blue')
    plt.plot(days, generated_seasonality_repeated, label='Generated Seasonality', color='orange') #, linestyle='dashed'

    # Add labels, legend, and title
    plt.xlabel('Days')
    plt.ylabel('Seasonality')
    plt.title('Original vs Generated Seasonality')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

#
# def plot_trends(original_trend_path, generated_trend_path):



import matplotlib.pyplot as plt


# Function to plot the losses over time
def plot_losses(d_losses, g_losses):
    plt.figure(figsize=(10, 6))

    # Plot discriminator loss
    plt.plot(d_losses, label='Discriminator Loss', color='blue')

    # Plot generator loss
    plt.plot(g_losses, label='Generator Loss', color='orange')

    # Adding labels and title
    plt.title('GAN Losses Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Show the plot
    plt.show()


import matplotlib.pyplot as plt


def plot_time_series(original_series, augmented_series, feature_names):
    """
    Plot the original and augmented time series for each feature.

    Parameters:
    - original_series: list of 2D numpy arrays, each representing an original time series (time_steps, features)
    - augmented_series: 2D numpy array, augmented time series of shape (time_steps, features)
    - feature_names: list of strings, names of features for labeling plots
    """
    num_features = original_series[0].shape[1]  # Number of features

    fig, axes = plt.subplots(num_features, 1, figsize=(10, 6), sharex=True)
    fig.suptitle("Original and Augmented Time Series")

    # If there's only one feature, ensure axes is iterable
    if num_features == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        # Plot each original series for the feature
        for series in original_series:
            ax.plot(series[:, i], alpha=0.5, label=f"Original {feature_names[i]}", color="blue")

        # Plot the augmented series
        ax.plot(augmented_series[:, i], label=f"Augmented {feature_names[i]}", color="red", linewidth=2)
        ax.set_ylabel(feature_names[i])
        ax.legend(loc='upper right')

    plt.xlabel("Time Steps")
    plt.show()


import matplotlib.pyplot as plt
import numpy as np




# PLOT FOR SEPARATE DATASETS FOR TS-MIXUP
import matplotlib.pyplot as plt
import numpy as np

def plot_time_series_datasets(datasets, dataset_names=None, augmented_series=None, augmented_name='Augmented Series'):
    """
    Plot multiple time series datasets with separate subplots for each feature in each dataset,
    including an augmented series if provided.

    Parameters:
    - datasets: list of 2D NumPy arrays, each with shape (time_steps, features)
    - dataset_names: list of names for each dataset, for labeling purposes
    - augmented_series: 2D NumPy array representing the augmented time series
    - augmented_name: name for the augmented series (used in the legend)
    """
    num_datasets = len(datasets)
    num_features = datasets[0].shape[1]

    # Set up the figure and axes for subplots, one for each feature across all datasets
    fig, axes = plt.subplots(num_features, num_datasets + 1, figsize=(15, num_features * 4), sharex=True)

    # If there is only one feature, axes will not be a 2D array, so we need to handle it gracefully
    if num_features == 1:
        axes = np.expand_dims(axes, axis=1)

    # Plot each dataset
    for i, data in enumerate(datasets):
        for feature in range(num_features):
            ax = axes[feature, i]  # Get the specific subplot for each feature
            time_steps = data.shape[0]
            ax.plot(range(time_steps), data[:, feature], label=f'{dataset_names[i] if dataset_names else f"Dataset {i + 1}"} Feature {feature + 1}')
            ax.set_ylabel('Value')
            ax.grid(True)
            if i == 0:  # Add title for each feature in the first column
                ax.set_title(f'Feature {feature + 1}')
            if feature == num_features - 1:  # Add legend only to the last row
                ax.legend()

    # Plot augmented series if provided
    if augmented_series is not None:
        for feature in range(num_features):
            ax = axes[feature, num_datasets]  # The last column for augmented series
            time_steps = augmented_series.shape[0]
            ax.plot(range(time_steps), augmented_series[:, feature], linestyle='dashed',
                    label=f'{augmented_name} Feature {feature + 1}')
            ax.set_ylabel('Value')
            ax.grid(True)
            if feature == 0:  # Add title for augmented series in the first row
                ax.set_title(f'Augmented {augmented_name}')
            if feature == num_features - 1:  # Add legend only to the last row
                ax.legend()

    plt.xlabel('Time Steps')
    plt.tight_layout()
    plt.show()