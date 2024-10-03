import os
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

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




