import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_augmentation import magnitude_warping, ts_mixup, jittering
from src.plots import plot_all_features_separate_subplots


# Load multivariate time-series data in csv format
def load_csv(file_path):
    # Load CSV using pandas
    df = pd.read_csv(file_path)

    headers = df.columns  # Save column names

    # Assuming the first column is the timestamp and the rest are numeric features
    timestamps = df.iloc[:, 0]  # First column (timestamps)
    features = df.iloc[:, 1:].select_dtypes(include=[np.number]).to_numpy()  # All other numeric columns

    return headers, timestamps, features  # Return column names, timestamps and numeric data


# 3. Save the augmented data
def save_csv(data, output_file):
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)


# Step 3: Save the augmented data with timestamps
def save_csv_with_timestamp(headers, timesteps, augmented_data, output_file):
    # Combine the timestamps with the augmented data
    augmented_df = pd.DataFrame(augmented_data, columns=headers[1:]) #using original columns names
    augmented_df.insert(0, headers[0], timesteps)  # Insert the first column from orig

    # Save the DataFrame to CSV
    augmented_df.to_csv(output_file, index=False)


# Example Usage:
csv_file = 'data/original/DailyDelhiClimate.csv'

# Load data
headers, timestamps, data = load_csv(csv_file)


# A code fragment repeats for each data augmentation method

# Magnitude warping
augmented_data_MW = magnitude_warping(data)
# save
output_file_MW = 'data/augmented/' + os.path.splitext(os.path.basename(csv_file))[0] + '_MW.csv'
save_csv_with_timestamp(headers, timestamps, augmented_data_MW, output_file_MW)
# provide plots for each feature
plot_all_features_separate_subplots(headers, timestamps, data, augmented_data_MW)


# TSMixup
augmented_data_TSMixup = ts_mixup(data)
# save
output_file_TSMixup = 'data/augmented/' + os.path.splitext(os.path.basename(csv_file))[0] + '_TSMixup.csv'
save_csv_with_timestamp(headers, timestamps, augmented_data_TSMixup, output_file_TSMixup)
# provide plots for each feature
plot_all_features_separate_subplots(headers, timestamps, data, augmented_data_TSMixup)


# Jittering
#augmented_data_Jit = jittering(data)
augmented_data_Jit = jittering(data)
# save
output_file_Jit = 'data/augmented/' + os.path.splitext(os.path.basename(csv_file))[0] + '_Jittering.csv'
save_csv_with_timestamp(headers, timestamps, augmented_data_Jit, output_file_Jit)
# provide plots for each feature
plot_all_features_separate_subplots(headers, timestamps, data, augmented_data_Jit)


