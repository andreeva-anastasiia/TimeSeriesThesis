'''
Performance metrics,
#
# distribution,
# autocorrelation (strong low),
# histogram relative balance of characteristics:
# Dataset. sthrengh of season, season length, time series properties??? Biases
# What else

Cross correlation between variables ()


One function per measure

Predictability



List with evaluation approaches framework.

'''


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the datasets
df1 = pd.read_csv('data/original/DailyDelhiClimate.csv', parse_dates=['date'])
df2 = pd.read_csv('data/augmented/DailyDelhiClimate_MW.csv', parse_dates=['date'])

# Display basic statistics
print("Dataset 1 Statistics:\n", df1.describe())
print("Dataset 2 Statistics:\n", df2.describe())

# Visualize distributions
fig, axs = plt.subplots(3, 2, figsize=(12, 10))

# Histograms
for i, column in enumerate(['meantemp', 'humidity', 'wind_speed', 'meanpressure']):
    sns.histplot(df1[column], kde=True, ax=axs[i//2, i%2], color='blue', label='Dataset 1', stat='density')
    sns.histplot(df2[column], kde=True, ax=axs[i//2, i%2], color='orange', label='Dataset 2', stat='density')
    axs[i//2, i%2].set_title(column)
    axs[i//2, i%2].legend()

plt.tight_layout()
plt.show()

# Statistical Tests
ks_results = {}
for column in ['meantemp', 'humidity', 'wind_speed', 'meanpressure']:
    ks_stat, ks_p_value = stats.ks_2samp(df1[column], df2[column])
    ks_results[column] = (ks_stat, ks_p_value)

print("KS Test Results:\n", ks_results)

# Correlation Analysis
correlation_df1 = df1.corr()
correlation_df2 = df2.corr()

# Correlation Heatmaps
plt.figure(figsize=(12, 6))
sns.heatmap(correlation_df1, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap - Dataset 1')
plt.show()

plt.figure(figsize=(12, 6))
sns.heatmap(correlation_df2, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap - Dataset 2')
plt.show()