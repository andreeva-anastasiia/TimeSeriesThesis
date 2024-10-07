from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose



# Rewrite to independent
# Performance metrics def

# One function per measure
def basic_statistics(data):

    # Load the dataset
    df = pd.read_csv(data, parse_dates=['date'])

    print("Dataset 1 Statistics:\n", df.describe())


# Histograms #relative balance of characteristics
def histograms(original, augmented):

    # Load the datasets
    df1 = pd.read_csv(original, parse_dates=['date'])
    df2 = pd.read_csv(augmented, parse_dates=['date'])

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    for i, column in enumerate(['meantemp', 'humidity', 'wind_speed', 'meanpressure']):
        sns.histplot(df1[column], kde=True, ax=axs[i//2, i%2], color='blue', label='original', stat='density')
        sns.histplot(df2[column], kde=True, ax=axs[i//2, i%2], color='orange', label='augmented', stat='density')
        axs[i//2, i%2].set_title(column)
        axs[i//2, i%2].legend()

    plt.tight_layout()
    #plt.show()
    save_eval_plot('histograms')

'''
A Kernel Density Estimate plot is applied 
to estimate the probability density function of a continuous random variable. 

KDE places a kernel (a smooth, bell-shaped function) at each data point 
and summing these functions to get an estimate of the overall data distribution.
'''

# KDE Plots #relative balance of characteristics
def KDE_plots(original, augmented):

    # Load the datasets
    df1 = pd.read_csv(original, parse_dates=['date'])
    df2 = pd.read_csv(augmented, parse_dates=['date'])

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    #
    for i, column in enumerate(['meantemp', 'humidity', 'wind_speed', 'meanpressure']):
        sns.kdeplot(df1[column], ax=axs[i // 2, i % 2], color='blue', label='original',
                    bw_adjust=0.5, fill=True, alpha=0.3)  # adjust bw_adjust for smoothness
        sns.kdeplot(df2[column], ax=axs[i // 2, i % 2], color='orange', label='augmented',
                    bw_adjust=0.5, fill=True, alpha=0.3)

        axs[i // 2, i % 2].set_title(column)
        axs[i // 2, i % 2].legend()

    plt.tight_layout()
    # plt.show()
    save_eval_plot('KDE_plots')


# Distributions of characteristics # Pairplot for all variables
def distributions(data):

    # Load the dataset
    df1 = pd.read_csv(data, parse_dates=['date'])

    sns.pairplot(df1[['meantemp', 'humidity', 'wind_speed', 'meanpressure']])
    plt.suptitle('Pair Plot for All Variables', y=1.02)
    # plt.show()
    save_eval_plot('distributions')


# Distributions of characteristics for 2 datasets # Pairplot for all variables
def distributions2(original, augmented):
    # Load the datasets
    df1 = pd.read_csv(original, parse_dates=['date'])
    df2 = pd.read_csv(augmented, parse_dates=['date'])

    # Add a new column to each DataFrame to indicate the dataset
    df1['dataset'] = 'Original'  # Mark df1 as 'Original'
    df2['dataset'] = 'Augmented'  # Mark df2 as 'Augmented'

    # Combine the datasets into one DataFrame
    combined_df = pd.concat([df1[['meantemp', 'humidity', 'wind_speed', 'meanpressure', 'dataset']],
                             df2[['meantemp', 'humidity', 'wind_speed', 'meanpressure', 'dataset']]])

    # Create the pair plot
    sns.pairplot(combined_df, hue='dataset', markers=["o", "s"], palette='husl', height=2.5)

    plt.suptitle('Pair Plot for All Variables', y=1.02)
    # plt.show()
    save_eval_plot('distributions 2')


# Statistical Tests
# Kolmogorov-Smirnov Test. A non-parametric test to compare the distributions of datasets
def statistical_tests(original, augmented):
    # Load the datasets
    df1 = pd.read_csv(original, parse_dates=['date'])
    df2 = pd.read_csv(augmented, parse_dates=['date'])

    ks_results = {}
    for column in ['meantemp', 'humidity', 'wind_speed', 'meanpressure']:
        ks_stat, ks_p_value = stats.ks_2samp(df1[column], df2[column])
        ks_results[column] = (ks_stat, ks_p_value)

    print("KS Test Results:\n", ks_results)


# Correlation Analysis
def corr_analysis(data):
    # Load the datasets
    df = pd.read_csv(data, parse_dates=['date'])

    correlation_df = df[['meantemp', 'humidity', 'wind_speed', 'meanpressure']].corr()

    # Correlation Heatmaps
    plt.figure(figsize=(12, 6))
    sns.heatmap(correlation_df, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap - original')
    # plt.show()
    save_eval_plot('corr_analysis')


'''
time-series data intrinsic properties:
trends, seasonality, autocorrelation, stationarity
'''

def trend_seas_resid(data):
    df = pd.read_csv(data, parse_dates=['date'])
    result = seasonal_decompose(df['meantemp'], model='additive', period=365)
    result.plot()
    # plt.show()
    save_eval_plot('trend_seas_resid')

    # You can access individual components
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid

'''
Autocorrelation:
Strong: Indicates that data points in the time series are highly dependent on their past values. 
Low: Data points are less dependent on their past values, showing little or no recurring pattern or trend over time.
'''

# Evaluation of Autocorrelation
def autocorrelation(data):

    df = pd.read_csv(data, parse_dates=['date'])
    # Autocorrelation plot
    plot_acf(df['meantemp'], lags=50)
    # plt.show()
    save_eval_plot('autocorrelation')

    # Partial autocorrelation plot
    plot_pacf(df['meantemp'], lags=50)
    # plt.show()
    save_eval_plot('part_autocorrelation')

# Save plot as PNG
def save_eval_plot(plot_name):
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    save_path = 'plots/' + plot_name + f'_{timestamp}.png'
    plt.savefig(save_path)

    plt.show()
