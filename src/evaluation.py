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



# Histograms #relative balance of characteristics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def histograms(original, augmented):
    # Load the datasets
    df1 = pd.read_csv(original, parse_dates=['date'])
    df2 = pd.read_csv(augmented, parse_dates=['date'])

    # Find common columns between the two DataFrames, excluding 'date' and any additional non-numeric columns
    common_columns = df1.columns.intersection(df2.columns).difference(['date'])
    numeric_columns = [col for col in common_columns if pd.api.types.is_numeric_dtype(df1[col])]

    # Set up the plotting grid dynamically based on the number of columns
    n_cols = 2  # Number of columns in the grid
    n_rows = (len(numeric_columns) + 1) // n_cols  # Number of rows needed
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    axs = axs.flatten()  # Flatten for easy indexing if more than 4 columns

    # Generate histograms for each numeric common column
    for i, column in enumerate(numeric_columns):
        sns.histplot(df1[column], kde=True, ax=axs[i], color='blue', label='Original', stat='density')
        sns.histplot(df2[column], kde=True, ax=axs[i], color='orange', label='Augmented', stat='density')
        axs[i].set_title(column)
        axs[i].legend()

    # Hide any extra subplots (if more grid space than columns)
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    # plt.show()  # Uncomment to display the plot
    save_eval_plot('histograms')
# def histograms(original, augmented):
#
#     # Load the datasets
#     df1 = pd.read_csv(original, parse_dates=['date'])
#     df2 = pd.read_csv(augmented, parse_dates=['date'])
#
#     fig, axs = plt.subplots(2, 2, figsize=(12, 10))
#
#     for i, column in enumerate(['meantemp', 'humidity', 'wind_speed', 'meanpressure']):
#         sns.histplot(df1[column], kde=True, ax=axs[i//2, i%2], color='blue', label='original', stat='density')
#         sns.histplot(df2[column], kde=True, ax=axs[i//2, i%2], color='orange', label='augmented', stat='density')
#         axs[i//2, i%2].set_title(column)
#         axs[i//2, i%2].legend()
#
#     plt.tight_layout()
#     #plt.show()
#     save_eval_plot('histograms')

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


'''
3.  Compare how generated data distributions with the original time-series data
'''
def basic_statistics(data):

    # Load the dataset
    df = pd.read_csv(data, parse_dates=['date'])

    print("Dataset 1 Statistics:\n", df.describe())


# Distributions of characteristics for 1 dataset
# Pairplot for all variables
def distributions(data):
    # Load the dataset
    df1 = pd.read_csv(data, parse_dates=['date'])

    # Select numeric columns, excluding 'date' if it exists
    numeric_columns = [col for col in df1.columns if pd.api.types.is_numeric_dtype(df1[col]) and col != 'date']

    # Create pair plot for numeric columns only
    sns.pairplot(df1[numeric_columns])
    plt.suptitle('Pair Plot for All Variables', y=1.02)
    # plt.show()  # Uncomment to display the plot
    save_eval_plot('distributions')



# Distributions of characteristics for 2 datasets
# Pairplot for all variables
def distributions2(original, augmented):
    # Load the datasets
    df1 = pd.read_csv(original, parse_dates=['date'])
    df2 = pd.read_csv(augmented, parse_dates=['date'])

    # Add a new column to each DataFrame to indicate the dataset
    df1['dataset'] = 'Original'  # Mark df1 as 'Original'
    df2['dataset'] = 'Augmented'  # Mark df2 as 'Augmented'

    # Find the common columns (excluding the 'dataset' column that we added)
    common_columns = df1.columns.intersection(df2.columns).difference(['dataset'])

    # Combine the datasets into one DataFrame with the common columns and the 'dataset' column
    combined_df = pd.concat([df1[common_columns.union(['dataset'])],
                             df2[common_columns.union(['dataset'])]], ignore_index=True)

    # Create the pair plot using Seaborn
    sns.pairplot(combined_df, hue='dataset', markers=["o", "s"], palette='husl', height=2.5)

    plt.suptitle('Pair Plot for All Variables', y=1.02)
    # plt.show()  # Uncomment to display the plot
    save_eval_plot('distributions_2')


# Statistical Tests
# Kolmogorov-Smirnov Test. A non-parametric test to compare the distributions of datasets
def statistical_tests(original, augmented):
    # Load the datasets
    df1 = pd.read_csv(original, parse_dates=['date'])
    df2 = pd.read_csv(augmented, parse_dates=['date'])

    # Identify common numeric columns, excluding 'date' if it exists
    common_columns = df1.columns.intersection(df2.columns).difference(['date'])
    numeric_columns = [col for col in common_columns if pd.api.types.is_numeric_dtype(df1[col])]

    # Initialize dictionary to store KS test results
    ks_results = {}

    # Perform KS test for each numeric common column
    for column in numeric_columns:
        ks_stat, ks_p_value = stats.ks_2samp(df1[column].dropna(), df2[column].dropna())
        ks_results[column] = (ks_stat, ks_p_value)

    print("KS Test Results:\n", ks_results)



'''
5.  Compare cross-variable dependencies. 
'''

# Correlation Analysis
def corr_analysis(data):
    # Load the dataset
    df = pd.read_csv(data, parse_dates=['date'])

    # Select numeric columns, excluding 'date' if it exists
    numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != 'date']

    # Calculate the correlation matrix for numeric columns
    correlation_df = df[numeric_columns].corr()

    # Plot the Correlation Heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(correlation_df, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    # plt.show()  # Uncomment to display the plot
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
    seasonality = result.seasonal
    residual = result.resid



'''
    4. Does the generated data capture the same correlation dependencies 
    within each variable with its past values
    Autocorrelation:
    Strong: Indicates that data points in the time series are highly dependent on their past values. 
    Low: Data points are less dependent on their past values, showing little or no recurring pattern or trend over time.
    
    Y - The autocorrelation coefficient (from -1 to 1). 
    Values close to 1 indicate a strong positive correlation, 
    Values close to -1 indicate a strong negative correlation. 
    
    X - The lag, or the number of time periods between the observations compared
'''

# Evaluation of Autocorrelation
# def autocorrelation(data):
#
#     df = pd.read_csv(data, parse_dates=['date'])
#     # Autocorrelation plot
#     plot_acf(df['meantemp'], lags=50)
#     # plt.show()
#     save_eval_plot('autocorrelation')
#
#     # Partial autocorrelation plot
#     plot_pacf(df['meantemp'], lags=50)
#     # plt.show()
#     save_eval_plot('part_autocorrelation')


def autocorrelation(data):
    # Load the dataset
    df = pd.read_csv(data, parse_dates=['date'])

    # Select numeric columns, excluding 'date' if it exists
    numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != 'date']

    # Iterate over each numeric column to plot autocorrelation and partial autocorrelation
    for column in numeric_columns:
        # Autocorrelation plot
        plt.figure(figsize=(12, 6))
        plot_acf(df[column], lags=50)
        plt.title(f'Autocorrelation Plot for {column}')
        # plt.show()  # Uncomment to display the plot
        save_eval_plot(f'autocorrelation_{column}')

        # Partial autocorrelation plot
        plt.figure(figsize=(12, 6))
        plot_pacf(df[column], lags=50)
        plt.title(f'Partial Autocorrelation Plot for {column}')
        # plt.show()  # Uncomment to display the plot
        save_eval_plot(f'part_autocorrelation_{column}')

# Save plot as PNG
def save_eval_plot(plot_name):
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    save_path = 'plots/' + plot_name + f'_{timestamp}.png'
    plt.savefig(save_path)

    plt.show()
