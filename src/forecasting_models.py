import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split

def data_split(file_path):
    # Load the data
    data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')

    # Display the first few rows of the dataset
    print(data.head())

    # Split the data into training and testing sets
    train_size = int(len(data) * 0.8)  # Use 80% of the data for training
    train, test = data[:train_size], data[train_size:]

    print(train.head())
    print(test.head())

    return train, test


# Function to fit SARIMAX and make predictions
# SARIMAX - Seasonal Autoregressive Integrated Moving Average with eXogenous regressors
def fit_sarimax(train, test, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    # Get the target variable (first column) and exogenous variables (remaining columns)
    target_column = train.columns[0]  # First column as target
    exog_columns = train.columns[1:]  # Remaining columns as exogenous variables

    # Fit the SARIMAX model
    model = SARIMAX(train[target_column], exog=train[exog_columns],
                    order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)

    # Make predictions
    predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1,
                                    exog=test[exog_columns])

    return predictions, model_fit

# performance metrics evaluation

def perform_metrics(test, predicted_temp):

    # Assuming you have your actual and predicted values
    actual_values = test[test.columns[0]]  # Actual values (e.g., the first column)
    predicted_values = predicted_temp  # Predicted values from the SARIMAX model

    # Calculate performance metrics
    mae = mean_absolute_error(actual_values, predicted_values)
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
    r_squared = 1 - (np.sum((actual_values - predicted_values) ** 2) / np.sum((actual_values - np.mean(actual_values)) ** 2))

    # Print the metrics
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
    print(f'R-squared: {r_squared:.2f}')