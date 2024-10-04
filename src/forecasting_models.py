from statsmodels.tsa.statespace.sarimax import SARIMAX

# Define the SARIMAX model with exogenous variables
model = SARIMAX(target_variable,
                exog=exogenous_variables,  # Humidity, Wind Speed, Mean Pressure
                order=(p,d,q),             # ARIMA parameters
                seasonal_order=(P,D,Q,m))  # Seasonal ARIMA parameters

# Fit the model
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=number_of_forecast_steps, exog=new_exogenous_variables)