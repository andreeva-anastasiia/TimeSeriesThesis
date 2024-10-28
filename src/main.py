import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#from scr_trash_bin.GAN import GAN
from src.GAN_seasonality import seasonality_generate, extract_components, save_component, prepare_gan_data, \
    save_generated_seasonality, save_generated_trend, trend_generate, prepare_gan_data_trend, built_new_feature
from src.data_augmentation import magnitude_warping, ts_mixup, jittering
from src.evaluation import basic_statistics, histograms, KDE_plots, statistical_tests, corr_analysis, \
    distributions2, autocorrelation, trend_seas_resid
from src.forecasting_models import data_split, fit_sarimax, perform_metrics
from src.plots import plot_all_features_separate_subplots, plot_component, plot_seasonalities


# Load multivariate time-series data in csv format
def load_csv(file_path):
    # Load CSV using pandas
    df = pd.read_csv(file_path)

    # Load the file with ';' as delimiter
    # df = pd.read_csv(file_path, delimiter=';')

    headers = df.columns  # Save column names

    # Assuming the first column is the timestamp and the rest are numeric features
    timestamps = df.iloc[:, 0]  # First column (timestamps)
    features = df.iloc[:, 1:].select_dtypes(include=[np.number]).to_numpy()  # All other numeric columns

    return headers, timestamps, features  # Return column names, timestamps and numeric data


# Save the augmented data with timestamps
def save_csv_with_timestamp(headers, timesteps, augmented_data, output_file):
    # Combine the timestamps with the augmented data
    augmented_df = pd.DataFrame(augmented_data, columns=headers[1:]) #using original columns names
    augmented_df.insert(0, headers[0], timesteps)  # Insert the first column from orig

    # Save the DataFrame to CSV
    augmented_df.to_csv(output_file, index=False)


# the source file
# csv_file = 'data/original/DailyDelhiClimate/DailyDelhiClimate.csv'
csv_file_path = 'data/original/GOOGL_stock_2006_to_2018/GOOGL_stock_2006_to_2018.csv'
csv_file = os.path.splitext(os.path.basename(csv_file_path))[0]

# Load data
headers, timestamps, data = load_csv(csv_file_path)

# print(timestamps)
# print(data)



# A code fragment repeats for each data augmentation method

# Magnitude warping
# augmented_data_MW = magnitude_warping(data)
# # save
# output_file_MW = 'data/augmented/' + os.path.splitext(os.path.basename(csv_file))[0] + '_MW.csv'
# save_csv_with_timestamp(headers, timestamps, augmented_data_MW, output_file_MW)
# # provide plots for each feature
# plot_all_features_separate_subplots(headers, timestamps, data, augmented_data_MW)


# TSMixup
# augmented_data_TSMixup = ts_mixup(data)
# # save
# output_file_TSMixup = 'data/augmented/' + os.path.splitext(os.path.basename(csv_file))[0] + '_TSMixup.csv'
# save_csv_with_timestamp(headers, timestamps, augmented_data_TSMixup, output_file_TSMixup)
# # provide plots for each feature
# plot_all_features_separate_subplots(headers, timestamps, data, augmented_data_TSMixup)


# Jittering
# augmented_data_Jit = jittering(data)
# # save
# output_file_Jit = 'data/augmented/' + os.path.splitext(os.path.basename(csv_file))[0] + '_Jittering.csv'
# save_csv_with_timestamp(headers, timestamps, augmented_data_Jit, output_file_Jit)
# # provide plots for each feature
# plot_all_features_separate_subplots(headers, timestamps, data, augmented_data_Jit)


# Permutation
# augmented_data_Perm = permutation(data, 4)
# #save
# output_file_Perm = 'data/augmented/' + os.path.splitext(os.path.basename(csv_file))[0] + '_Permutation.csv'
# save_csv_with_timestamp(headers, timestamps, augmented_data_Perm, output_file_Perm)
# # provide plots for each feature
# plot_all_features_separate_subplots(headers, timestamps, data, augmented_data_Perm)


# Create the GAN instance
# gan = GAN(data)
# gan.train()
#
# synthetic_data_GAN = gan.generate_synthetic_data(1462)
# output_file_GAN = 'data/synthetic/' + os.path.splitext(os.path.basename(csv_file))[0] + '_GAN.csv'
# save_csv_with_timestamp(headers, timestamps, synthetic_data_GAN, output_file_GAN)
# plot_all_features_separate_subplots(headers, timestamps, data, synthetic_data_GAN)


# # Create the CWGAN instance
# cwgan = CWGAN(data, timestamps, headers)
# cwgan.train(num_epochs=1000, batch_size=64)
#
# synthetic_data_CWGAN = cwgan.generate_fake_data(1462)
# output_file_CWGAN = 'data/synthetic/' + os.path.splitext(os.path.basename(csv_file))[0] + '_CWGAN.csv'
# save_csv_with_timestamp(headers, timestamps, synthetic_data_CWGAN, output_file_CWGAN)
# #plot_all_features_separate_subplots(headers, timestamps, data, synthetic_data_CWGAN)




# #evaluation
#
# basic_statistics(csv_file)
# basic_statistics(output_file_GAN)
#
# histograms(csv_file, output_file_GAN)
# KDE_plots(csv_file, output_file_GAN)
#
# distributions2(csv_file, output_file_GAN)
#
# statistical_tests(csv_file, output_file_GAN)
#
# corr_analysis(csv_file)
# corr_analysis(output_file_GAN)
#
# trend_seas_resid(csv_file)
# trend_seas_resid(output_file_GAN)
# autocorrelation(csv_file)


#evaluation

# basic_statistics(csv_file_path)
#
# corr_analysis(csv_file_path)

# trend_seas_resid(csv_file_path)
#
# autocorrelation(csv_file_path)





# gan = GAN(data)
# gan.train()
#
# synthetic_data_GAN = gan.generate_synthetic_data(1462)
# output_file_GAN = 'data/synthetic/' + os.path.splitext(os.path.basename(csv_file))[0] + '_GAN.csv'
# save_csv_with_timestamp(headers, timestamps, synthetic_data_GAN, output_file_GAN)
# plot_all_features_separate_subplots(headers, timestamps, data, synthetic_data_GAN)





# Load data
df = pd.read_csv(csv_file_path, parse_dates=['date'])
df_gan = df

# meantemp = df["meantemp"]  # Use 'df' here to extract the 'meantemp' column
# Dictionary to store each column as a separate variable
# data_columns = {col: df[col] for col in df.columns}

# Dictionary to store each feature, but without date
data_columns = {col: df[col] for col in df.columns if col != 'date'}


'''
    A loop for all data features
'''
# for col_name, col_data in data_columns.items():
#     # print(f"{col_name}:")
#     # print(col_data.head())  # or any operation you want to pe
#     feature = df[f"{col_name}"]  # extract the feature column
#     seasonality, trend, residual = extract_components(feature)
#     # save_component(seasonality, "data/original/DailyDelhiClimate_seasonality_"+f"{col_name}.csv")
#     # save_component(trend, "data/original/DailyDelhiClimate_trend_"+f"{col_name}.csv")
#     # save_component(residual, "data/original/DailyDelhiClimate_residual_"+f"{col_name}.csv")
#
#     save_component(seasonality, "data/original/" + f"{csv_file}/{csv_file}_seasonality_" + f"{col_name}.csv")
#     save_component(trend, "data/original/" + f"{csv_file}/{csv_file}_trend_" + f"{col_name}.csv")
#     save_component(residual, "data/original/" + f"{csv_file}/{csv_file}_residual_" + f"{col_name}.csv")
#
#     # Prepare data for GAN
#     seasonality_sequences, scaler = prepare_gan_data(seasonality)
#
#     # Train GANs and generate new seasonality
#     generated_seasonality = seasonality_generate(seasonality_sequences, scaler)
#     print(generated_seasonality)
#     print(f"current length of seasonality: {len(generated_seasonality)}")
#     print(f"needed length of dataframe: {len(df_gan)}")
#
#     target_length = len(df_gan)
#
#     # Repeat the generated_seasonality until the length exceeds the target_length
#     extended_seasonality = np.tile(generated_seasonality, (target_length // len(generated_seasonality) + 1, 1))
#
#     # Slice to get exactly target_length
#     extended_seasonality = extended_seasonality[:target_length]
#
#     print(extended_seasonality)
#
#     # Add the feature to the final .csv with generated features
#     built_new_feature(trend, extended_seasonality, residual, df_gan, f"{col_name}")
#
#     # Save the generated seasonality
#     save_generated_seasonality(extended_seasonality, "data/synthetic/" + f"{csv_file}/{csv_file}_seasonality_"
#                                + f"{col_name}.csv")
#
#
# '''
#     Loop finish
# '''
#
output_GAN_seas = "data/synthetic/" + f"{csv_file}/{csv_file}_generated_seasonality_GAN.csv"
# save_csv_with_timestamp(headers, timestamps, df_gan, output_GAN_seas)
#
# # provide plots for each feature
# # Load generated from .csv file
# headers, timestamps, data_generated = load_csv(output_GAN_seas)
# plot_all_features_separate_subplots(headers, timestamps, data, data_generated)


# Extract seasonality, trend and residual
# Save each
# seasonality, trend, residual = extract_components(meantemp)
# save_component(seasonality, "data/original/DailyDelhiClimate_seasonality_meantemp.csv")
# save_component(trend, "data/original/DailyDelhiClimate_trend_meantemp.csv")
# save_component(residual, "data/original/DailyDelhiClimate_residual_meantemp.csv")


# Prepare data for GAN
# seasonality_sequences, scaler = prepare_gan_data(seasonality)
# trend_sequences, trend_scaler = prepare_gan_data_trend(trend)

# Train GANs and generate new seasonality and trend
# generated_seasonality = seasonality_generate(seasonality_sequences, scaler)

# generated_trend = trend_generate(trend_sequences, trend_scaler)


# Save the generated seasonality, trend
# save_generated_seasonality(generated_seasonality, "data/synthetic/DailyDelhiClimate_generated_seasonality.csv")

# save_generated_trend(generated_trend, "data/synthetic/DailyDelhiClimate_generated_trend.csv")




# Example usage:
# visualize_seasonalities("data/original/DailyDelhiClimate_seasonality_meantemp.csv",
#                         "data/synthetic/DailyDelhiClimate_generated_seasonality.csv")


# The plot comparing original and generated seasonality:
# plot_seasonalities("data/original/DailyDelhiClimate_seasonality_meantemp.csv",
#                    "data/synthetic/DailyDelhiClimate_generated_seasonality.csv")


# The plot comparing original and generated trend:
# plot_component("data/original/DailyDelhiClimate_trend_meantemp.csv",
#                "data/synthetic/DailyDelhiClimate_generated_trend.csv")



'''
    Evaluation concept
'''
# 3. Does the generated time-series have almost similar variables distributions with the original time-series data?
# basic_statistics(csv_file_path)
# basic_statistics(output_GAN_seas)
#
# distributions2(csv_file_path, output_GAN_seas)
# histograms(csv_file_path, output_GAN_seas)


# 4. Does the generated data capture the same correlation dependencies
# within each variable with its past values (autocorrelation)?
# autocorrelation(csv_file_path)



# 5. Does the generated data capture similar cross-variable dependencies.
# Dependencies can be contemporaneous (relations between variables at the same timestamp) or lagged (one variable influences another after a time delay)?
# Check the corr between external influence and aug time-series to reach more variety with aug external influences

# corr_analysis(csv_file_path)
# corr_analysis(output_GAN_seas)


# 6. A global model performance
train, test = data_split(csv_file_path)
train_g, test_g = data_split(output_GAN_seas)

# Fit the model and make predictions
predicted_temp, model_fit = fit_sarimax(train, test)

predicted_temp_g, model_fit_g = fit_sarimax(train_g, test_g)

# Performance metrics to console
perform_metrics(test, predicted_temp)
perform_metrics(test_g, predicted_temp_g)

# Ensure train and test datasets are defined before calling fit_sarimax
if 'train' in locals() and 'test' in locals():
    # Fit the model and make predictions
    predicted_temp, model_fit = fit_sarimax(train, test)

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train[train.columns[0]], label='Train', color='blue')  # Train target
    plt.plot(test.index, test[test.columns[0]], label='Test', color='orange')  # Test target
    plt.plot(test.index, predicted_temp, label='Predicted', color='green')  # Predictions
    plt.title(f'{train.columns[0]} Prediction')
    plt.xlabel('Date')
    plt.ylabel(train.columns[0])
    plt.legend()
    plt.show()

    # Print the summary of the model
    print(model_fit.summary())
else:
    print("Train and test datasets are not defined.")



# Ensure train and test datasets are defined before calling fit_sarimax
if 'train' in locals() and 'test' in locals():
    # Fit the model and make predictions
    predicted_temp_g, model_fit_g = fit_sarimax(train_g, test_g)

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(train_g.index, train_g[train_g.columns[0]], label='Train', color='blue')  # Train target
    plt.plot(test_g.index, test_g[test_g.columns[0]], label='Test', color='orange')  # Test target
    plt.plot(test_g.index, predicted_temp_g, label='Predicted', color='green')  # Predictions
    plt.title(f'{train_g.columns[0]} Prediction')
    plt.xlabel('Date')
    plt.ylabel(train_g.columns[0])
    plt.legend()
    plt.show()

    # Print the summary of the model
    print(model_fit_g.summary())
else:
    print("Train and test datasets are not defined.")



# KDE_plots(csv_file_path, output_GAN_seas)


# statistical_tests(csv_file_path, output_GAN_seas)
#

#
# trend_seas_resid(csv_file_path)
# trend_seas_resid(output_GAN_seas)


