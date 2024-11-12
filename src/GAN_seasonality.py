from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import tensorflow as tf
import numpy as np
import pandas as pd

from src.plots import plot_losses


def extract_components(feature):
    result = seasonal_decompose(feature, model="additive", period=365)  # Assuming yearly seasonality
    seasonality = result.seasonal
    # trend = result.trend
    # residual = result.resid

    # Extract the trend component and handle missing values
    trend = result.trend.fillna(method='bfill').fillna(method='ffill')
    residual = result.resid.fillna(method='bfill').fillna(method='ffill')

    return seasonality, trend, residual


def save_component(component, path):
    component.to_csv(path)


def prepare_gan_data(seasonality, sequence_length=365):
    #scaler = MinMaxScaler(feature_range=(0, 1))
    '''
    You can scale the data with a more appropriate range,
    like feature_range=(-1, 1) to better capture the natural variations,
    including negative values.
    Both the real and generated data should lie in the same range.
    '''
    scaler = MinMaxScaler(feature_range=(-1, 1))  # Change to (-1, 1) for better negative/positive representation

    seasonality_scaled = scaler.fit_transform(seasonality.values.reshape(-1, 1))

    seasonality_sequences = []

    for i in range(len(seasonality_scaled) - sequence_length):
        seasonality_sequences.append(seasonality_scaled[i:i + sequence_length])

    return np.array(seasonality_sequences), scaler

def prepare_gan_data_trend(trend, sequence_length=1462):
    # Normalize trend data (GANs usually perform better with normalized data)
    # scaler = MinMaxScaler()
    trend_scaler = MinMaxScaler(feature_range=(-1, 1))
    trend_scaled = trend_scaler.fit_transform(trend.values.reshape(-1, 1))
    # Prepare trend sequences (sliding window approach)
    trend_sequences = np.array([trend_scaled[i:i + sequence_length]
                                for i in range(len(trend_scaled) - sequence_length + 1)])
    return trend_sequences, trend_scaler

# 5. Build GAN models

def build_generator_seas(sequence_length, noise_dim=100):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_dim=noise_dim),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        #tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(sequence_length, activation='tanh'),
        tf.keras.layers.Reshape((sequence_length, 1))  # Reshape to ensure output shape is (batch_size, sequence_length, 1)
    ])
    return model

def build_discriminator_seas(sequence_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(sequence_length, 1)),  # make input shape compatible
        #tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer
    ])
    return model


def build_gan_seas(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

'''
generator, discriminator, GAN for TREND
'''

def build_generator(sequence_length, noise_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_dim=noise_dim),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(sequence_length, activation='linear')  # Output synthetic trend sequence
    ])
    return model

def build_discriminator(sequence_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(sequence_length,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output probability (real or fake)
    ])
    return model

def build_gan(generator, discriminator):
    model = tf.keras.Sequential([generator, discriminator])
    return model


#train trend
def trend_generate(trend_sequences, scaler, sequence_length=1462, noise_dim=100, epochs=10000, batch_size=32):
    """
    Train a GAN to generate synthetic trend data.
    """
    # Adjust sequence length to match the entire dataset (1462 rows)
    generator = build_generator(sequence_length, noise_dim)
    discriminator = build_discriminator(sequence_length)

    # Use Adam optimizer
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0002, clipnorm=1.0)
    discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)
    gan = build_gan(generator, discriminator)
    gan.compile(loss="binary_crossentropy", optimizer=optimizer)

    d_losses = []
    g_losses = []

    for epoch in range(epochs):
        # Generate random noise
        noise = np.random.normal(0, 1, (batch_size, noise_dim)).astype(np.float32)

        # Generate fake sequences
        generated_sequences = generator.predict(noise).reshape(-1, sequence_length)

        # Select a random batch of real sequences
        idx = np.random.randint(0, trend_sequences.shape[0], batch_size)
        real_sequences = trend_sequences[idx].reshape(-1, sequence_length)

        # Train the discriminator
        real_labels = np.ones((batch_size, 1)) * 0.9  # Smoothed real labels
        fake_labels = np.zeros((batch_size, 1)) + 0.1  # Smoothed fake labels

        d_loss_real = discriminator.train_on_batch(real_sequences, real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_sequences, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        d_losses.append(d_loss)

        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        valid_labels = np.ones((batch_size, 1))  # Real labels for generator
        g_loss = gan.train_on_batch(noise, valid_labels)
        g_losses.append(g_loss)

        # Print losses every epoch
        print(f"Epoch {epoch}, D Loss: {d_loss}, G Loss: {g_loss}")

    plot_losses(d_losses, g_losses)
    # Generate new synthetic trend
    noise = np.random.normal(0, 1, (1, noise_dim))
    generated_trend = generator.predict(noise)
    generated_trend_original = scaler.inverse_transform(generated_trend.reshape(-1, 1))

    return generated_trend_original

# def trend_generate(trend_sequences, trend_scaler, sequence_length=1098, noise_dim=100, epochs=10000, batch_size=32):
#     """
#     Train a GAN to generate synthetic trend data based on truncated sequence (182 to 1279 positions).
#     """
#     # Build generator and discriminator with adjusted sequence length
#     generator = build_generator(sequence_length, noise_dim)
#     discriminator = build_discriminator(sequence_length)
#
#     # Use Adam optimizer
#     optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0002, clipnorm=1.0)
#     discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)
#     gan = build_gan(generator, discriminator)
#     gan.compile(loss="binary_crossentropy", optimizer=optimizer)
#
#     d_losses = []
#     g_losses = []
#
#     # Iterate over epochs for GAN training
#     for epoch in range(epochs):
#         # Generate random noise for generator input
#         noise = np.random.normal(0, 1, (batch_size, noise_dim)).astype(np.float32)
#
#         # Generate fake sequences from generator
#         generated_sequences = generator.predict(noise).reshape(-1, sequence_length)
#
#         # Select a random batch of real trend sequences
#         idx = np.random.randint(0, trend_sequences.shape[0], batch_size)
#         real_sequences = trend_sequences[idx, 182:1280].reshape(-1, sequence_length)
#
#         # Train the discriminator with real and fake sequences
#         real_labels = np.ones((batch_size, 1)) * 0.9  # Smoothed real labels
#         fake_labels = np.zeros((batch_size, 1)) + 0.1  # Smoothed fake labels
#
#         d_loss_real = discriminator.train_on_batch(real_sequences, real_labels)
#         d_loss_fake = discriminator.train_on_batch(generated_sequences, fake_labels)
#         d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
#         d_losses.append(d_loss)
#
#         # Train the generator
#         noise = np.random.normal(0, 1, (batch_size, noise_dim))
#         valid_labels = np.ones((batch_size, 1))  # Real labels for generator
#         g_loss = gan.train_on_batch(noise, valid_labels)
#         g_losses.append(g_loss)
#
#         print(f"Epoch {epoch}, D Loss: {d_loss}, G Loss: {g_loss}")
#
#     # Plot the GAN losses
#     plot_losses(d_losses, g_losses)
#
#     # Generate new synthetic trend sequence
#     noise = np.random.normal(0, 1, (1, noise_dim))
#     generated_trend = generator.predict(noise)
#     generated_trend_original = trend_scaler.inverse_transform(generated_trend.reshape(-1, 1))
#
#     return generated_trend_original



# train seasonality
def seasonality_generate(seasonality_sequences, scaler, sequence_length=365, noise_dim=100, epochs=7000,
                         batch_size=32):


    generator = build_generator_seas(sequence_length, noise_dim)
    discriminator = build_discriminator_seas(sequence_length)

    # Use separate optimizers for the generator and discriminator
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, clipnorm=1.0)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)

    # Compile discriminator with its optimizer
    discriminator.compile(loss="binary_crossentropy", optimizer=discriminator_optimizer)

    # Build the GAN model but make the discriminator untrainable when training the GAN
    gan = build_gan_seas(generator, discriminator)
    gan.compile(loss="binary_crossentropy", optimizer=generator_optimizer)

    d_losses = []
    g_losses = []

    for epoch in range(epochs):
        # Generate random noise for the generator input
        noise = np.random.normal(0, 1, (batch_size, noise_dim)).astype(np.float32)

        # Generate fake sequences from the generator
        generated_sequences = generator.predict(noise).reshape(-1, sequence_length, 1)

        # Select a random batch of real sequences
        idx = np.random.randint(0, seasonality_sequences.shape[0], batch_size)
        real_sequences = seasonality_sequences[idx].reshape(-1, sequence_length, 1)

        # Create labels for real and fake data
        # real_labels = np.ones((batch_size, 1))
        # fake_labels = np.zeros((batch_size, 1))

        # Use label smoothing
        real_labels = np.ones((batch_size, 1)) * 0.9  # Real labels smoothed to 0.9
        fake_labels = np.zeros((batch_size, 1)) + 0.1  # Fake labels smoothed to 0.1

        # Train the discriminator on real and fake data separately
        d_loss_real = discriminator.train_on_batch(real_sequences, real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_sequences, fake_labels)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        # Store the discriminator loss
        d_losses.append(d_loss)

        # Train the generator to trick the discriminator (make it think the generated sequences are real)
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        valid_labels = np.ones((batch_size, 1))  # Labels for generated data that should look real
        g_loss = gan.train_on_batch(noise, valid_labels)

        # Store the generator loss
        g_losses.append(g_loss)

        # Check for NaN or infinite losses and report
        if np.isnan(d_loss) or np.isinf(d_loss):
            print("Discriminator loss is NaN or infinite!")
        if np.isnan(g_loss) or np.isinf(g_loss):
            print("Generator loss is NaN or infinite!")

        # Print losses every epoch
        print(f"Epoch {epoch}, D Loss: {d_loss}, G Loss: {g_loss}")
        #
        # # Visualize generated sequences every 500 epochs
        # if epoch % 500 == 0:
        #     plt.plot(generated_sequences[0], label='Generated')
        #     plt.plot(real_sequences[0], label='Real')
        #     plt.title(f'Epoch {epoch}')
        #     plt.legend()
        #     plt.show()

    plot_losses(d_losses, g_losses)

    # Generate a new sequence of seasonality using the generator
    noise = np.random.normal(0, 1, (1, noise_dim))
    generated_seasonality = generator.predict(noise)

    # Inverse transform the generated data to its original scale
    generated_seasonality_original = scaler.inverse_transform(generated_seasonality.reshape(-1, 1))
    return generated_seasonality_original


import numpy as np
import pandas as pd


def built_new_feature(trend, seasonality, residual, df, col_name):
    # Ensure the target length matches the DataFrame's length
    # target_length = len(df)
    # print(target_length)

    # print(seasonality)

    # Extend `trend` and `seasonality` to exactly match the target length
    # if len(trend) < target_length:
    #     trend = np.resize(trend, target_length)  # Resize to fit exactly
    # if len(seasonality) < target_length:
    #     seasonality = np.resize(seasonality, target_length)  # Resize to fit exactly
    # if len(seasonality) != target_length:
    # seasonality = np.tile(seasonality, (target_length // len(seasonality)))[:target_length]


    # Convert to Series and align with DataFrame index
    trend = pd.Series(trend, index=df.index)
    seasonality = seasonality.ravel()
    seasonality = pd.Series(seasonality, index=df.index)
    residual = pd.Series(residual, index=df.index)  # Ensure residual is also aligned

    # Combine the components
    reconstructed_series = trend + seasonality + residual

    # Add the reconstructed series to the DataFrame
    df[col_name] = reconstructed_series
    print(f"{col_name} successfully added to the DataFrame.")




# Save results

def save_generated_seasonality(generated_seasonality, path):
    #np.savetxt(path, generated_seasonality, delimiter=",")
    # Convert the generated seasonality to a pandas DataFrame
    df_generated = pd.DataFrame(generated_seasonality, columns=['seasonal'])

    # Add an index column starting from 0
    df_generated.index.name = 'Index'  # This will name the index column as "Index"

    # Save the DataFrame to CSV without the index but with the header
    df_generated.to_csv(path, header=True)


def save_generated_trend(generated_trend, path):
    #np.savetxt(path, generated_seasonality, delimiter=",")
    # Convert the generated trend to a pandas DataFrame
    df_generated = pd.DataFrame(generated_trend, columns=['trend'])

    # Add an index column starting from 0
    df_generated.index.name = 'Index'  # This will name the index column as "Index"

    # Save the DataFrame to CSV without the index but with the header
    df_generated.to_csv(path, header=True)


