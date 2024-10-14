from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd



# 2. Decompose time series to extract seasonality
def extract_seasonality(meantemp):
    result = seasonal_decompose(meantemp, model="additive", period=365)  # Assuming yearly seasonality
    seasonality = result.seasonal
    return seasonality


# 3. Save seasonality to a file
def save_seasonality(seasonality, path):
    seasonality.to_csv(path)


# 4. Prepare data for GAN
def prepare_gan_data(seasonality, sequence_length=365):
    scaler = MinMaxScaler(feature_range=(0, 1))
    seasonality_scaled = scaler.fit_transform(seasonality.values.reshape(-1, 1))

    seasonality_sequences = []
    for i in range(len(seasonality_scaled) - sequence_length):
        seasonality_sequences.append(seasonality_scaled[i:i + sequence_length])

    return np.array(seasonality_sequences), scaler


# 5. Build GAN models
def build_generator(sequence_length, noise_dim=100):
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, input_dim=noise_dim, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(sequence_length, activation='tanh'))  # Output sequence
    return model


def build_discriminator(sequence_length):
    model = tf.keras.Sequential()
    model.add(layers.Dense(512, input_shape=(sequence_length,), activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Real/Fake classification
    return model


def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model


# 6. Train the GAN
def seasonality_generate(seasonality_sequences, scaler, sequence_length=365, noise_dim=100, epochs=10000, batch_size=32):
    generator = build_generator(sequence_length, noise_dim)
    discriminator = build_discriminator(sequence_length)
    discriminator.compile(loss="binary_crossentropy", optimizer="adam")

    gan = build_gan(generator, discriminator)
    gan.compile(loss="binary_crossentropy", optimizer="adam")

    for epoch in range(epochs):
        # Generate random noise
        noise = np.random.normal(0, 1, (batch_size, noise_dim))

        # Generate fake sequences
        generated_sequences = generator.predict(noise)

        # Select a random batch of real sequences
        idx = np.random.randint(0, seasonality_sequences.shape[0], batch_size)
        real_sequences = seasonality_sequences[idx]

        # Labels for real and fake sequences
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(real_sequences, real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_sequences, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        valid_labels = np.ones((batch_size, 1))  # Trick the GAN into thinking these are real
        g_loss = gan.train_on_batch(noise, valid_labels)

        # Print progress every 1000 epochs
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss}, G Loss: {g_loss}")

    # Generate new seasonality
    noise = np.random.normal(0, 1, (1, noise_dim))
    generated_seasonality = generator.predict(noise)

    # Inverse transform the generated data
    generated_seasonality_original = scaler.inverse_transform(generated_seasonality.reshape(-1, 1))
    return generated_seasonality_original


# 7. Save generated seasonality
def save_generated_seasonality(generated_seasonality, path):
    np.savetxt(path, generated_seasonality, delimiter=",")


