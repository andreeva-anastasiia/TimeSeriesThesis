import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# Generator
class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class GAN:
    def __init__(self, data, input_size=100, lr=0.0001, num_epochs=10000, batch_size=64):
        # Normalize data
        self.scaler = MinMaxScaler()
        self.data_scaled = self.scaler.fit_transform(data)

        # Hyperparameters
        self.input_size = input_size  # Random noise vector size
        self.output_size = self.data_scaled.shape[1]  # Number of features
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Create the models
        self.generator = Generator(self.input_size, self.output_size)
        self.discriminator = Discriminator(self.output_size)

        # Optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr)

        # Loss function
        self.loss_fn = nn.BCELoss()



    def train(self):
        for epoch in range(self.num_epochs):
            # Train discriminator on real data
            real_data = torch.tensor(self.data_scaled, dtype=torch.float32)
            real_labels = torch.ones((real_data.size(0), 1))

            self.optimizer_D.zero_grad()
            real_output = self.discriminator(real_data)
            d_loss_real = self.loss_fn(real_output, real_labels)

            # Train discriminator on fake data
            noise = torch.randn((self.batch_size, self.input_size))  # Noise vector for fake data generation
            fake_data = self.generator(noise)
            fake_labels = torch.zeros((self.batch_size, 1))

            fake_output = self.discriminator(fake_data.detach())
            d_loss_fake = self.loss_fn(fake_output, fake_labels)

            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.optimizer_D.step()

            # Train generator
            self.optimizer_G.zero_grad()
            fake_output = self.discriminator(fake_data)

            # Create real labels with the same batch size
            real_labels = torch.ones((self.batch_size, 1))  # Create labels with shape [batch_size, 1]

            g_loss = self.loss_fn(fake_output, real_labels)  # We want generator to fool discriminator
            g_loss.backward()
            self.optimizer_G.step()

            # Print loss every 1000 epochs
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')


    # Generate synthetic data using trained generator
    def generate_synthetic_data(self, num_samples):

        noise = torch.randn((num_samples, self.input_size))  # Generate random noise
        synthetic_data = self.generator(
            noise).detach().numpy()  # Generate synthetic data and detach from the computation graph
        synthetic_data = self.scaler.inverse_transform(synthetic_data)  # Inverse transform to original scale
        return synthetic_data