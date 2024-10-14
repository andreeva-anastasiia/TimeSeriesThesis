import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd



# Generator class
class Generator(nn.Module):
    def __init__(self, noise_dim, condition_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(noise_dim + condition_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, noise, condition):
        x = torch.cat([noise, condition], dim=1)  # Concatenate noise and condition
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Critic (Discriminator) class
class Critic(nn.Module):
    def __init__(self, data_dim, condition_dim, hidden_dim):
        super(Critic, self).__init__()
        total_input_dim = data_dim + condition_dim  # Adjusted input size (4 features + 1 condition)
        self.fc1 = nn.Linear(total_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, data, condition):
        # Concatenate data (real/fake) and condition
        x = torch.cat([data, condition], dim=1)  # Concatenated shape will be [batch_size, 5]
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Gradient Penalty for WGAN-GP
def gradient_penalty(critic, real_data, fake_data, condition):
    alpha = torch.rand(real_data.size(0), 1)
    alpha = alpha.expand_as(real_data)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)

    critic_interpolates = critic(interpolates, condition)
    gradients = torch.autograd.grad(
        outputs=critic_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(critic_interpolates),
        create_graph=True, retain_graph=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty


# GAN class definition
class CWGAN:
    def __init__(self, data_df, timestamps_df, headers, noise_dim=10, condition_dim=1, hidden_dim=64, output_dim=4,
                 lambda_gp=10):
        self.data_df = data_df  # Data DataFrame with features (e.g., meantemp, humidity, etc.)
        self.timestamps_df = timestamps_df # Timestamps DataFrame (with date or time info)
        self.headers = headers
        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lambda_gp = lambda_gp

        # Convert timestamps to numeric values (e.g., day_of_year)
        # self.timestamps_df['numeric_time'] = pd.to_datetime(self.timestamps_df.iloc[:, 0]).dt.dayofyear
        # self.timestamps_df['numeric_time'] = pd.to_datetime(self.timestamps_df).dt.dayofyear
        self.timestamps_df = pd.DataFrame(self.timestamps_df)  # Ensure it's a DataFrame
        self.timestamps_df['numeric_time'] = pd.to_datetime(self.timestamps_df.iloc[:, 0]).dt.dayofyear


        # Initialize generator and critic
        self.generator = Generator(noise_dim, condition_dim, hidden_dim, output_dim)
        self.critic = Critic(output_dim, condition_dim, hidden_dim)

        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.5))
        self.c_optimizer = optim.Adam(self.critic.parameters(), lr=0.0002, betas=(0.5, 0.5))

    # Function to generate synthetic data
    def generate_data(self, condition):
        noise = torch.randn(condition.size(0), self.noise_dim)
        fake_data = self.generator(noise, condition)
        return fake_data

    # Function to retrieve real data and conditions in batches
    def get_batch(self, batch_size):
        indices = np.random.randint(0, len(self.data_df), batch_size)

        # Extract real data (features: meantemp, humidity, etc.) from the DataFrame
        real_data = torch.tensor(self.data_df[indices], dtype=torch.float32)

        # Extract corresponding conditions (numeric representation of the timestamps)
        condition_data = torch.tensor(self.timestamps_df.iloc[indices]['numeric_time'].values,
                                      dtype=torch.float32).unsqueeze(1)

        return real_data, condition_data

    # Training function
    def train(self, num_epochs=1000, critic_iterations=5, batch_size=32):
        for epoch in range(num_epochs):
            for _ in range(critic_iterations):
                # Get real data and condition
                real_data, condition = self.get_batch(batch_size)

                # Generate fake data
                noise = torch.randn(batch_size, self.noise_dim)
                fake_data = self.generator(noise, condition).detach()  # Detach to avoid training G

                # Train Critic
                real_output = self.critic(real_data, condition)
                fake_output = self.critic(fake_data, condition)

                # Compute Wasserstein loss
                c_loss = -(real_output.mean() - fake_output.mean())

                # Gradient Penalty
                gp = gradient_penalty(self.critic, real_data, fake_data, condition)
                total_c_loss = c_loss + self.lambda_gp * gp

                self.c_optimizer.zero_grad()
                total_c_loss.backward()
                self.c_optimizer.step()

            # Train Generator
            noise = torch.randn(batch_size, self.noise_dim)
            fake_data = self.generator(noise, condition)
            fake_output = self.critic(fake_data, condition)
            g_loss = -fake_output.mean()

            self.g_optimizer.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()

            if epoch % 100 == 0:
                print(
                    f'Epoch [{epoch}/{num_epochs}], Critic Loss: {total_c_loss.item()}, Generator Loss: {g_loss.item()}')


    # Assume `self.generator` is your generator model
    # Assume `num_samples` is the number of samples you want to generate
    # Assume `noise_dim` is the dimensionality of the noise input to the generator
    def generate_fake_data(self, num_samples):
        # Generate random noise
        noise_dim = 10
        noise = torch.randn(num_samples, noise_dim)  # Adjust noise_dim accordingly

        # Generate conditions (e.g., numeric representations of timestamps)
        # Assuming you have a way to create conditions
        # Here we just create random conditions for demonstration
        condition = torch.randint(1, 366, (num_samples, 1), dtype=torch.float32)  # Assuming days of the year

        # Generate fake data
        fake_data_tensor = self.generator(noise, condition).detach()  # Detach to avoid tracking gradients

        # Convert to NumPy array
        fake_data_array = fake_data_tensor.numpy()

        # Create DataFrame
        headers_list = self.headers.tolist()  # Converts Index to a list
        fake_data_df = pd.DataFrame(fake_data_array, columns=headers_list[1:])

        return fake_data_df


