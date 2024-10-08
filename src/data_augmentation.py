import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


# Magnitude Warping
def magnitude_warping(data, sigma=0.2, knot=4):
    # data: a numpy array of shape (time_steps, features)
    # sigma: the standard deviation of the random amplitude perturbations
    # knot: the number of knots for the spline

    time_steps, features = data.shape
    warped_data = np.zeros_like(data)

    # For each feature, apply magnitude warping
    for feature in range(features):
        # Generate random points for cubic spline
        random_curve = np.random.normal(loc=1.0, scale=sigma, size=(knot,))
        # Interpolation positions (uniformly spaced knots)
        knot_positions = np.linspace(0, time_steps - 1, knot)

        # Create a cubic spline for smooth warping
        spline = CubicSpline(knot_positions, random_curve, extrapolate=True)
        warping_curve = spline(np.arange(time_steps))  # Apply spline to all time steps

        # Multiply original data by the warping curve
        warped_data[:, feature] = data[:, feature] * warping_curve

    return warped_data


# TSMixup
def ts_mixup(data, alpha=0.2):
    """
    data: Multivariate time-series data excluding timestamp
    alpha: parameter for the Beta distribution
    """
    # Initialize augmented data list
    tsmixuped_data = []

    # Iterate over the dataset and apply mixup
    for _ in range(data.shape[0]):
        # Randomly choose two different indices
        i, j = np.random.choice(data.shape[0], 2, replace=False)

        # Generate a lambda from Beta distribution
        lam = np.random.beta(alpha, alpha)

        # Apply the mixup formula
        mixed_data = lam * data[i] + (1 - lam) * data[j]

        # Append the new mixed sample
        tsmixuped_data.append(mixed_data)

    return np.array(tsmixuped_data)


# Jittering

def jittering(data, noise_factor=0.01):
    """
    Adds random noise (jittering) to each feature/column in the given dataframe.

    :param data: The original multivariate time-series num features.
    :param noise_factor: Scale of the noise to add (default is 1% of the data's standard deviation).
    :return: A new dataframe with jittering applied to each column.
    """

    # Create a copy of the dataframe to hold the augmented data
    jittered_data = data.copy()

    # Apply jittering to each column of the dataframe
    noise = np.random.normal(loc=0, scale=noise_factor * np.std(data), size=data.shape)
    jittered_data = data + noise

    return jittered_data


# Permutation
import numpy as np


def permutation(data, n_permutations=1):
    """
    Perform permutation data augmentation on 3D multivariate data.

    Parameters:
    - data (np.ndarray): The original 3D dataset with shape (n_batches, n_samples, n_features).
    - n_permutations (int): Number of permuted datasets to generate.

    Returns:
    - augmented_data (list): A list of permuted datasets.
    """
    if len(data.shape) != 3:
        raise ValueError(f"Expected 3D input. Got {data.shape}")

    n_batches, n_samples, n_features = data.shape
    augmented_data = []

    # Generate n_permutations datasets
    for _ in range(n_permutations):
        # Initialize an empty array to store permuted data
        permuted_data = np.zeros_like(data)

        # Permute each batch independently
        for batch in range(n_batches):
            for feature in range(n_features):
                # Shuffle the time steps for each feature within each batch
                permuted_data[batch, :, feature] = np.random.permutation(data[batch, :, feature])

        augmented_data.append(permuted_data)

    return augmented_data
