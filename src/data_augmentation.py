import numpy as np
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


# Function to perform TSMixup
def ts_mixup(data, alpha=0.2):
    # Randomly select two samples from the dataset
    idx1, idx2 = np.random.choice(data.shape[0], 2, replace=False)

    # Generate a mixing coefficient
    lambda_ = np.random.beta(alpha, alpha)

    # Create the mixed sample
    mixed_data = lambda_ * data[idx1] + (1 - lambda_) * data[idx2]

    return mixed_data