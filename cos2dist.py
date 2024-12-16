import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Define the cos^2 PDF centered at 90 degrees
def cos2_pdf(x):
    # Shift x to center around 90 degrees
    x_rad = np.radians(x - 90)
    return (2/np.pi)*np.cos(x_rad)**2

# Define the CDF of the cos^2 distribution
def cos2_cdf(x, lower_bound=0, upper_bound=180):
    # Compute the integral of the PDF from lower_bound to x
    result, _ = quad(cos2_pdf, lower_bound, x)
    # Normalize by the total integral (for a proper PDF)
    normalization, _ = quad(cos2_pdf, lower_bound, upper_bound)
    return result / normalization

# Inverse CDF (Quantile Function) for cos^2
def cos2_quantile(p, lower_bound=0, upper_bound=180):
    """
    Find the x such that the CDF(x) = p using numerical methods.
    """
    from scipy.optimize import bisect

    # Root-finding function: CDF(x) - p = 0
    func = lambda x: cos2_cdf(x, lower_bound, upper_bound) - p

    # Use bisection method to find x in the valid range
    return bisect(func, lower_bound, upper_bound)

# Generate theoretical quantiles
def generate_theoretical_quantiles(n, lower_bound=0, upper_bound=180):
    probabilities = [(i + 1) / (n + 1) for i in range(n)]
    quantiles = [cos2_quantile(p, lower_bound, upper_bound) for p in probabilities]
    return quantiles

# Plot the Q-Q plot
def qqplot_cos2(observed_data, lower_bound=0, upper_bound=180):
    n = len(observed_data)
    # Sort observed data
    observed_sorted = np.sort(observed_data)

    # Generate theoretical quantiles
    theoretical_quantiles = generate_theoretical_quantiles(n, lower_bound, upper_bound)

    # Create the Q-Q plot
    plt.figure(figsize=(8, 6))
    plt.scatter(observed_sorted, theoretical_quantiles, color='blue', label='Q-Q points')
    plt.plot(observed_sorted, observed_sorted, color='red', linestyle='--', label='y=x (perfect fit)')
    plt.xlabel('Observed Data Quantiles')
    plt.ylabel('Theoretical Quantiles (cos^2 centered at 90°)')
    plt.title('Q-Q Plot for cos^2 Distribution (Centered at 90°)')
    plt.legend()
    plt.grid()
    plt.show()

# Example Usage
if __name__ == "__main__":
    # Example observed data (replace with your actual data)
    observed_data = [np.float64(82.01332281680506), np.float64(106.29799390171067), np.float64(93.99710602468342), np.float64(90.98754190248079), np.float64(19.69265694732536), np.float64(76.3189791087813), np.float64(87.83103401913503), np.float64(76.6551251002323), np.float64(90.0), np.float64(20.435311710601837), np.float64(74.72426898943515), np.float64(81.68357434172499), np.float64(71.36365494203423), np.float64(87.39822451341325)]
    # Generate the Q-Q plot
    qqplot_cos2(observed_data)

