import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_empirical_cdf_with_dkw(csv_path, delta=0.05):
    # Load accuracy data from CSV
    df = pd.read_csv(csv_path)
    if 'accuracy' not in df.columns:
        raise ValueError("CSV must contain a column named 'accuracy'")
    
    data = df['accuracy'].dropna().values
    n = len(data)

    if n == 0:
        raise ValueError("No accuracy values found in the CSV.")

    # Compute epsilon from DKW
    epsilon = np.sqrt(np.log(2 / delta) / (2 * n))

    # Sort data and compute empirical CDF
    x_emp = np.sort(data)
    cdf_emp = np.arange(1, n + 1) / n

    # Compute DKW bounds
    cdf_upper = np.clip(cdf_emp + epsilon, 0, 1)
    cdf_lower = np.clip(cdf_emp - epsilon, 0, 1)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.step(x_emp, cdf_emp, where='post', label=f'Empirical CDF $F_{{{n}}}$', color='black')
    plt.step(x_emp, cdf_upper, where='post', linestyle='--', label='Upper DKW Bound', color='red')
    plt.step(x_emp, cdf_lower, where='post', linestyle='--', label='Lower DKW Bound', color='blue')
    plt.fill_between(x_emp, cdf_lower, cdf_upper, step='post', color='gray', alpha=0.2, label='DKW 95% Band')
    
    plt.xlabel('Accuracy')
    plt.ylabel('CDF')
    plt.title('Empirical CDF with DKW Confidence Band (95%)')
    plt.legend()
    plt.savefig("dkw_plot.png", dpi=300, bbox_inches='tight')
    
    return x_emp, cdf_emp, epsilon

def get_accuracy_bounds_for_cdf(x_emp, cdf_emp, epsilon, target_cdf):
    # Given a CDF value, return the min and max accuracies where the true CDF could match that value
    lower_bound = target_cdf - epsilon
    upper_bound = target_cdf + epsilon

    accuracies_within_band = x_emp[(cdf_emp >= lower_bound) & (cdf_emp <= upper_bound)]

    if len(accuracies_within_band) == 0:
        return None, None
    return np.min(accuracies_within_band), np.max(accuracies_within_band)

def main():
    parser = argparse.ArgumentParser(description='Plot empirical CDF with DKW bounds and get accuracy interval for a given CDF value.')
    parser.add_argument('csv_path', type=str, help='Path to the CSV file containing an "accuracy" column.')
    parser.add_argument('target_cdf', type=float, help='Target CDF value (e.g., 0.6).')
    parser.add_argument('--delta', type=float, default=0.05, help='Confidence level (default: 0.05).')

    args = parser.parse_args()

    x_emp, cdf_emp, epsilon = plot_empirical_cdf_with_dkw(args.csv_path, args.delta)
    acc_min, acc_max = get_accuracy_bounds_for_cdf(x_emp, cdf_emp, epsilon, args.target_cdf)

    if acc_min is not None:
        print(f"Estimated accuracy range for CDF={args.target_cdf:.2f} with DKW bounds: {acc_min:.4f} to {acc_max:.4f}")
    else:
        print("No accuracy values found within the DKW bounds for the specified CDF.")

if __name__ == "__main__":
    main()
