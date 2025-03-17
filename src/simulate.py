import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ez_diffusion import EZDiffusion
import os
import time

def run_simulation(sample_sizes=[10, 40, 4000], iterations=1000, output_dir="results"):
    """
    Run the simulate-and-recover exercise for the EZ diffusion model.

    Parameters:
    -----------
    sample_sizes : list
        List of sample sizes to test
    iterations : int
        Number of iterations for each sample size
    output_dir : str
        Directory to save results

    Returns:
    --------
    dict
        Dictionary containing simulation results
    """
    # Create output directory structure
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    
    # Initialize EZ diffusion model
    ez = EZDiffusion()

    # Initialize results dictionary
    results = {
        'N': [], 'iteration': [],
        'v_true': [], 'a_true': [], 'T_true': [],
        'v_est': [], 'a_est': [], 'T_est': [],
        'v_bias': [], 'a_bias': [], 'T_bias': [],
        'v_squared_error': [], 'a_squared_error': [], 'T_squared_error': []
    }

    # Set random seed for reproducibility
    np.random.seed(42)

    # Start timing
    start_time = time.time()

    # Run simulations for each sample size
    for N in sample_sizes:
        print(f"Running simulations for N = {N}")

        for i in range(iterations):
            if i % 100 == 0:
                print(f"  Iteration {i}/{iterations}")
            
            # Randomly sample true parameters
            v_true = np.random.uniform(0.5, 2)
            a_true = np.random.uniform(0.5, 2)
            T_true = np.random.uniform(0.1, 0.5)

            try:
                # Generate predicted summary statistics
                Rpred, Mpred, Vpred = ez.forward(v_true, a_true, T_true)
                
                # Simulate observed summary statistics
                Robs, Mobs, Vobs = ez.simulate(Rpred, Mpred, Vpred, N)
                
                # Compute estimated parameters
                v_est, a_est, T_est = ez.inverse(Robs, Mobs, Vobs)

                # Compute bias and squared error
                v_bias = v_est - v_true
                a_bias = a_est - a_true
                T_bias = T_est - T_true

                results['N'].append(N)
                results['iteration'].append(i)
                results['v_true'].append(v_true)
                results['a_true'].append(a_true)
                results['T_true'].append(T_true)
                results['v_est'].append(v_est)
                results['a_est'].append(a_est)
                results['T_est'].append(T_est)
                results['v_bias'].append(v_bias)
                results['a_bias'].append(a_bias)
                results['T_bias'].append(T_bias)
                results['v_squared_error'].append(v_bias**2)
                results['a_squared_error'].append(a_bias**2)
                results['T_squared_error'].append(T_bias**2)

            except ValueError as e:
                print(f"Error in iteration {i} with N={N}: {e}")
                continue  # Skip this iteration if there’s an error

    # End timing
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")

    # Convert results to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "simulation_results.csv"), index=False)

    # Generate and save summary statistics
    summary = df.groupby('N').agg({
        'v_bias': ['mean', 'std'],
        'a_bias': ['mean', 'std'],
        'T_bias': ['mean', 'std'],
        'v_squared_error': ['mean'],
        'a_squared_error': ['mean'],
        'T_squared_error': ['mean']
    })
    summary.to_csv(os.path.join(output_dir, "summary_results.csv"))

    # Generate visualizations
    create_plots(df, output_dir)

    return df

def create_plots(df, output_dir):
    """
    Create visualizations for the simulation results.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing simulation results
    output_dir : str
        Directory to save plots
    """
    params = ['v', 'a', 'T']
    sample_sizes = sorted(df['N'].unique())

    # Plot bias distributions
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for i, param in enumerate(params):
        for j, N in enumerate(sample_sizes):
            subset = df[df['N'] == N]
            axes[i, j].hist(subset[f'{param}_bias'], bins=30, alpha=0.7)
            axes[i, j].axvline(x=0, color='red', linestyle='--', label='Zero Bias')
            axes[i, j].legend()
            axes[i, j].set_title(f'{param.upper()} Bias (N={N})')
            axes[i, j].set_xlabel('Bias')
            axes[i, j].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figures", "bias_distributions.png"))

    # Plot MSE vs. Sample Size
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, param in enumerate(params):
        mean_mse = df.groupby('N')[f'{param}_squared_error'].mean()
        axes[i].plot(mean_mse.index, mean_mse.values, 'o-', label='MSE')
        axes[i].set_xscale('log')
        axes[i].set_yscale('log')
        axes[i].set_title(f'{param.upper()} Mean Squared Error')
        axes[i].set_xlabel('Sample Size (N) [log scale]')
        axes[i].set_ylabel('MSE [log scale]')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figures", "mse_vs_sample_size.png"))

    # Plot true vs. estimated parameters
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for i, param in enumerate(params):
        for j, N in enumerate(sample_sizes):
            subset = df[df['N'] == N]
            axes[i, j].scatter(subset[f'{param}_true'], subset[f'{param}_est'], alpha=0.3)
            
            # Add identity line with legend
            min_val = min(subset[f'{param}_true'].min(), subset[f'{param}_est'].min())
            max_val = max(subset[f'{param}_true'].max(), subset[f'{param}_est'].max())
            axes[i, j].plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
            axes[i, j].legend()

            axes[i, j].set_title(f'{param.upper()} True vs. Estimated (N={N})')
            axes[i, j].set_xlabel('True Value')
            axes[i, j].set_ylabel('Estimated Value')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figures", "true_vs_estimated.png"))

if __name__ == "__main__":
    run_simulation()
