#written using zotgpt - claude sonnet 3.7
# src/simulate.py
import numpy as np
import time
from ez_diffusion import EZDiffusion

def run_simulation(sample_sizes=[10, 40, 4000], iterations=1000):
    """
    Run the simulate-and-recover exercise for the EZ diffusion model.

    Parameters:
    -----------
    sample_sizes : list
        List of sample sizes to test
    iterations : int
        Number of iterations for each sample size

    Returns:
    --------
    dict
        Dictionary containing summary of simulation results
    """
    # Validate inputs
    if any(N <= 0 for N in sample_sizes):
        raise ValueError("Sample sizes must be positive")
    if iterations <= 0:
        raise ValueError("Number of iterations must be positive")
    
    # Initialize EZ diffusion model
    ez = EZDiffusion()

    # Initialize results dictionary
    results = {N: {'v_bias': [], 'a_bias': [], 'T_bias': [], 
                   'v_squared_error': [], 'a_squared_error': [], 'T_squared_error': []} 
              for N in sample_sizes}

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
            
            # Randomly sample true parameters within specified ranges
            v_true = np.random.uniform(0.5, 2)
            a_true = np.random.uniform(0.5, 2)
            T_true = np.random.uniform(0.1, 0.5)

            try:
                # Step 2: Generate predicted summary statistics using forward equations
                Rpred, Mpred, Vpred = ez.forward(v_true, a_true, T_true)
                
                # Step 3: Simulate observed summary statistics using sampling distributions
                Robs, Mobs, Vobs = ez.simulate(Rpred, Mpred, Vpred, N)
                
                # Step 4: Compute estimated parameters using inverse equations
                v_est, a_est, T_est = ez.inverse(Robs, Mobs, Vobs)

                # Step 5: Compute bias and squared error
                v_bias = v_est - v_true
                a_bias = a_est - a_true
                T_bias = T_est - T_true

                # Store results
                results[N]['v_bias'].append(v_bias)
                results[N]['a_bias'].append(a_bias)
                results[N]['T_bias'].append(T_bias)
                results[N]['v_squared_error'].append(v_bias**2)
                results[N]['a_squared_error'].append(a_bias**2)
                results[N]['T_squared_error'].append(T_bias**2)

            except ValueError as e:
                print(f"Error in iteration {i} with N={N}: {e}")
                continue  # Skip this iteration if there's an error

    # End timing
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")

    # Calculate summary statistics
    summary = {}
    for N in sample_sizes:
        summary[N] = {
            'v_bias_mean': np.mean(results[N]['v_bias']),
            'a_bias_mean': np.mean(results[N]['a_bias']),
            'T_bias_mean': np.mean(results[N]['T_bias']),
            'v_bias_std': np.std(results[N]['v_bias']),
            'a_bias_std': np.std(results[N]['a_bias']),
            'T_bias_std': np.std(results[N]['T_bias']),
            'v_mse': np.mean(results[N]['v_squared_error']),
            'a_mse': np.mean(results[N]['a_squared_error']),
            'T_mse': np.mean(results[N]['T_squared_error'])
        }
        
        '''# Print summary statistics
        print(f"\nSummary for N = {N}:")
        print(f"  Drift rate (v) - Bias: {summary[N]['v_bias_mean']:.6f} ± {summary[N]['v_bias_std']:.6f}, MSE: {summary[N]['v_mse']:.6f}")
        print(f"  Boundary separation (a) - Bias: {summary[N]['a_bias_mean']:.6f} ± {summary[N]['a_bias_std']:.6f}, MSE: {summary[N]['a_mse']:.6f}")
        print(f"  Non-decision time (T) - Bias: {summary[N]['T_bias_mean']:.6f} ± {summary[N]['T_bias_std']:.6f}, MSE: {summary[N]['T_mse']:.6f}")'''

    return summary

if __name__ == "__main__":
    run_simulation()