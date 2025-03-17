#written using zotgpt - claude sonnet 3.7
# test/testsuite.py
import unittest
import sys
import os
import numpy as np
import time
import tempfile
import shutil
from unittest.mock import patch

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ez_diffusion import EZDiffusion
import simulate

class TestEZDiffusion(unittest.TestCase):
    """Unit tests for the EZ Diffusion model."""
    
    def setUp(self):
        """Set up the EZ Diffusion model for testing."""
        self.ez = EZDiffusion()
        print(f"\nRunning: {self._testMethodName}")
        
    def test_forward_equations_basic(self):
        """Test that forward equations produce expected values."""
        # Test with known values
        v, a, T = 1.0, 1.0, 0.3
        Rpred, Mpred, Vpred = self.ez.forward(v, a, T)
        
        # Check that Rpred is between 0 and 1
        self.assertTrue(0 < Rpred < 1)
        
        # Check that Mpred is greater than T
        self.assertTrue(Mpred > T)
        
        # Check that Vpred is positive
        self.assertTrue(Vpred > 0)
        print("✓ Forward equations produce valid outputs")
        
    def test_forward_equations_multiple_cases(self):
        """Test forward equations with multiple parameter sets."""
        test_cases = [
            (0.5, 0.5, 0.1),  # Lower bounds
            (2.0, 2.0, 0.5),  # Upper bounds
            (1.2, 0.7, 0.3),  # Mixed values
        ]
        
        for i, (v, a, T) in enumerate(test_cases):
            Rpred, Mpred, Vpred = self.ez.forward(v, a, T)
            
            # Basic sanity checks
            self.assertTrue(0 < Rpred < 1)
            self.assertTrue(Mpred > T)
            self.assertTrue(Vpred > 0)
            print(f"✓ Case {i+1}: Forward equations valid for v={v}, a={a}, T={T}")
            
            # Additional check: as drift rate increases, mean RT should decrease
            if v > 1.0:
                _, Mpred_baseline, _ = self.ez.forward(1.0, a, T)
                self.assertTrue(Mpred < Mpred_baseline)
                print(f"✓ Case {i+1}: Mean RT decreases as drift rate increases")
    
    def test_forward_equations_out_of_bounds(self):
        """Test that forward equations raise ValueError for out-of-bounds parameters."""
        out_of_bounds_cases = [
            (0.4, 1.0, 0.3),  # v below lower bound
            (2.1, 1.0, 0.3),  # v above upper bound
            (1.0, 0.4, 0.3),  # a below lower bound
            (1.0, 2.1, 0.3),  # a above upper bound
            (1.0, 1.0, 0.05), # T below lower bound
            (1.0, 1.0, 0.6),  # T above upper bound
        ]
        
        for i, (v, a, T) in enumerate(out_of_bounds_cases):
            with self.assertRaises(ValueError):
                self.ez.forward(v, a, T)
            print(f"✓ Case {i+1}: Correctly raised ValueError for out-of-bounds parameters v={v}, a={a}, T={T}")
        
    def test_inverse_equations_basic(self):
        """Test that inverse equations produce expected values."""
        # Test with known values
        Robs, Mobs, Vobs = 0.6, 0.8, 0.2
        vest, aest, Test = self.ez.inverse(Robs, Mobs, Vobs)
        
        # Check that parameters are in expected ranges
        self.assertTrue(vest != 0)  # Drift rate should not be zero
        self.assertTrue(aest > 0)   # Boundary separation should be positive
        self.assertTrue(Test > 0)   # Non-decision time should be positive
        print(f"✓ Inverse equations produce valid outputs: v={vest:.4f}, a={aest:.4f}, T={Test:.4f}")
    
    def test_inverse_equations_multiple_cases(self):
        """Test inverse equations with multiple statistic sets."""
        test_cases = [
            (0.2, 0.5, 0.1),  # Low accuracy, fast responses
            (0.8, 1.5, 0.5),  # High accuracy, slow responses
            (0.51, 1.0, 0.3),  # Medium values
        ]
        
        for i, (Robs, Mobs, Vobs) in enumerate(test_cases):
            vest, aest, Test = self.ez.inverse(Robs, Mobs, Vobs)
            
            # Basic sanity checks
            self.assertTrue(vest != 0)
            self.assertTrue(aest > 0)
            self.assertTrue(Test > 0)
            print(f"✓ Case {i+1}: Inverse equations valid for R={Robs}, M={Mobs}, V={Vobs}")
            
            # Additional check: higher accuracy should lead to higher drift rate
            if Robs > 0.5:
                self.assertTrue(vest > 0)
                print(f"✓ Case {i+1}: High accuracy (R={Robs}) gives positive drift rate (v={vest:.4f})")
            else:
                self.assertTrue(vest < 0)
                print(f"✓ Case {i+1}: Low accuracy (R={Robs}) gives negative drift rate (v={vest:.4f})")
    
    def test_inverse_equations_invalid_input(self):
        """Test that inverse equations raise ValueError for invalid inputs."""
        invalid_cases = [
            (0.0, 0.8, 0.2),  # Robs = 0
            (1.0, 0.8, 0.2),  # Robs = 1
            (0.4, 0.0, 0.2),  # Mobs = 0
            (0.4, -0.1, 0.2), # Mobs < 0
            (0.4, 0.8, 0.0),  # Vobs = 0
            (0.4, 0.8, -0.1), # Vobs < 0
        ]
        
        for i, (Robs, Mobs, Vobs) in enumerate(invalid_cases):
            with self.assertRaises(ValueError):
                self.ez.inverse(Robs, Mobs, Vobs)
            print(f"✓ Case {i+1}: Correctly raised ValueError for invalid inputs R={Robs}, M={Mobs}, V={Vobs}")
        
    def test_parameter_recovery_no_noise(self):
        """Test that parameters can be recovered exactly when there is no noise, bias should be 0."""
        # Test with multiple parameter sets
        test_cases = [
            (1.0, 1.0, 0.3),
            (0.8, 1.5, 0.2),
            (1.5, 0.7, 0.4),
        ]
        
        for i, (v_true, a_true, T_true) in enumerate(test_cases):
            # Generate predicted summary statistics
            Rpred, Mpred, Vpred = self.ez.forward(v_true, a_true, T_true)
            
            # Recover parameters without adding noise (use predicted as observed)
            v_est, a_est, T_est = self.ez.inverse(Rpred, Mpred, Vpred)
            
            # Parameters should be recovered with high precision
            self.assertAlmostEqual(v_true, v_est, places=4)
            self.assertAlmostEqual(a_true, a_est, places=4)
            self.assertAlmostEqual(T_true, T_est, places=4)
            print(f"✓ Case {i+1}: Parameters recovered with no noise: v={v_est:.4f}, a={a_est:.4f}, T={T_est:.4f}")
    
    def test_simulate_method(self):
        """Test that the simulate method produces valid outputs."""
        # Test with known values
        Rpred, Mpred, Vpred = 0.7, 0.8, 0.2
        N_values = [10, 40, 4000]
        
        for i, N in enumerate(N_values):
            Robs, Mobs, Vobs = self.ez.simulate(Rpred, Mpred, Vpred, N)
            
            # Check that outputs are in valid ranges
            self.assertTrue(0 < Robs < 1)
            self.assertTrue(Mobs > 0)
            self.assertTrue(Vobs > 0)
            print(f"✓ Simulation valid for N={N}: R={Robs:.4f}, M={Mobs:.4f}, V={Vobs:.4f}")
            
            # With large N, observed values should be close to predicted values
            if N == 4000:
                self.assertAlmostEqual(Rpred, Robs, places=1)
                self.assertAlmostEqual(Mpred, Mobs, places=1)
                self.assertAlmostEqual(Vpred, Vobs, places=1)
                print(f"✓ Large N={N}: Observed values close to predicted values")
    
    def test_simulate_edge_cases(self):
        """Test that the simulate method handles edge cases correctly."""
        # Test with extreme Rpred values
        edge_cases = [
            (0.01, 0.8, 0.2, 10),  # Very low Rpred
            (0.99, 0.8, 0.2, 10),  # Very high Rpred
        ]
        
        for i, (Rpred, Mpred, Vpred, N) in enumerate(edge_cases):
            Robs, Mobs, Vobs = self.ez.simulate(Rpred, Mpred, Vpred, N)
            
            # Check that outputs are in valid ranges
            self.assertTrue(0 < Robs < 1)
            self.assertTrue(Mobs > 0)
            self.assertTrue(Vobs > 0)
            print(f"✓ Edge case {i+1}: Simulation handles extreme Rpred={Rpred}")

    def test_corruption_invalid_parameters(self):
        """Test handling of corrupted or invalid parameters."""
        # Test with invalid parameter types
        with self.assertRaises(Exception):
            self.ez.forward("invalid", 1.0, 0.3)
        print("✓ Correctly handled invalid parameter type for v")
        
        with self.assertRaises(Exception):
            self.ez.forward(1.0, None, 0.3)
        print("✓ Correctly handled invalid parameter type for a")
        
        with self.assertRaises(Exception):
            self.ez.forward(1.0, 1.0, [0.3])
        print("✓ Correctly handled invalid parameter type for T")


class TestSimulation(unittest.TestCase):
    """Tests for the simulate module."""
    
    def setUp(self):
        """Set up for simulation tests."""
        print(f"\nRunning: {self._testMethodName}")
    
    def test_run_simulation_small_scale(self):
        """Test that run_simulation works with small iterations."""
        # Run a small-scale simulation to test functionality
        print("Running small-scale simulation...")
        summary = simulate.run_simulation(
            sample_sizes=[10], 
            iterations=5
        )
        
        # Check that the summary has the expected structure
        self.assertIn(10, summary)
        expected_keys = [
            'v_bias_mean', 'a_bias_mean', 'T_bias_mean',
            'v_bias_std', 'a_bias_std', 'T_bias_std',
            'v_mse', 'a_mse', 'T_mse'
        ]
        
        for key in expected_keys:
            self.assertIn(key, summary[10])
            print(f"✓ Summary contains key: {key}")
    
    @patch('numpy.random.uniform')
    def test_run_simulation_reproducibility(self, mock_uniform):
        """Test that run_simulation produces reproducible results with fixed seed."""
        # Mock the random parameter generation to always return the same values
        mock_uniform.side_effect = [1.0, 1.0, 0.3]  # v, a, T
        
        # Run a minimal simulation
        print("Running first simulation...")
        np.random.seed(42)
        summary1 = simulate.run_simulation(
            sample_sizes=[10], 
            iterations=1
        )
        
        # Reset the mock and run again
        mock_uniform.side_effect = [1.0, 1.0, 0.3]  # v, a, T
        print("Running second simulation...")
        np.random.seed(42)
        summary2 = simulate.run_simulation(
            sample_sizes=[10], 
            iterations=1
        )
        
        # With the same seed and parameters, the results should be identical
        self.assertEqual(summary1[10]['v_bias_mean'], summary2[10]['v_bias_mean'])
        self.assertEqual(summary1[10]['a_bias_mean'], summary2[10]['a_bias_mean'])
        self.assertEqual(summary1[10]['T_bias_mean'], summary2[10]['T_bias_mean'])
        print("✓ Simulations with same seed produce identical results")

    def test_corruption_invalid_inputs(self):
        """Test handling of corrupted or invalid inputs to run_simulation."""
        # Test with invalid sample sizes
        with self.assertRaises(Exception):
            simulate.run_simulation(
                sample_sizes=[-10], 
                iterations=5
            )
        print("✓ Correctly handled negative sample size")
        
        # Test with invalid iterations
        with self.assertRaises(Exception):
            simulate.run_simulation(
                sample_sizes=[10], 
                iterations=0
            )
        print("✓ Correctly handled zero iterations")


class TestIntegration(unittest.TestCase):
    """Integration tests for the EZ diffusion model and simulation."""
    
    def setUp(self):
        """Set up for integration tests."""
        self.ez = EZDiffusion()
        print(f"\nRunning: {self._testMethodName}")
        
    def test_full_pipeline_single_iteration(self):
            """Test the full simulate-and-recover pipeline for a single iteration."""
            # 1. Select parameters
            v_true = 1.0
            a_true = 1.0
            T_true = 0.3
            N = 1000  # Large N for stability
            
            print(f"Testing full pipeline with parameters: v={v_true}, a={a_true}, T={T_true}")
            
            # 2. Generate predicted summary statistics
            Rpred, Mpred, Vpred = self.ez.forward(v_true, a_true, T_true)
            print(f"Generated predicted stats: R={Rpred:.4f}, M={Mpred:.4f}, V={Vpred:.4f}")
            
            # 3. Simulate observed summary statistics
            Robs, Mobs, Vobs = self.ez.simulate(Rpred, Mpred, Vpred, N)
            print(f"Simulated observed stats: R={Robs:.4f}, M={Mobs:.4f}, V={Vobs:.4f}")
            
            # 4. Recover parameters
            v_est, a_est, T_est = self.ez.inverse(Robs, Mobs, Vobs)
            print(f"Recovered parameters: v={v_est:.4f}, a={a_est:.4f}, T={T_est:.4f}")
            
            # 5. Check that recovered parameters are close to true parameters
            # With large N, they should be quite close
            self.assertAlmostEqual(v_true, v_est, places=1)
            self.assertAlmostEqual(a_true, a_est, places=1)
            self.assertAlmostEqual(T_true, T_est, places=1)
            print("✓ Recovered parameters are close to true parameters")
    
    def test_squared_error_decreases_with_n(self):
        """Test that squared error decreases as N increases."""
        # Run a small simulation with different N values
        summary = simulate.run_simulation(
            sample_sizes=[10, 100, 1000], 
            iterations=10
        )
        
        # Check that MSE decreases as N increases
        try:
            print(f"MSE for v: N=10: {summary[10]['v_mse']:.6f}, N=1000: {summary[1000]['v_mse']:.6f}")
            print(f"MSE for a: N=10: {summary[10]['a_mse']:.6f}, N=1000: {summary[1000]['a_mse']:.6f}")
            print(f"MSE for T: N=10: {summary[10]['T_mse']:.6f}, N=1000: {summary[1000]['T_mse']:.6f}")
            
            # Test that MSE decreases with larger N
            # Only check parameters that have valid values for both sample sizes
            if not np.isnan(summary[10]['v_mse']) and not np.isnan(summary[1000]['v_mse']):
                self.assertTrue(summary[10]['v_mse'] > summary[1000]['v_mse'])
                print("✓ MSE for v decreases as sample size increases")
                
            if not np.isnan(summary[10]['a_mse']) and not np.isnan(summary[1000]['a_mse']):
                self.assertTrue(summary[10]['a_mse'] > summary[1000]['a_mse'])
                print("✓ MSE for a decreases as sample size increases")
                
            if not np.isnan(summary[10]['T_mse']) and not np.isnan(summary[1000]['T_mse']):
                self.assertTrue(summary[10]['T_mse'] > summary[1000]['T_mse'])
                print("✓ MSE for T decreases as sample size increases")
        except Exception as e:
            self.fail(f"Failed to calculate MSE comparison: {e}")
            
    def test_bias_averages_to_zero(self):
        """explicitly test if bias averages to 0."""
        print("\nTesting if parameter recovery bias averages to zero...")
        
        # Run a simulation with moderate sample size and many iterations
        # for statistical reliability
        iterations = 100
        N = 1000
        
        # Initialize arrays to store bias values
        v_biases = []
        a_biases = []
        T_biases = []
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Run multiple iterations
        for i in range(iterations):
            if i % 20 == 0:
                print(f"  Iteration {i}/{iterations}")
            
            # 1. Select true parameters within valid ranges
            v_true = np.random.uniform(0.5, 2)
            a_true = np.random.uniform(0.5, 2)
            T_true = np.random.uniform(0.1, 0.5)
            
            # 2. Generate predicted summary statistics
            Rpred, Mpred, Vpred = self.ez.forward(v_true, a_true, T_true)
            
            # 3. Simulate observed summary statistics
            Robs, Mobs, Vobs = self.ez.simulate(Rpred, Mpred, Vpred, N)
            
            # 4. Recover parameters
            v_est, a_est, T_est = self.ez.inverse(Robs, Mobs, Vobs)
            
            # 5. Calculate bias
            v_bias = v_est - v_true
            a_bias = a_est - a_true
            T_bias = T_est - T_true
            
            # Store biases
            v_biases.append(v_bias)
            a_biases.append(a_bias)
            T_biases.append(T_bias)
        
        # Calculate mean biases
        mean_v_bias = np.mean(v_biases)
        mean_a_bias = np.mean(a_biases)
        mean_T_bias = np.mean(T_biases)
        
        # Calculate standard errors
        se_v_bias = np.std(v_biases) / np.sqrt(iterations)
        se_a_bias = np.std(a_biases) / np.sqrt(iterations)
        se_T_bias = np.std(T_biases) / np.sqrt(iterations)
        
        print(f"Mean bias for v: {mean_v_bias:.6f} ± {se_v_bias:.6f}")
        print(f"Mean bias for a: {mean_a_bias:.6f} ± {se_a_bias:.6f}")
        print(f"Mean bias for T: {mean_T_bias:.6f} ± {se_T_bias:.6f}")
        
        # Test that biases are not significantly different from zero
        # Using a 3-sigma rule (99.7% confidence)
        self.assertTrue(abs(mean_v_bias) < 3 * se_v_bias)
        self.assertTrue(abs(mean_a_bias) < 3 * se_a_bias)
        self.assertTrue(abs(mean_T_bias) < 3 * se_T_bias)
        
        print("✓ All parameter biases are not significantly different from zero")


def print_test_summary(result, start_time):
    """Print a summary of the test results."""
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"SUMMARY: Ran {result.testsRun} tests in {elapsed_time:.3f}s")
    
    if result.wasSuccessful():
        print("SUCCESS: All tests passed!")
    else:
        print(f"FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
        
        if result.failures:
            print("\nFAILURES:")
            for i, (test, traceback) in enumerate(result.failures, 1):
                print(f"{i}. {test}")
                print("-" * 70)
                print(traceback)
                print("-" * 70)
                
        if result.errors:
            print("\nERRORS:")
            for i, (test, traceback) in enumerate(result.errors, 1):
                print(f"{i}. {test}")
                print("-" * 70)
                print(traceback)
                print("-" * 70)
    
    print("="*70)


if __name__ == "__main__":
    # Start timing
    start_time = time.time()
    
    # Create a test suite with all tests
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    # Run the tests and capture the result
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    
    # Print a summary of the test results
    print_test_summary(result, start_time)
    
    # Exit with appropriate status code
    sys.exit(not result.wasSuccessful())
              
              