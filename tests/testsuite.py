# test/test_ez.py
import unittest
import sys
import os
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ez_diffusion import EZDiffusion

class TestEZDiffusion(unittest.TestCase):
    """Test cases for the EZ Diffusion model."""
    
    def setUp(self):
        """Set up the EZ Diffusion model for testing."""
        self.ez = EZDiffusion()
        
    def test_forward_equations(self):
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
        
    def test_inverse_equations(self):
        """Test that inverse equations produce expected values."""
        # Test with known values
        Robs, Mobs, Vobs = 0.4, 0.8, 0.2
        vest, aest, Test = self.ez.inverse(Robs, Mobs, Vobs)
        
        # Check that parameters are in expected ranges
        self.assertTrue(vest != 0)  # Drift rate should not be zero
        self.assertTrue(aest > 0)   # Boundary separation should be positive
        self.assertTrue(Test > 0)   # Non-decision time should be positive
        
    def test_parameter_recovery_no_noise(self):
        """Test that parameters can be recovered exactly when there is no noise."""
        # Generate true parameters
        v_true = 1.0
        a_true = 1.0
        T_true = 0.3
        
        # Generate predicted summary statistics
        Rpred, Mpred, Vpred = self.ez.forward(v_true, a_true, T_true)
        
        # Recover parameters without adding noise (use predicted as observed)
        v_est, a_est, T_est = self.ez.inverse(Rpred, Mpred, Vpred)