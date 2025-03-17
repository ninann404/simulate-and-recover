#written using zotgpt - claude sonnet 3.7
# src/ez_diffusion.py
import numpy as np
from scipy import stats

class EZDiffusion: 
    """
    Implementation of the EZ Diffusion model with forward and inverse equations.
    """
    
    def __init__(self):
        """Initialize the EZ Diffusion model."""
        pass
    
    def forward(self, v, a, T):
        """
        Forward EZ equations: Calculate predicted summary statistics from parameters.
        
        Parameters:
        -----------
        v : float
            Drift rate
        a : float
            Boundary separation
        T : float
            Non-decision time
            
        Returns:
        --------
        tuple
            (Rpred, Mpred, Vpred) - predicted response proportion, mean RT, and variance of RT
        """
        # Check parameter bounds
        if not (0.5 <= a <= 2 and 0.5 <= v <= 2 and 0.1 <= T <= 0.5):
            raise ValueError("Parameters out of bounds: v should be in [0.5, 2], a in [0.5, 2], T in [0.1, 0.5]")
            
        # Calculate y
        y = np.exp(-a * v)
        
        # Equation 1: Response proportion
        Rpred = 1 / (y + 1)
        
        # Equation 2: Mean response time
        Mpred = T + (a / (2 * v)) * ((1 - y) / (1 + y))
        
        # Equation 3: Variance of response time
        Vpred = (a / (2 * v**3)) * ((1 - 2*a*v*y - y**2) / (y + 1)**2)
        
        return Rpred, Mpred, Vpred
    
    def inverse(self, Robs, Mobs, Vobs):
        """
        Inverse EZ equations: Calculate estimated parameters from observed statistics.
        
        Parameters:
        -----------
        Robs : float
            Observed response proportion
        Mobs : float
            Observed mean response time
        Vobs : float
            Observed variance of response time
            
        Returns:
        --------
        tuple
            (vest, aest, Test) - estimated drift rate, boundary separation, and non-decision time
        """
        # Check for valid input
        if not (0 < Robs < 1):
            raise ValueError("Robs must be between 0 and 1")
        if Mobs <= 0 or Vobs <= 0:
            raise ValueError("Mobs and Vobs must be positive")
            
        # Calculate L
        L = np.log(Robs / (1 - Robs))
        
        # Equation 4: Estimated drift rate
        sign = 1 if Robs > 0.5 else -1
        numerator = L * (Robs**2 * L - Robs * L + Robs - 0.5)
        vest = sign * np.power(numerator / Vobs, 0.25)
        
        # Equation 5: Estimated boundary separation
        aest = L / vest
        
        # Equation 6: Estimated non-decision time
        exp_term = np.exp(-vest * aest)
        numerator = 1 - exp_term
        denominator = 1 + exp_term
        Test = Mobs - (aest / (2 * vest)) * (numerator / denominator)
        
        return vest, aest, Test
    
    def simulate(self, Rpred, Mpred, Vpred, N):
        """
        Simulate observed summary statistics based on predicted statistics and sample size.
        Uses equations 7-9 from the EZ diffusion model.
        """
        # Equation 7: Simulate observed response proportion using Binomial
        n_correct = np.random.binomial(n=N, p=Rpred)
        
        # Handle edge cases to ensure Robs is strictly between 0 and 1
        if n_correct == 0:
            n_correct = 0.5  # Add a small correction
        elif n_correct == N:
            n_correct = N - 0.5  # Add a small correction
            
        Robs = n_correct / N
        
        # Equation 8: Simulate observed mean using Normal
        Mobs = np.random.normal(loc=Mpred, scale=np.sqrt(Vpred/N))
        
        # Equation 9: Simulate observed variance using Gamma
        shape = (N - 1) / 2
        scale = 2 * Vpred / (N - 1)
        Vobs = np.random.gamma(shape=shape, scale=scale)
        
        return Robs, Mobs, Vobs
    