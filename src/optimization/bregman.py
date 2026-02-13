"""
Bregman Divergence implementation for LMSR cost functions.

The Bregman divergence D(μ||θ) measures the "distance" between price vectors
in a way that respects the information-theoretic structure of prediction markets.

For LMSR (Logarithmic Market Scoring Rule):
- R(μ) = Σ μ_i ln(μ_i)  (negative entropy)
- D(μ||θ) = KL divergence = Σ μ_i ln(μ_i / p_i(θ))

Key insight from the research:
The maximum guaranteed profit from any arbitrage trade equals D(μ*||θ) - g(μ*),
where μ* is the Bregman projection onto the arbitrage-free manifold M.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from numpy.typing import NDArray

import structlog

logger = structlog.get_logger(__name__)


class BregmanDivergence(ABC):
    """
    Abstract base class for Bregman divergence.
    
    The Bregman divergence associated with a convex function R is:
    D_R(μ||θ) = R(μ) + R*(θ) - θ·μ
    
    Where R* is the convex conjugate of R.
    """
    
    @abstractmethod
    def divergence(self, mu: NDArray, theta: NDArray) -> float:
        """
        Compute the Bregman divergence D(μ||θ).
        
        Args:
            mu: Target price vector (what we're projecting to)
            theta: Current market state
            
        Returns:
            The divergence value
        """
        pass
    
    @abstractmethod
    def gradient(self, mu: NDArray) -> NDArray:
        """
        Compute the gradient ∇R(μ).
        
        Args:
            mu: Price vector
            
        Returns:
            Gradient vector
        """
        pass
    
    @abstractmethod
    def conjugate(self, theta: NDArray) -> float:
        """
        Compute the convex conjugate R*(θ).
        
        Args:
            theta: Market state
            
        Returns:
            Conjugate value
        """
        pass


class LMSRBregman(BregmanDivergence):
    """
    Bregman divergence for LMSR (Logarithmic Market Scoring Rule).
    
    For LMSR:
    - Cost function: C(θ) = b * log(Σ exp(θ_i/b))
    - Prices: p_i(θ) = exp(θ_i/b) / Σ exp(θ_j/b)
    - Conjugate: R(μ) = Σ μ_i ln(μ_i) (negative entropy)
    
    The Bregman divergence becomes KL divergence:
    D(μ||θ) = Σ μ_i ln(μ_i / p_i(θ))
    
    This measures how much the current market prices p(θ) differ from
    the target coherent prices μ.
    """
    
    def __init__(self, liquidity: float = 100.0, epsilon: float = 1e-10):
        """
        Initialize LMSR Bregman divergence.
        
        Args:
            liquidity: LMSR liquidity parameter b (higher = more liquidity)
            epsilon: Small constant to prevent log(0)
        """
        self.b = liquidity
        self.epsilon = epsilon
    
    def prices_from_state(self, theta: NDArray) -> NDArray:
        """
        Convert market state to prices using softmax.
        
        Args:
            theta: Market state vector
            
        Returns:
            Price vector (probabilities)
        """
        # Numerical stability: subtract max before exp
        theta_shifted = theta - np.max(theta)
        exp_theta = np.exp(theta_shifted / self.b)
        return exp_theta / np.sum(exp_theta)
    
    def state_from_prices(self, prices: NDArray) -> NDArray:
        """
        Convert prices to market state (inverse softmax).
        
        Args:
            prices: Price vector (must sum to 1)
            
        Returns:
            Market state vector
        """
        # θ_i = b * ln(p_i) + constant
        # The constant doesn't matter for LMSR
        prices = np.clip(prices, self.epsilon, 1 - self.epsilon)
        return self.b * np.log(prices)
    
    def divergence(self, mu: NDArray, theta: NDArray) -> float:
        """
        Compute KL divergence D(μ||θ) = Σ μ_i ln(μ_i / p_i(θ)).
        
        This is the maximum arbitrage profit available if we could
        move prices from p(θ) to μ perfectly.
        
        Args:
            mu: Target coherent price vector
            theta: Current market state
            
        Returns:
            KL divergence (non-negative)
        """
        p_theta = self.prices_from_state(theta)
        
        # Clip to prevent numerical issues
        mu_safe = np.clip(mu, self.epsilon, 1 - self.epsilon)
        p_safe = np.clip(p_theta, self.epsilon, 1 - self.epsilon)
        
        # KL divergence
        kl = np.sum(mu_safe * np.log(mu_safe / p_safe))
        
        return max(0.0, kl)  # Ensure non-negative
    
    def gradient(self, mu: NDArray) -> NDArray:
        """
        Compute gradient ∇R(μ) = ln(μ) + 1.
        
        WARNING: This explodes as μ → 0. Use Barrier Frank-Wolfe
        with contraction to handle this.
        
        Args:
            mu: Price vector
            
        Returns:
            Gradient vector
        """
        mu_safe = np.clip(mu, self.epsilon, 1 - self.epsilon)
        return np.log(mu_safe) + 1
    
    def conjugate(self, theta: NDArray) -> float:
        """
        Compute conjugate R*(θ) = C(θ) for LMSR.
        
        C(θ) = b * log(Σ exp(θ_i/b))
        
        Args:
            theta: Market state
            
        Returns:
            Cost function value
        """
        theta_shifted = theta - np.max(theta)  # Numerical stability
        return self.b * (np.max(theta) + np.log(np.sum(np.exp(theta_shifted / self.b))))
    
    def cost_function(self, theta: NDArray) -> float:
        """Alias for conjugate - the LMSR cost function."""
        return self.conjugate(theta)
    
    def hessian_diagonal(self, mu: NDArray) -> NDArray:
        """
        Compute diagonal of Hessian ∂²R/∂μ_i² = 1/μ_i.
        
        This shows why contraction is needed: as μ_i → 0,
        the Hessian explodes.
        
        Args:
            mu: Price vector
            
        Returns:
            Diagonal of Hessian
        """
        mu_safe = np.clip(mu, self.epsilon, 1 - self.epsilon)
        return 1.0 / mu_safe
    
    def lipschitz_constant(self, mu: NDArray) -> float:
        """
        Compute Lipschitz constant L = max_i |∂²R/∂μ_i²|.
        
        For contracted polytope with min coordinate ε:
        L ≤ 1/ε
        
        Args:
            mu: Price vector
            
        Returns:
            Lipschitz constant estimate
        """
        return np.max(self.hessian_diagonal(mu))
    
    def profit_guarantee(
        self,
        mu: NDArray,
        theta: NDArray,
        fw_gap: float,
    ) -> float:
        """
        Compute guaranteed profit from Proposition 4.1.
        
        Guaranteed profit ≥ D(μ||θ) - g(μ)
        
        Where:
        - D(μ||θ) is the maximum arbitrage if projection were perfect
        - g(μ) is the Frank-Wolfe gap (how suboptimal μ is)
        
        Args:
            mu: Current iterate (approximate projection)
            theta: Market state
            fw_gap: Frank-Wolfe gap g(μ)
            
        Returns:
            Guaranteed minimum profit
        """
        divergence = self.divergence(mu, theta)
        return divergence - fw_gap


class MultiMarketBregman(BregmanDivergence):
    """
    Bregman divergence for multiple independent markets.
    
    When markets are independent, the total divergence is the sum
    of individual market divergences.
    """
    
    def __init__(
        self,
        market_sizes: list[int],
        liquidity: float = 100.0,
        epsilon: float = 1e-10,
    ):
        """
        Initialize multi-market Bregman.
        
        Args:
            market_sizes: Number of outcomes in each market
            liquidity: LMSR liquidity parameter
            epsilon: Numerical stability constant
        """
        self.market_sizes = market_sizes
        self.n_markets = len(market_sizes)
        self.total_outcomes = sum(market_sizes)
        self.lmsr = LMSRBregman(liquidity, epsilon)
        
        # Compute index boundaries for each market
        self.market_indices = []
        start = 0
        for size in market_sizes:
            self.market_indices.append((start, start + size))
            start += size
    
    def _split_by_market(self, x: NDArray) -> list[NDArray]:
        """Split a vector into per-market components."""
        return [x[start:end] for start, end in self.market_indices]
    
    def divergence(self, mu: NDArray, theta: NDArray) -> float:
        """Compute sum of per-market divergences."""
        mu_parts = self._split_by_market(mu)
        theta_parts = self._split_by_market(theta)
        
        total = 0.0
        for mu_m, theta_m in zip(mu_parts, theta_parts):
            total += self.lmsr.divergence(mu_m, theta_m)
        
        return total
    
    def gradient(self, mu: NDArray) -> NDArray:
        """Compute gradient for all markets."""
        result = np.zeros_like(mu)
        
        for start, end in self.market_indices:
            result[start:end] = self.lmsr.gradient(mu[start:end])
        
        return result
    
    def conjugate(self, theta: NDArray) -> float:
        """Compute sum of per-market conjugates."""
        theta_parts = self._split_by_market(theta)
        return sum(self.lmsr.conjugate(t) for t in theta_parts)
