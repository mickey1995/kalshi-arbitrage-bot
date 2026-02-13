"""
Frank-Wolfe Algorithm with Barrier variant for Bregman Projection.

This implements the core optimization from the research paper:
1. Standard Frank-Wolfe for convex optimization over polytopes
2. Barrier Frank-Wolfe with adaptive contraction for LMSR

Key insight: Instead of projecting onto M directly (exponentially hard),
Frank-Wolfe iteratively finds descent vertices by solving INTEGER PROGRAMS.
This reduces exponential enumeration to polynomial optimization.

Algorithm 2 from Kroer et al.:
    For t = 0, 1, 2, ...
        1. Solve convex subproblem over conv(Z_t)
        2. Find descent vertex: z_t = argmin_{z∈Z} ∇F(μ_t)·z  [IP SOLVE]
        3. Update active set: Z_{t+1} = Z_t ∪ {z_t}
        4. Compute gap: g(μ_t) = ∇F(μ_t)·(μ_t - z_t)
        5. If g(μ_t) ≤ ε, STOP
        6. Update ε adaptively (Barrier FW)
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import time
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

import structlog

from .bregman import BregmanDivergence, LMSRBregman
from .ip_solver import IPSolver, IPResult
from .marginal_polytope import MarginalPolytope

logger = structlog.get_logger(__name__)


@dataclass
class FrankWolfeResult:
    """Result from Frank-Wolfe optimization."""
    
    # Solution
    mu: NDArray  # Projected price vector
    theta: NDArray  # Original market state
    
    # Convergence info
    iterations: int
    gap: float  # Final Frank-Wolfe gap g(μ)
    divergence: float  # D(μ||θ)
    
    # Profit guarantee (Proposition 4.1)
    guaranteed_profit: float  # D(μ||θ) - g(μ)
    
    # Timing
    total_time: float
    ip_solve_time: float
    
    # Active set
    active_vertices: int
    
    # Status
    status: str  # "converged", "alpha_extracted", "time_limit", etc.
    
    @property
    def extraction_ratio(self) -> float:
        """Fraction of maximum profit captured: 1 - g(μ)/D(μ||θ)."""
        if self.divergence < 1e-10:
            return 1.0
        return 1.0 - self.gap / self.divergence
    
    @property
    def is_profitable(self) -> bool:
        """Check if guaranteed profit is positive."""
        return self.guaranteed_profit > 0


@dataclass
class InitFWResult:
    """Result from InitFW initialization."""
    
    active_set: List[NDArray]  # Initial vertices Z_0
    interior_point: NDArray  # Interior point u
    extended_partial: Dict[int, int]  # Extended settled variables
    
    @property
    def n_vertices(self) -> int:
        return len(self.active_set)


class FrankWolfe:
    """
    Standard Frank-Wolfe algorithm for Bregman projection.
    
    WARNING: Standard FW fails on LMSR due to gradient explosion.
    Use BarrierFrankWolfe for production.
    """
    
    def __init__(
        self,
        bregman: BregmanDivergence,
        ip_solver: IPSolver,
        polytope: MarginalPolytope,
        alpha: float = 0.9,
        convergence_threshold: float = 1e-6,
        max_iterations: int = 150,
        time_limit: float = 1800.0,
    ):
        """
        Initialize Frank-Wolfe solver.
        
        Args:
            bregman: Bregman divergence (typically LMSRBregman)
            ip_solver: Integer programming solver for LMO
            polytope: Marginal polytope defining valid outcomes
            alpha: Extraction ratio for stopping (stop when capturing α of profit)
            convergence_threshold: Gap threshold ε_D for stopping
            max_iterations: Maximum iterations
            time_limit: Time limit in seconds
        """
        self.bregman = bregman
        self.ip_solver = ip_solver
        self.polytope = polytope
        
        self.alpha = alpha
        self.epsilon_D = convergence_threshold
        self.max_iterations = max_iterations
        self.time_limit = time_limit
        
        # Get constraint matrices
        self.A, self.b = polytope.get_constraint_matrices()
    
    def _lmo(self, gradient: NDArray) -> Tuple[NDArray, float]:
        """
        Linear Minimization Oracle: find vertex minimizing gradient·z.
        
        This is where the IP solver is called.
        
        Args:
            gradient: Current gradient ∇F(μ)
            
        Returns:
            (vertex z, objective value)
        """
        result = self.ip_solver.solve(
            objective=gradient,
            constraints_A=self.A,
            constraints_b=self.b,
            time_limit=30.0,
        )
        
        if not result.is_feasible:
            logger.warning("lmo_infeasible", status=result.status)
            return None, float('inf')
        
        return result.solution, result.objective
    
    def _compute_gap(self, mu: NDArray, gradient: NDArray, vertex: NDArray) -> float:
        """
        Compute Frank-Wolfe gap: g(μ) = ∇F(μ)·(μ - z).
        
        The gap measures how suboptimal μ is. When g(μ) = 0, μ is optimal.
        """
        return np.dot(gradient, mu - vertex)
    
    def _line_search(
        self,
        mu: NDArray,
        vertex: NDArray,
        theta: NDArray,
    ) -> float:
        """
        Find optimal step size γ ∈ [0, 1].
        
        We want γ that minimizes F(μ + γ(z - μ)) = D(μ + γ(z - μ) || θ).
        """
        direction = vertex - mu
        
        def objective(gamma):
            mu_new = mu + gamma * direction
            mu_new = np.clip(mu_new, 1e-10, 1 - 1e-10)
            return self.bregman.divergence(mu_new, theta)
        
        # Golden section search
        result = minimize(objective, x0=0.5, bounds=[(0, 1)], method='L-BFGS-B')
        return result.x[0]
    
    def run(
        self,
        theta: NDArray,
        initial_mu: Optional[NDArray] = None,
    ) -> FrankWolfeResult:
        """
        Run Frank-Wolfe optimization.
        
        Args:
            theta: Current market state
            initial_mu: Starting point (if None, uses softmax of theta)
            
        Returns:
            FrankWolfeResult with optimal μ and profit guarantee
        """
        start_time = time.time()
        ip_time = 0.0
        
        n = len(theta)
        
        # Initialize
        if initial_mu is None:
            # Use current market prices as starting point
            if isinstance(self.bregman, LMSRBregman):
                mu = self.bregman.prices_from_state(theta)
            else:
                mu = np.ones(n) / n  # Uniform
        else:
            mu = initial_mu.copy()
        
        # Ensure valid starting point
        mu = self.polytope.project_to_simplex(mu)
        
        active_set = []
        best_mu = mu.copy()
        best_profit = float('-inf')
        
        for iteration in range(self.max_iterations):
            # Check time limit
            elapsed = time.time() - start_time
            if elapsed > self.time_limit:
                logger.info("fw_time_limit", iteration=iteration)
                break
            
            # Compute gradient
            gradient = self.bregman.gradient(mu)
            
            # LMO: find descent vertex
            ip_start = time.time()
            vertex, _ = self._lmo(gradient)
            ip_time += time.time() - ip_start
            
            if vertex is None:
                logger.warning("fw_lmo_failed", iteration=iteration)
                break
            
            active_set.append(vertex)
            
            # Compute gap
            gap = self._compute_gap(mu, gradient, vertex)
            divergence = self.bregman.divergence(mu, theta)
            profit = divergence - gap
            
            # Track best iterate
            if profit > best_profit:
                best_profit = profit
                best_mu = mu.copy()
            
            # Check stopping conditions
            if divergence < self.epsilon_D:
                logger.info("fw_near_arbitrage_free", divergence=divergence)
                return self._make_result(
                    best_mu, theta, iteration + 1, gap, divergence,
                    time.time() - start_time, ip_time, len(active_set),
                    "near_arbitrage_free"
                )
            
            if gap <= (1 - self.alpha) * divergence:
                logger.info(
                    "fw_alpha_extracted",
                    alpha=self.alpha,
                    extraction=1 - gap/divergence,
                )
                return self._make_result(
                    best_mu, theta, iteration + 1, gap, divergence,
                    time.time() - start_time, ip_time, len(active_set),
                    "alpha_extracted"
                )
            
            # Step
            gamma = self._line_search(mu, vertex, theta)
            mu = mu + gamma * (vertex - mu)
            mu = np.clip(mu, 1e-10, 1 - 1e-10)
            
            if iteration % 10 == 0:
                logger.debug(
                    "fw_iteration",
                    iteration=iteration,
                    gap=gap,
                    divergence=divergence,
                    profit=profit,
                )
        
        # Max iterations reached
        gradient = self.bregman.gradient(best_mu)
        vertex, _ = self._lmo(gradient)
        if vertex is not None:
            gap = self._compute_gap(best_mu, gradient, vertex)
        else:
            gap = float('inf')
        divergence = self.bregman.divergence(best_mu, theta)
        
        return self._make_result(
            best_mu, theta, self.max_iterations, gap, divergence,
            time.time() - start_time, ip_time, len(active_set),
            "max_iterations"
        )
    
    def _make_result(
        self,
        mu: NDArray,
        theta: NDArray,
        iterations: int,
        gap: float,
        divergence: float,
        total_time: float,
        ip_time: float,
        active_vertices: int,
        status: str,
    ) -> FrankWolfeResult:
        """Create result object."""
        return FrankWolfeResult(
            mu=mu,
            theta=theta,
            iterations=iterations,
            gap=gap,
            divergence=divergence,
            guaranteed_profit=divergence - gap,
            total_time=total_time,
            ip_solve_time=ip_time,
            active_vertices=active_vertices,
            status=status,
        )


class BarrierFrankWolfe(FrankWolfe):
    """
    Barrier Frank-Wolfe with adaptive contraction.
    
    Solves the gradient explosion problem in LMSR by optimizing over
    contracted polytope M' = (1-ε)M + εu, where u is an interior point.
    
    The contraction keeps all coordinates away from 0, bounding the
    Lipschitz constant L ≤ 1/ε.
    
    Adaptive ε rule:
        If g(μ_t) / (-4g_u) < ε_{t-1}:
            ε_t = min{g(μ_t)/(-4g_u), ε_{t-1}/2}
        Else:
            ε_t = ε_{t-1}
    
    This ensures ε → 0 as optimization converges.
    """
    
    def __init__(
        self,
        bregman: BregmanDivergence,
        ip_solver: IPSolver,
        polytope: MarginalPolytope,
        initial_epsilon: float = 0.1,
        **kwargs,
    ):
        """
        Initialize Barrier Frank-Wolfe.
        
        Args:
            initial_epsilon: Initial contraction parameter
            **kwargs: Passed to FrankWolfe.__init__
        """
        super().__init__(bregman, ip_solver, polytope, **kwargs)
        self.initial_epsilon = initial_epsilon
    
    def init_fw(self) -> InitFWResult:
        """
        Algorithm 3: InitFW
        
        Initializes Frank-Wolfe by:
        1. Finding valid vertices for each unsettled variable being 0 and 1
        2. Building interior point u as average of vertices
        3. Extending partial outcome with logically settled variables
        
        Returns:
            InitFWResult with active set, interior point, and extended partials
        """
        n = self.polytope.n_variables
        active_set = []
        extended_partial = dict(self.polytope.partial_outcome)
        
        # For each unsettled variable, check if it can be 0 and 1
        unsettled = self.polytope.unsettled_indices
        
        for i in unsettled:
            # Can z[i] = 1?
            can_be_1 = self.ip_solver.check_feasibility(i, 1, self.A, self.b)
            # Can z[i] = 0?
            can_be_0 = self.ip_solver.check_feasibility(i, 0, self.A, self.b)
            
            if can_be_1 and can_be_0:
                # Variable is genuinely uncertain, find vertices for both
                # Find vertex with z[i] = 1
                obj = np.zeros(n)
                obj[i] = -1  # Minimize -z[i] to get z[i] = 1
                result = self.ip_solver.solve(obj, self.A, self.b)
                if result.is_feasible:
                    active_set.append(result.solution)
                
                # Find vertex with z[i] = 0
                obj[i] = 1  # Minimize z[i] to get z[i] = 0
                result = self.ip_solver.solve(obj, self.A, self.b)
                if result.is_feasible:
                    active_set.append(result.solution)
                    
            elif can_be_1 and not can_be_0:
                # Must be 1
                extended_partial[i] = 1
                logger.info("variable_must_be_1", index=i)
                
            elif can_be_0 and not can_be_1:
                # Must be 0
                extended_partial[i] = 0
                logger.info("variable_must_be_0", index=i)
                
            else:
                # Infeasible - shouldn't happen with valid polytope
                logger.error("variable_infeasible", index=i)
        
        # Deduplicate active set
        unique_vertices = []
        for v in active_set:
            is_dup = any(np.allclose(v, u) for u in unique_vertices)
            if not is_dup:
                unique_vertices.append(v)
        active_set = unique_vertices
        
        # Build interior point as average of vertices
        if active_set:
            interior_point = np.mean(active_set, axis=0)
        else:
            # Fallback: uniform over unsettled, fixed for settled
            interior_point = np.zeros(n)
            for market in self.polytope.markets:
                unsettled_in_market = [
                    i for i in range(market.start_index, market.end_index)
                    if i not in extended_partial
                ]
                if unsettled_in_market:
                    for i in unsettled_in_market:
                        interior_point[i] = 1.0 / len(unsettled_in_market)
            for i, v in extended_partial.items():
                interior_point[i] = v
        
        logger.info(
            "init_fw_complete",
            n_vertices=len(active_set),
            n_extended=len(extended_partial) - len(self.polytope.partial_outcome),
        )
        
        return InitFWResult(
            active_set=active_set,
            interior_point=interior_point,
            extended_partial=extended_partial,
        )
    
    def _contract_vertex(self, vertex: NDArray, u: NDArray, epsilon: float) -> NDArray:
        """Contract vertex toward interior point."""
        return (1 - epsilon) * vertex + epsilon * u
    
    def _compute_g_u(self, theta: NDArray, u: NDArray) -> float:
        """
        Compute g_u = ∇F(u)·(u - z*) where z* minimizes ∇F(u)·z.
        
        This is used in the adaptive epsilon rule.
        """
        gradient = self.bregman.gradient(u)
        vertex, _ = self._lmo(gradient)
        if vertex is None:
            return float('-inf')
        return np.dot(gradient, u - vertex)
    
    def run(
        self,
        theta: NDArray,
        initial_mu: Optional[NDArray] = None,
    ) -> FrankWolfeResult:
        """
        Run Barrier Frank-Wolfe with adaptive contraction.
        """
        start_time = time.time()
        ip_time = 0.0
        
        # Initialize
        init_result = self.init_fw()
        u = init_result.interior_point
        
        n = len(theta)
        epsilon = self.initial_epsilon
        
        if initial_mu is None:
            if isinstance(self.bregman, LMSRBregman):
                mu = self.bregman.prices_from_state(theta)
            else:
                mu = u.copy()
        else:
            mu = initial_mu.copy()
        
        # Contract initial point
        mu = (1 - epsilon) * mu + epsilon * u
        
        active_set = init_result.active_set.copy()
        best_mu = mu.copy()
        best_profit = float('-inf')
        
        # Compute g_u for adaptive epsilon
        g_u = self._compute_g_u(theta, u)
        
        for iteration in range(self.max_iterations):
            elapsed = time.time() - start_time
            if elapsed > self.time_limit:
                break
            
            # Gradient at contracted point
            gradient = self.bregman.gradient(mu)
            
            # LMO on original polytope, then contract
            ip_start = time.time()
            vertex, _ = self._lmo(gradient)
            ip_time += time.time() - ip_start
            
            if vertex is None:
                break
            
            # Contract vertex
            contracted_vertex = self._contract_vertex(vertex, u, epsilon)
            active_set.append(vertex)
            
            # Gap on contracted polytope
            gap = self._compute_gap(mu, gradient, contracted_vertex)
            divergence = self.bregman.divergence(mu, theta)
            profit = divergence - gap
            
            if profit > best_profit:
                best_profit = profit
                best_mu = mu.copy()
            
            # Stopping conditions
            if divergence < self.epsilon_D:
                return self._make_result(
                    best_mu, theta, iteration + 1, gap, divergence,
                    time.time() - start_time, ip_time, len(active_set),
                    "near_arbitrage_free"
                )
            
            if gap <= (1 - self.alpha) * divergence:
                return self._make_result(
                    best_mu, theta, iteration + 1, gap, divergence,
                    time.time() - start_time, ip_time, len(active_set),
                    "alpha_extracted"
                )
            
            # Adaptive epsilon update
            if g_u < 0:  # g_u should be negative
                ratio = gap / (-4 * g_u)
                if ratio < epsilon:
                    epsilon = min(ratio, epsilon / 2)
                    logger.debug("epsilon_updated", new_epsilon=epsilon)
            
            # Step toward contracted vertex
            gamma = self._line_search(mu, contracted_vertex, theta)
            mu = mu + gamma * (contracted_vertex - mu)
            mu = np.clip(mu, 1e-10, 1 - 1e-10)
            
            if iteration % 10 == 0:
                logger.debug(
                    "barrier_fw_iteration",
                    iteration=iteration,
                    epsilon=epsilon,
                    gap=gap,
                    divergence=divergence,
                    profit=profit,
                )
        
        # Final result
        gradient = self.bregman.gradient(best_mu)
        vertex, _ = self._lmo(gradient)
        if vertex is not None:
            gap = self._compute_gap(best_mu, gradient, vertex)
        else:
            gap = float('inf')
        divergence = self.bregman.divergence(best_mu, theta)
        
        return self._make_result(
            best_mu, theta, self.max_iterations, gap, divergence,
            time.time() - start_time, ip_time, len(active_set),
            "max_iterations"
        )
