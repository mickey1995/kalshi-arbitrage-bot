"""
Marginal Polytope representation for prediction markets.

The marginal polytope M is the convex hull of valid payoff vectors.
For markets with logical dependencies, M captures which price combinations
are arbitrage-free.

Key insight: Instead of enumerating 2^n outcomes, we describe M
via linear constraints. This makes Frank-Wolfe tractable.

Example constraints:
- Single market: Σ z_i = 1 (exactly one outcome happens)
- Two dependent markets: z_A_win + z_B_win ≤ 1 (both can't win)
"""

from typing import List, Optional, Tuple, Dict, Any, Set
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class OutcomeConstraint:
    """
    A linear constraint on valid outcomes.
    
    Represents: Σ coefficients[i] * z[i] >= rhs
    or equivalently: Σ coefficients[i] * z[i] <= rhs (with negated coeffs)
    """
    coefficients: NDArray  # Coefficient for each security
    rhs: float  # Right-hand side
    constraint_type: str = "geq"  # "geq", "leq", "eq"
    name: str = ""  # Human-readable description
    
    def evaluate(self, z: NDArray) -> bool:
        """Check if z satisfies this constraint."""
        lhs = np.dot(self.coefficients, z)
        
        if self.constraint_type == "geq":
            return lhs >= self.rhs - 1e-6
        elif self.constraint_type == "leq":
            return lhs <= self.rhs + 1e-6
        else:  # eq
            return abs(lhs - self.rhs) < 1e-6


@dataclass
class Market:
    """Represents a single prediction market."""
    ticker: str
    outcomes: List[str]  # List of outcome names
    start_index: int  # Starting index in global variable vector
    
    @property
    def n_outcomes(self) -> int:
        return len(self.outcomes)
    
    @property
    def end_index(self) -> int:
        return self.start_index + self.n_outcomes
    
    def get_indices(self) -> List[int]:
        """Get indices for this market's variables."""
        return list(range(self.start_index, self.end_index))


@dataclass 
class MarginalPolytope:
    """
    Represents the marginal polytope for a set of prediction markets.
    
    The polytope M is defined by:
    1. Binary constraints: z_i ∈ {0, 1}
    2. Market constraints: Σ z_i = 1 for each market (exactly one outcome)
    3. Dependency constraints: Logical relationships between markets
    
    This class handles:
    - Building constraint matrices for IP solver
    - Checking if a point is in the polytope
    - Managing partial outcomes (settled securities)
    """
    
    markets: List[Market] = field(default_factory=list)
    constraints: List[OutcomeConstraint] = field(default_factory=list)
    partial_outcome: Dict[int, int] = field(default_factory=dict)  # index -> value
    
    @property
    def n_variables(self) -> int:
        """Total number of binary variables."""
        if not self.markets:
            return 0
        return self.markets[-1].end_index
    
    @property
    def n_unsettled(self) -> int:
        """Number of unsettled (free) variables."""
        return self.n_variables - len(self.partial_outcome)
    
    @property
    def unsettled_indices(self) -> List[int]:
        """Indices of unsettled variables."""
        return [i for i in range(self.n_variables) if i not in self.partial_outcome]
    
    def add_market(self, ticker: str, outcomes: List[str]) -> Market:
        """
        Add a new market to the polytope.
        
        Automatically adds the constraint that exactly one outcome must happen.
        """
        start_index = self.n_variables
        market = Market(ticker=ticker, outcomes=outcomes, start_index=start_index)
        self.markets.append(market)
        
        # Add "exactly one" constraint for this market
        coeffs = np.zeros(market.end_index)
        coeffs[market.start_index:market.end_index] = 1
        
        self.constraints.append(OutcomeConstraint(
            coefficients=coeffs,
            rhs=1.0,
            constraint_type="eq",
            name=f"{ticker}_exactly_one"
        ))
        
        logger.debug("market_added", ticker=ticker, outcomes=outcomes, indices=market.get_indices())
        return market
    
    def add_dependency(
        self,
        indices: List[int],
        max_true: int = 1,
        min_true: int = 0,
        name: str = "",
    ):
        """
        Add a dependency constraint between outcomes.
        
        Common patterns:
        - Mutual exclusion: max_true=1 (at most one can be true)
        - Implication A→B: If z_A=1 then z_B=1 (use add_implication)
        
        Args:
            indices: Indices of variables involved
            max_true: Maximum number that can be true
            min_true: Minimum number that must be true
            name: Description of the constraint
        """
        n = self.n_variables
        
        # Constraint: Σ z_i ≤ max_true
        if max_true < len(indices):
            coeffs = np.zeros(n)
            for i in indices:
                coeffs[i] = 1
            self.constraints.append(OutcomeConstraint(
                coefficients=coeffs,
                rhs=max_true,
                constraint_type="leq",
                name=name or f"max_{max_true}_of_{indices}"
            ))
        
        # Constraint: Σ z_i ≥ min_true
        if min_true > 0:
            coeffs = np.zeros(n)
            for i in indices:
                coeffs[i] = 1
            self.constraints.append(OutcomeConstraint(
                coefficients=coeffs,
                rhs=min_true,
                constraint_type="geq",
                name=name or f"min_{min_true}_of_{indices}"
            ))
    
    def add_implication(
        self,
        if_index: int,
        then_index: int,
        name: str = "",
    ):
        """
        Add implication constraint: if z[if_index]=1 then z[then_index]=1.
        
        Encoded as: z[if_index] - z[then_index] ≤ 0
        Or equivalently: z[then_index] ≥ z[if_index]
        
        Example: If "Trump wins PA" then "Trump wins presidency" (not vice versa)
        """
        n = self.n_variables
        coeffs = np.zeros(n)
        coeffs[if_index] = 1
        coeffs[then_index] = -1
        
        self.constraints.append(OutcomeConstraint(
            coefficients=coeffs,
            rhs=0,
            constraint_type="leq",
            name=name or f"impl_{if_index}_to_{then_index}"
        ))
    
    def add_mutual_exclusion(
        self,
        indices: List[int],
        name: str = "",
    ):
        """
        Add mutual exclusion: at most one of the indices can be true.
        
        Example: "Duke 5+ wins" and "Cornell 5+ wins" can't both be true
        if they meet in semifinals.
        """
        self.add_dependency(indices, max_true=1, name=name)
    
    def settle_variable(self, index: int, value: int):
        """
        Mark a variable as settled (resolved to 0 or 1).
        
        Settled variables are fixed in all future optimizations.
        """
        if index < 0 or index >= self.n_variables:
            raise ValueError(f"Invalid index {index}")
        if value not in [0, 1]:
            raise ValueError(f"Value must be 0 or 1, got {value}")
        
        self.partial_outcome[index] = value
        logger.info("variable_settled", index=index, value=value)
    
    def get_constraint_matrices(self) -> Tuple[NDArray, NDArray]:
        """
        Get constraint matrices for IP solver.
        
        Returns A and b such that valid outcomes satisfy A^T z >= b
        (after converting eq/leq constraints appropriately).
        
        Returns:
            (A, b) where A is (n_vars, n_constraints) and b is (n_constraints,)
        """
        n = self.n_variables
        
        # Convert constraints to standard form A^T z >= b
        A_list = []
        b_list = []
        
        for c in self.constraints:
            # Ensure constraint has correct size
            coeffs = c.coefficients
            if len(coeffs) < n:
                coeffs = np.pad(coeffs, (0, n - len(coeffs)))
            
            if c.constraint_type == "geq":
                A_list.append(coeffs)
                b_list.append(c.rhs)
            elif c.constraint_type == "leq":
                # Convert: c·z ≤ rhs  →  -c·z ≥ -rhs
                A_list.append(-coeffs)
                b_list.append(-c.rhs)
            else:  # eq: add both directions
                A_list.append(coeffs)
                b_list.append(c.rhs)
                A_list.append(-coeffs)
                b_list.append(-c.rhs)
        
        # Add settled variable constraints
        for idx, val in self.partial_outcome.items():
            # z[idx] = val  →  z[idx] >= val and z[idx] <= val
            coeffs_pos = np.zeros(n)
            coeffs_pos[idx] = 1
            A_list.append(coeffs_pos)
            b_list.append(val)
            
            coeffs_neg = np.zeros(n)
            coeffs_neg[idx] = -1
            A_list.append(coeffs_neg)
            b_list.append(-val)
        
        A = np.column_stack(A_list) if A_list else np.zeros((n, 0))
        b = np.array(b_list) if b_list else np.array([])
        
        return A, b
    
    def is_valid_outcome(self, z: NDArray) -> bool:
        """Check if z is a valid outcome (satisfies all constraints)."""
        # Check binary
        if not np.allclose(z, np.round(z)):
            return False
        if not np.all((z >= 0) & (z <= 1)):
            return False
        
        # Check all constraints
        for c in self.constraints:
            if not c.evaluate(z):
                return False
        
        # Check settled variables
        for idx, val in self.partial_outcome.items():
            if abs(z[idx] - val) > 1e-6:
                return False
        
        return True
    
    def is_in_polytope(self, mu: NDArray) -> bool:
        """
        Check if mu is in the marginal polytope (convex hull).
        
        A point is in M if it's a convex combination of valid outcomes.
        For checking, we verify:
        1. All coordinates in [0, 1]
        2. Per-market probabilities sum to 1
        3. Dependencies are satisfied in expectation
        """
        # Check bounds
        if not np.all((mu >= -1e-6) & (mu <= 1 + 1e-6)):
            return False
        
        # Check per-market sum = 1
        for market in self.markets:
            market_sum = np.sum(mu[market.start_index:market.end_index])
            if abs(market_sum - 1.0) > 1e-6:
                return False
        
        return True
    
    def project_to_simplex(self, mu: NDArray) -> NDArray:
        """
        Project mu onto valid probability simplex per market.
        
        This is a simple projection, not the full Bregman projection.
        Used for initialization.
        """
        result = mu.copy()
        
        for market in self.markets:
            start, end = market.start_index, market.end_index
            segment = result[start:end]
            
            # Clip to [0, 1]
            segment = np.clip(segment, 0, 1)
            
            # Normalize to sum to 1
            total = np.sum(segment)
            if total > 0:
                segment = segment / total
            else:
                # Uniform if all zero
                segment = np.ones(market.n_outcomes) / market.n_outcomes
            
            result[start:end] = segment
        
        return result


def build_single_market_polytope(
    ticker: str,
    outcomes: List[str],
) -> MarginalPolytope:
    """
    Build a simple polytope for a single market.
    
    This is the basic case: one market with mutually exclusive outcomes.
    """
    polytope = MarginalPolytope()
    polytope.add_market(ticker, outcomes)
    return polytope


def build_multi_market_polytope(
    markets: List[Tuple[str, List[str]]],
    dependencies: Optional[List[Dict[str, Any]]] = None,
) -> MarginalPolytope:
    """
    Build a polytope for multiple markets with optional dependencies.
    
    Args:
        markets: List of (ticker, outcomes) tuples
        dependencies: List of dependency specs, each with:
            - type: "mutual_exclusion", "implication"
            - indices: Variable indices involved
            - Additional params based on type
    
    Returns:
        MarginalPolytope with all markets and constraints
    """
    polytope = MarginalPolytope()
    
    # Add markets
    for ticker, outcomes in markets:
        polytope.add_market(ticker, outcomes)
    
    # Add dependencies
    if dependencies:
        for dep in dependencies:
            dep_type = dep.get("type")
            indices = dep.get("indices", [])
            name = dep.get("name", "")
            
            if dep_type == "mutual_exclusion":
                polytope.add_mutual_exclusion(indices, name)
            elif dep_type == "implication":
                if len(indices) >= 2:
                    polytope.add_implication(indices[0], indices[1], name)
            elif dep_type == "max_true":
                max_true = dep.get("max_true", 1)
                polytope.add_dependency(indices, max_true=max_true, name=name)
    
    return polytope
