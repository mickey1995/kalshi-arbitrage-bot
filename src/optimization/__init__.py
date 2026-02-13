"""
Optimization module for arbitrage detection.

Implements:
- Frank-Wolfe algorithm with Bregman projection
- Barrier Frank-Wolfe with adaptive contraction
- Integer Programming solver integration
- LMSR cost function handling
"""

from .frank_wolfe import FrankWolfe, BarrierFrankWolfe, FrankWolfeResult
from .bregman import BregmanDivergence, LMSRBregman
from .ip_solver import IPSolver, IPSolverFactory
from .marginal_polytope import MarginalPolytope, OutcomeConstraint

__all__ = [
    "FrankWolfe",
    "BarrierFrankWolfe",
    "FrankWolfeResult",
    "BregmanDivergence",
    "LMSRBregman",
    "IPSolver",
    "IPSolverFactory",
    "MarginalPolytope",
    "OutcomeConstraint",
]
