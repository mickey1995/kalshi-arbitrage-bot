"""
Integer Programming Solver for finding valid payoff vectors.

The IP solver is the computational core that makes Frank-Wolfe tractable
for combinatorial prediction markets. Instead of enumerating 2^n outcomes,
we describe valid outcomes with linear constraints and solve:

    min c·z  subject to  A^T z ≥ b, z ∈ {0,1}^n

This finds the vertex of the marginal polytope M that minimizes c·z,
which is exactly what Frank-Wolfe needs at each iteration.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class IPResult:
    """Result from integer programming solver."""
    solution: Optional[NDArray]  # Optimal z vector (binary)
    objective: float  # Optimal objective value
    status: str  # "optimal", "infeasible", "timeout", etc.
    solve_time: float  # Time in seconds
    
    @property
    def is_optimal(self) -> bool:
        return self.status == "optimal"
    
    @property
    def is_feasible(self) -> bool:
        return self.solution is not None


class IPSolver(ABC):
    """
    Abstract base class for Integer Programming solvers.
    
    Solvers find valid payoff vectors z ∈ {0,1}^n that satisfy
    outcome constraints. This is the LMO (Linear Minimization Oracle)
    for Frank-Wolfe.
    """
    
    @abstractmethod
    def solve(
        self,
        objective: NDArray,
        constraints_A: NDArray,
        constraints_b: NDArray,
        time_limit: float = 30.0,
    ) -> IPResult:
        """
        Solve the integer program:
            min c·z  s.t.  A^T z ≥ b, z ∈ {0,1}^n
        
        Args:
            objective: Coefficient vector c
            constraints_A: Constraint matrix A (columns are constraints)
            constraints_b: Right-hand side b
            time_limit: Maximum solve time in seconds
            
        Returns:
            IPResult with solution
        """
        pass
    
    @abstractmethod
    def check_feasibility(
        self,
        variable_index: int,
        fixed_value: int,
        constraints_A: NDArray,
        constraints_b: NDArray,
    ) -> bool:
        """
        Check if fixing z[variable_index] = fixed_value is feasible.
        
        Used by InitFW to determine if a security must resolve
        to a specific value.
        
        Args:
            variable_index: Index of variable to fix
            fixed_value: Value to fix (0 or 1)
            constraints_A: Constraint matrix
            constraints_b: Right-hand side
            
        Returns:
            True if feasible, False otherwise
        """
        pass


class ORToolsSolver(IPSolver):
    """
    IP Solver using Google OR-Tools (free, open-source).
    
    OR-Tools is a good open-source alternative to Gurobi.
    For most prediction market problems, it provides adequate
    performance (solving in seconds to minutes).
    """
    
    def __init__(self, solver_name: str = "SCIP"):
        """
        Initialize OR-Tools solver.
        
        Args:
            solver_name: Backend solver ("SCIP", "CBC", "GLOP" for LP)
        """
        try:
            from ortools.linear_solver import pywraplp
            self.pywraplp = pywraplp
        except ImportError:
            raise ImportError("OR-Tools not installed. Run: pip install ortools")
        
        self.solver_name = solver_name
        logger.info("ortools_solver_initialized", backend=solver_name)
    
    def solve(
        self,
        objective: NDArray,
        constraints_A: NDArray,
        constraints_b: NDArray,
        time_limit: float = 30.0,
    ) -> IPResult:
        """Solve IP using OR-Tools."""
        import time
        start_time = time.time()
        
        n = len(objective)
        m = len(constraints_b)
        
        # Create solver
        solver = self.pywraplp.Solver.CreateSolver(self.solver_name)
        if not solver:
            return IPResult(None, float('inf'), "solver_unavailable", 0.0)
        
        solver.SetTimeLimit(int(time_limit * 1000))  # Milliseconds
        
        # Create binary variables
        z = [solver.IntVar(0, 1, f'z_{i}') for i in range(n)]
        
        # Add constraints: A^T z >= b
        for j in range(m):
            constraint = solver.Constraint(float(constraints_b[j]), solver.infinity())
            for i in range(n):
                constraint.SetCoefficient(z[i], float(constraints_A[i, j]))
        
        # Set objective: minimize c·z
        objective_fn = solver.Objective()
        for i in range(n):
            objective_fn.SetCoefficient(z[i], float(objective[i]))
        objective_fn.SetMinimization()
        
        # Solve
        status = solver.Solve()
        solve_time = time.time() - start_time
        
        if status == self.pywraplp.Solver.OPTIMAL:
            solution = np.array([z[i].solution_value() for i in range(n)])
            obj_value = objective_fn.Value()
            return IPResult(solution, obj_value, "optimal", solve_time)
        elif status == self.pywraplp.Solver.INFEASIBLE:
            return IPResult(None, float('inf'), "infeasible", solve_time)
        elif status == self.pywraplp.Solver.NOT_SOLVED:
            return IPResult(None, float('inf'), "timeout", solve_time)
        else:
            return IPResult(None, float('inf'), f"unknown_{status}", solve_time)
    
    def check_feasibility(
        self,
        variable_index: int,
        fixed_value: int,
        constraints_A: NDArray,
        constraints_b: NDArray,
    ) -> bool:
        """Check feasibility with one variable fixed."""
        n = constraints_A.shape[0]
        
        # Create solver for feasibility check
        solver = self.pywraplp.Solver.CreateSolver(self.solver_name)
        if not solver:
            return False
        
        solver.SetTimeLimit(5000)  # 5 second limit for feasibility checks
        
        # Create variables
        z = [solver.IntVar(0, 1, f'z_{i}') for i in range(n)]
        
        # Fix the specified variable
        z[variable_index].SetBounds(fixed_value, fixed_value)
        
        # Add constraints
        m = len(constraints_b)
        for j in range(m):
            constraint = solver.Constraint(float(constraints_b[j]), solver.infinity())
            for i in range(n):
                constraint.SetCoefficient(z[i], float(constraints_A[i, j]))
        
        # Just check feasibility (objective doesn't matter)
        status = solver.Solve()
        
        return status == self.pywraplp.Solver.OPTIMAL


class GurobiSolver(IPSolver):
    """
    IP Solver using Gurobi (commercial, fastest).
    
    Gurobi is the industry standard for IP solving.
    Requires a license ($12K/year commercial, free academic).
    
    For production systems with millions in capital,
    the license cost is trivial compared to execution speed gains.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize Gurobi solver.
        
        Args:
            verbose: Whether to show Gurobi output
        """
        try:
            import gurobipy as gp
            from gurobipy import GRB
            self.gp = gp
            self.GRB = GRB
        except ImportError:
            raise ImportError(
                "Gurobi not installed. Get license at gurobi.com, "
                "then: pip install gurobipy"
            )
        
        self.verbose = verbose
        logger.info("gurobi_solver_initialized")
    
    def solve(
        self,
        objective: NDArray,
        constraints_A: NDArray,
        constraints_b: NDArray,
        time_limit: float = 30.0,
    ) -> IPResult:
        """Solve IP using Gurobi."""
        import time
        start_time = time.time()
        
        n = len(objective)
        m = len(constraints_b)
        
        # Create model
        model = self.gp.Model("frank_wolfe_lmo")
        if not self.verbose:
            model.Params.OutputFlag = 0
        model.Params.TimeLimit = time_limit
        
        # Create binary variables
        z = model.addVars(n, vtype=self.GRB.BINARY, name="z")
        
        # Add constraints: A^T z >= b
        for j in range(m):
            model.addConstr(
                self.gp.quicksum(constraints_A[i, j] * z[i] for i in range(n))
                >= constraints_b[j]
            )
        
        # Set objective
        model.setObjective(
            self.gp.quicksum(objective[i] * z[i] for i in range(n)),
            self.GRB.MINIMIZE
        )
        
        # Solve
        model.optimize()
        solve_time = time.time() - start_time
        
        if model.Status == self.GRB.OPTIMAL:
            solution = np.array([z[i].X for i in range(n)])
            obj_value = model.ObjVal
            return IPResult(solution, obj_value, "optimal", solve_time)
        elif model.Status == self.GRB.INFEASIBLE:
            return IPResult(None, float('inf'), "infeasible", solve_time)
        elif model.Status == self.GRB.TIME_LIMIT:
            # Return best found if available
            if model.SolCount > 0:
                solution = np.array([z[i].X for i in range(n)])
                return IPResult(solution, model.ObjVal, "timeout_with_solution", solve_time)
            return IPResult(None, float('inf'), "timeout", solve_time)
        else:
            return IPResult(None, float('inf'), f"status_{model.Status}", solve_time)
    
    def check_feasibility(
        self,
        variable_index: int,
        fixed_value: int,
        constraints_A: NDArray,
        constraints_b: NDArray,
    ) -> bool:
        """Check feasibility with Gurobi."""
        n = constraints_A.shape[0]
        m = len(constraints_b)
        
        model = self.gp.Model("feasibility")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = 5
        
        z = model.addVars(n, vtype=self.GRB.BINARY, name="z")
        
        # Fix variable
        model.addConstr(z[variable_index] == fixed_value)
        
        # Add other constraints
        for j in range(m):
            model.addConstr(
                self.gp.quicksum(constraints_A[i, j] * z[i] for i in range(n))
                >= constraints_b[j]
            )
        
        model.optimize()
        
        return model.Status == self.GRB.OPTIMAL


class IPSolverFactory:
    """Factory for creating IP solvers."""
    
    @staticmethod
    def create(solver_type: str = "ortools") -> IPSolver:
        """
        Create an IP solver.
        
        Args:
            solver_type: "ortools", "gurobi", or "scip"
            
        Returns:
            IPSolver instance
        """
        solver_type = solver_type.lower()
        
        if solver_type == "gurobi":
            return GurobiSolver()
        elif solver_type in ["ortools", "scip"]:
            return ORToolsSolver("SCIP")
        elif solver_type == "cbc":
            return ORToolsSolver("CBC")
        else:
            logger.warning(f"Unknown solver '{solver_type}', falling back to OR-Tools")
            return ORToolsSolver("SCIP")
