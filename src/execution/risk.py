"""
Risk Management and Position Sizing.

Implements:
- Kelly Criterion for optimal position sizing
- Drawdown limits
- Position concentration limits
- Execution risk scoring
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

import structlog

from ..api.models import ArbitrageOpportunity, Position, PortfolioBalance

logger = structlog.get_logger(__name__)


@dataclass
class RiskLimits:
    """Risk management limits."""
    max_position_size: float = 0.1  # Max 10% of portfolio per position
    max_single_trade: float = 0.05  # Max 5% of portfolio per trade
    max_drawdown: float = 0.20  # 20% max drawdown (more tolerance for learning)
    max_daily_loss: float = 0.10  # 10% max daily loss
    max_open_positions: int = 50  # More positions for learning
    max_concentration: float = 0.3  # Max 30% in any single market
    min_profit_threshold: float = 0.01  # $0.01 minimum profit (1 cent for learning)
    
    # Learning mode
    learning_mode: bool = True
    learning_bet_size: float = 1.0  # $1 per trade
    max_daily_trades: int = 500


class RiskManager:
    """
    Risk management system.
    
    Enforces position limits, drawdown controls, and trade validation.
    """
    
    def __init__(self, limits: RiskLimits = None):
        """Initialize risk manager."""
        self.limits = limits or RiskLimits()
        
        # Track P&L
        self.starting_balance: Optional[float] = None
        self.peak_balance: float = 0.0
        self.daily_pnl: float = 0.0
        
        logger.info("risk_manager_initialized", limits=self.limits)
    
    def set_starting_balance(self, balance: float):
        """Set starting balance for drawdown calculation."""
        self.starting_balance = balance
        self.peak_balance = balance
    
    def update_balance(self, current_balance: float):
        """Update balance and check drawdown."""
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        
        # Check drawdown
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - current_balance) / self.peak_balance
            if drawdown > self.limits.max_drawdown:
                logger.warning(
                    "max_drawdown_exceeded",
                    drawdown=drawdown,
                    limit=self.limits.max_drawdown,
                )
                return False
        
        return True
    
    def can_trade(
        self,
        opportunity: ArbitrageOpportunity,
        balance: PortfolioBalance,
        positions: List[Position],
    ) -> tuple[bool, str]:
        """
        Check if a trade is allowed under risk limits.
        
        Returns:
            (allowed, reason) tuple
        """
        # Check minimum profit
        if opportunity.guaranteed_profit < self.limits.min_profit_threshold:
            return False, f"Profit {opportunity.guaranteed_profit:.4f} below threshold"
        
        # Check available balance
        if opportunity.total_cost > balance.available_balance:
            return False, "Insufficient balance"
        
        # Check single trade size
        max_trade = balance.total_balance * self.limits.max_single_trade
        if opportunity.total_cost > max_trade:
            return False, f"Trade size {opportunity.total_cost:.2f} exceeds limit {max_trade:.2f}"
        
        # Check position count
        active_positions = len([p for p in positions if p.has_position])
        if active_positions >= self.limits.max_open_positions:
            return False, f"Max positions ({self.limits.max_open_positions}) reached"
        
        # Check concentration
        for ticker in opportunity.tickers:
            existing = next((p for p in positions if p.ticker == ticker), None)
            if existing and existing.has_position:
                existing_value = (existing.yes_quantity + existing.no_quantity) * 0.50  # Rough estimate
                new_value = existing_value + opportunity.total_cost
                if new_value > balance.total_balance * self.limits.max_concentration:
                    return False, f"Position in {ticker} would exceed concentration limit"
        
        # Check execution risk
        if opportunity.execution_risk_score > 0.5:
            return False, f"Execution risk {opportunity.execution_risk_score:.2f} too high"
        
        return True, "OK"
    
    def record_trade_result(self, pnl: float):
        """Record trade result for daily P&L tracking."""
        self.daily_pnl += pnl
        
        if self.starting_balance and self.starting_balance > 0:
            daily_loss_ratio = -self.daily_pnl / self.starting_balance
            if daily_loss_ratio > self.limits.max_daily_loss:
                logger.warning(
                    "daily_loss_limit_exceeded",
                    daily_pnl=self.daily_pnl,
                    limit=self.limits.max_daily_loss,
                )
    
    def reset_daily(self):
        """Reset daily tracking (call at start of each day)."""
        self.daily_pnl = 0.0


class PositionSizer:
    """
    Position sizing using modified Kelly Criterion.
    
    Kelly formula: f* = (bp - q) / b
    Where:
    - b = odds (profit ratio)
    - p = probability of winning
    - q = 1 - p
    
    For arbitrage, p is based on execution success probability.
    
    In LEARNING MODE: Uses fixed small bet sizes to make many trades.
    """
    
    def __init__(
        self,
        kelly_fraction: float = 0.5,  # Half-Kelly for safety
        max_position_fraction: float = 0.1,
        min_position: float = 1.0,  # Minimum $1 position for learning
        learning_mode: bool = True,
        learning_bet_size: float = 1.0,  # $1 per trade in learning mode
    ):
        """
        Initialize position sizer.
        
        Args:
            kelly_fraction: Fraction of Kelly to use (0.5 = half-Kelly)
            max_position_fraction: Maximum fraction of portfolio
            min_position: Minimum position size in dollars
            learning_mode: If True, use fixed bet size
            learning_bet_size: Fixed bet size for learning mode
        """
        self.kelly_fraction = kelly_fraction
        self.max_position_fraction = max_position_fraction
        self.min_position = min_position
        self.learning_mode = learning_mode
        self.learning_bet_size = learning_bet_size
    
    def calculate_size(
        self,
        opportunity: ArbitrageOpportunity,
        balance: PortfolioBalance,
        execution_probability: float = 0.9,
    ) -> float:
        """
        Calculate optimal position size.
        
        In LEARNING MODE: Returns fixed small bet size for making many trades.
        In PRODUCTION MODE: Uses Kelly Criterion for optimal sizing.
        
        Args:
            opportunity: The arbitrage opportunity
            balance: Current portfolio balance
            execution_probability: Estimated probability of successful execution
            
        Returns:
            Recommended position size in dollars
        """
        if opportunity.total_cost <= 0:
            return 0.0
        
        # LEARNING MODE: Fixed small bets
        if self.learning_mode:
            # Use fixed bet size, but respect balance limits
            position_size = min(
                self.learning_bet_size,
                balance.available_balance * 0.02,  # Max 2% per trade
                opportunity.available_liquidity * (opportunity.total_cost if opportunity.total_cost > 0 else 1),
            )
            
            if position_size >= self.min_position:
                logger.debug(
                    "learning_mode_position",
                    size=position_size,
                    bet_size=self.learning_bet_size,
                )
                return position_size
            return 0.0
        
        # PRODUCTION MODE: Kelly Criterion
        # Calculate odds ratio (profit / cost)
        b = opportunity.guaranteed_profit / opportunity.total_cost
        
        # Execution probability
        p = execution_probability * (1 - opportunity.execution_risk_score)
        q = 1 - p
        
        # Kelly formula
        if b <= 0 or p <= 0:
            return 0.0
        
        kelly = (b * p - q) / b
        
        # Apply fraction and cap
        position_fraction = kelly * self.kelly_fraction
        position_fraction = max(0, min(position_fraction, self.max_position_fraction))
        
        # Calculate dollar amount
        position_size = balance.available_balance * position_fraction
        
        # Apply minimum
        if position_size < self.min_position:
            return 0.0  # Don't trade if below minimum
        
        # Cap by available liquidity
        position_size = min(position_size, opportunity.available_liquidity * opportunity.total_cost)
        
        logger.debug(
            "position_size_calculated",
            kelly=kelly,
            fraction=position_fraction,
            size=position_size,
        )
        
        return position_size
    
    def calculate_quantity(
        self,
        position_size: float,
        price_cents: int,
    ) -> int:
        """
        Convert dollar position size to contract quantity.
        
        Args:
            position_size: Position size in dollars
            price_cents: Price per contract in cents
            
        Returns:
            Number of contracts
        """
        if price_cents <= 0:
            return 0
        
        cost_per_contract = price_cents / 100
        quantity = int(position_size / cost_per_contract)
        
        return max(0, quantity)
