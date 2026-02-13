"""
Execution Engine for arbitrage trading.

Handles:
- Parallel order submission
- Order tracking and status monitoring
- Execution result analysis
- Failed trade recovery
"""

import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
import uuid

import structlog

from ..api.client import KalshiClient
from ..api.models import (
    ArbitrageOpportunity,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    Position,
    PortfolioBalance,
)
from .risk import RiskManager, PositionSizer

logger = structlog.get_logger(__name__)


@dataclass
class ExecutionResult:
    """Result of executing an arbitrage opportunity."""
    
    opportunity_id: str
    success: bool
    
    # Orders placed
    orders: List[Order] = field(default_factory=list)
    
    # Execution metrics
    total_cost: float = 0.0
    total_filled: int = 0
    fill_rate: float = 0.0
    
    # P&L
    realized_pnl: float = 0.0
    expected_pnl: float = 0.0
    
    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    execution_time_ms: float = 0.0
    
    # Errors
    errors: List[str] = field(default_factory=list)
    
    @property
    def slippage(self) -> float:
        """Slippage vs expected profit."""
        if self.expected_pnl <= 0:
            return 0.0
        return (self.expected_pnl - self.realized_pnl) / self.expected_pnl


@dataclass
class PendingExecution:
    """Tracks an in-progress execution."""
    
    opportunity: ArbitrageOpportunity
    orders: Dict[str, Order]  # order_id -> Order
    client_order_ids: Dict[str, str]  # client_order_id -> ticker
    started_at: datetime
    
    @property
    def all_filled(self) -> bool:
        return all(o.status == OrderStatus.FILLED for o in self.orders.values())
    
    @property
    def any_failed(self) -> bool:
        failed_statuses = {OrderStatus.CANCELLED, OrderStatus.EXPIRED}
        return any(o.status in failed_statuses for o in self.orders.values())


class ExecutionEngine:
    """
    Main execution engine for arbitrage trades.
    
    Features:
    - Parallel order submission for all legs
    - Real-time order status tracking
    - Automatic cancellation on partial fills
    - P&L tracking
    """
    
    def __init__(
        self,
        client: KalshiClient,
        risk_manager: RiskManager,
        position_sizer: PositionSizer,
        max_concurrent: int = 5,
    ):
        """
        Initialize execution engine.
        
        Args:
            client: Kalshi API client
            risk_manager: Risk management system
            position_sizer: Position sizing system
            max_concurrent: Maximum concurrent executions
        """
        self.client = client
        self.risk_manager = risk_manager
        self.position_sizer = position_sizer
        self.max_concurrent = max_concurrent
        
        # Tracking
        self.pending: Dict[str, PendingExecution] = {}
        self.completed: List[ExecutionResult] = []
        
        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
        # Statistics
        self.total_trades = 0
        self.successful_trades = 0
        self.total_profit = 0.0
        
        logger.info(
            "execution_engine_initialized",
            max_concurrent=max_concurrent,
        )
    
    async def execute(
        self,
        opportunity: ArbitrageOpportunity,
        balance: PortfolioBalance,
        positions: List[Position],
    ) -> ExecutionResult:
        """
        Execute an arbitrage opportunity.
        
        Args:
            opportunity: The opportunity to execute
            balance: Current portfolio balance
            positions: Current positions
            
        Returns:
            ExecutionResult with details
        """
        async with self._semaphore:
            return await self._execute_internal(opportunity, balance, positions)
    
    async def _execute_internal(
        self,
        opportunity: ArbitrageOpportunity,
        balance: PortfolioBalance,
        positions: List[Position],
    ) -> ExecutionResult:
        """Internal execution logic."""
        start_time = datetime.utcnow()
        result = ExecutionResult(
            opportunity_id=opportunity.opportunity_id,
            success=False,
            expected_pnl=opportunity.guaranteed_profit,
            started_at=start_time,
        )
        
        # Risk check
        can_trade, reason = self.risk_manager.can_trade(opportunity, balance, positions)
        if not can_trade:
            result.errors.append(f"Risk check failed: {reason}")
            logger.warning(
                "execution_blocked_by_risk",
                opportunity_id=opportunity.opportunity_id,
                reason=reason,
            )
            return result
        
        # Calculate position size
        position_size = self.position_sizer.calculate_size(
            opportunity, balance, 
            execution_probability=1.0 - opportunity.execution_risk_score
        )
        
        if position_size <= 0:
            result.errors.append("Position size too small")
            return result
        
        try:
            # Build orders based on opportunity type
            orders_to_place = self._build_orders(opportunity, position_size)
            
            if not orders_to_place:
                result.errors.append("No orders to place")
                return result
            
            # Submit all orders in parallel
            logger.info(
                "submitting_orders",
                opportunity_id=opportunity.opportunity_id,
                n_orders=len(orders_to_place),
            )
            
            placed_orders = await self.client.create_orders_parallel(orders_to_place)
            
            # Track execution
            pending = PendingExecution(
                opportunity=opportunity,
                orders={o.order_id: o for o in placed_orders},
                client_order_ids={o.client_order_id: o.ticker for o in placed_orders if o.client_order_id},
                started_at=start_time,
            )
            self.pending[opportunity.opportunity_id] = pending
            
            # Wait for fills (with timeout)
            await self._wait_for_fills(pending, timeout=30.0)
            
            # Calculate results
            result.orders = list(pending.orders.values())
            result.total_filled = sum(o.filled_quantity for o in result.orders)
            result.fill_rate = sum(o.fill_percent for o in result.orders) / len(result.orders) if result.orders else 0
            
            # Calculate actual cost and P&L
            result.total_cost = sum(
                (o.price or 0) * o.filled_quantity / 100
                for o in result.orders
            )
            
            # For arbitrage, profit is guaranteed payout minus cost
            # Actual P&L depends on settlement, but we can estimate
            if result.fill_rate >= 0.95:
                result.realized_pnl = opportunity.guaranteed_profit * result.fill_rate
                result.success = True
                self.successful_trades += 1
                self.total_profit += result.realized_pnl
            else:
                # Partial fill - may need to unwind
                result.errors.append(f"Partial fill: {result.fill_rate:.1%}")
                await self._handle_partial_fill(pending)
            
            self.total_trades += 1
            
        except Exception as e:
            result.errors.append(str(e))
            logger.error(
                "execution_failed",
                opportunity_id=opportunity.opportunity_id,
                error=str(e),
            )
        finally:
            # Cleanup
            self.pending.pop(opportunity.opportunity_id, None)
            
            result.completed_at = datetime.utcnow()
            result.execution_time_ms = (result.completed_at - start_time).total_seconds() * 1000
            
            self.completed.append(result)
            
            # Update risk manager
            self.risk_manager.record_trade_result(result.realized_pnl)
        
        logger.info(
            "execution_complete",
            opportunity_id=opportunity.opportunity_id,
            success=result.success,
            pnl=result.realized_pnl,
            execution_time_ms=result.execution_time_ms,
        )
        
        return result
    
    def _build_orders(
        self,
        opportunity: ArbitrageOpportunity,
        position_size: float,
    ) -> List[Dict[str, Any]]:
        """Build order specifications from opportunity."""
        orders = []
        
        for ticker, pos_spec in opportunity.recommended_positions.items():
            # Handle different opportunity types
            if "yes_action" in pos_spec:
                # Single condition with both sides
                if pos_spec.get("yes_action") == "buy":
                    quantity = self.position_sizer.calculate_quantity(
                        position_size / 2, pos_spec.get("yes_price", 50)
                    )
                    if quantity > 0:
                        orders.append({
                            "ticker": ticker,
                            "side": OrderSide.YES,
                            "quantity": quantity,
                            "price": pos_spec.get("yes_price"),
                        })
                
                if pos_spec.get("no_action") == "buy":
                    quantity = self.position_sizer.calculate_quantity(
                        position_size / 2, pos_spec.get("no_price", 50)
                    )
                    if quantity > 0:
                        orders.append({
                            "ticker": ticker,
                            "side": OrderSide.NO,
                            "quantity": quantity,
                            "price": pos_spec.get("no_price"),
                        })
            
            elif pos_spec.get("action") == "buy_yes":
                # Market rebalancing - buy YES
                quantity = self.position_sizer.calculate_quantity(
                    position_size / len(opportunity.recommended_positions),
                    pos_spec.get("price", 50)
                )
                if quantity > 0:
                    orders.append({
                        "ticker": ticker,
                        "side": OrderSide.YES,
                        "quantity": quantity,
                        "price": pos_spec.get("price"),
                    })
        
        return orders
    
    async def _wait_for_fills(
        self,
        pending: PendingExecution,
        timeout: float = 30.0,
    ):
        """Wait for all orders to fill or timeout."""
        start = datetime.utcnow()
        poll_interval = 0.5  # Poll every 500ms
        
        while (datetime.utcnow() - start).total_seconds() < timeout:
            # Check all order statuses
            all_terminal = True
            
            for order_id in list(pending.orders.keys()):
                order = pending.orders[order_id]
                
                if order.is_active:
                    all_terminal = False
                    
                    # Refresh order status
                    try:
                        updated = await self.client.get_order(order_id)
                        pending.orders[order_id] = updated
                    except Exception as e:
                        logger.warning(f"Failed to get order status: {e}")
            
            if all_terminal:
                break
            
            await asyncio.sleep(poll_interval)
        
        # Cancel any remaining active orders
        for order_id, order in pending.orders.items():
            if order.is_active:
                logger.warning("order_timeout_cancelling", order_id=order_id)
                await self.client.cancel_order(order_id)
    
    async def _handle_partial_fill(self, pending: PendingExecution):
        """Handle partial fills by unwinding positions if needed."""
        # For arbitrage, partial fills can leave us exposed
        # Cancel unfilled orders
        for order_id, order in pending.orders.items():
            if order.is_active:
                await self.client.cancel_order(order_id)
        
        # TODO: Implement position unwinding logic
        # This would involve selling filled positions at market
        logger.warning(
            "partial_fill_needs_unwind",
            opportunity_id=pending.opportunity.opportunity_id,
        )
    
    async def execute_batch(
        self,
        opportunities: List[ArbitrageOpportunity],
        balance: PortfolioBalance,
        positions: List[Position],
    ) -> List[ExecutionResult]:
        """
        Execute multiple opportunities in parallel.
        
        Respects max_concurrent limit.
        """
        tasks = [
            self.execute(opp, balance, positions)
            for opp in opportunities[:self.max_concurrent]
        ]
        
        return await asyncio.gather(*tasks)
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "total_trades": self.total_trades,
            "successful_trades": self.successful_trades,
            "success_rate": self.successful_trades / self.total_trades if self.total_trades > 0 else 0,
            "total_profit": self.total_profit,
            "avg_profit_per_trade": self.total_profit / self.successful_trades if self.successful_trades > 0 else 0,
            "pending_executions": len(self.pending),
        }
