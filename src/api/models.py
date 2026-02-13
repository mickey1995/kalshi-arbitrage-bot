"""
Data models for Kalshi API responses.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from decimal import Decimal


class OrderSide(str, Enum):
    """Order side (buy/sell)."""
    YES = "yes"
    NO = "no"


class OrderType(str, Enum):
    """Order type."""
    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(str, Enum):
    """Order status."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class MarketStatus(str, Enum):
    """Market status."""
    OPEN = "open"
    ACTIVE = "active"
    INITIALIZED = "initialized"
    CLOSED = "closed"
    SETTLED = "settled"
    FINALIZED = "finalized"
    DETERMINATION = "determination"


class OrderBookLevel(BaseModel):
    """Single level in order book."""
    price: int  # Price in cents (0-100)
    quantity: int  # Number of contracts


class OrderBook(BaseModel):
    """Order book for a market."""
    ticker: str
    yes_bids: List[OrderBookLevel] = Field(default_factory=list)
    no_bids: List[OrderBookLevel] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def best_yes_bid(self) -> Optional[int]:
        """Best YES bid price in cents."""
        return self.yes_bids[0].price if self.yes_bids else None
    
    @property
    def best_no_bid(self) -> Optional[int]:
        """Best NO bid price in cents."""
        return self.no_bids[0].price if self.no_bids else None
    
    @property
    def yes_ask(self) -> Optional[int]:
        """Implied YES ask (100 - best NO bid)."""
        if self.best_no_bid is not None:
            return 100 - self.best_no_bid
        return None
    
    @property
    def no_ask(self) -> Optional[int]:
        """Implied NO ask (100 - best YES bid)."""
        if self.best_yes_bid is not None:
            return 100 - self.best_yes_bid
        return None
    
    def get_vwap(self, side: OrderSide, quantity: int) -> Optional[float]:
        """
        Calculate Volume-Weighted Average Price for a given quantity.
        
        Args:
            side: YES or NO side
            quantity: Number of contracts to fill
            
        Returns:
            VWAP in cents, or None if insufficient liquidity
        """
        bids = self.yes_bids if side == OrderSide.YES else self.no_bids
        if not bids:
            return None
        
        remaining = quantity
        total_cost = 0
        total_filled = 0
        
        for level in bids:
            fill_qty = min(remaining, level.quantity)
            total_cost += fill_qty * level.price
            total_filled += fill_qty
            remaining -= fill_qty
            
            if remaining <= 0:
                break
        
        if total_filled < quantity:
            return None  # Insufficient liquidity
        
        return total_cost / total_filled
    
    def get_depth(self, side: OrderSide) -> int:
        """Get total depth on a side."""
        bids = self.yes_bids if side == OrderSide.YES else self.no_bids
        return sum(level.quantity for level in bids)


class Market(BaseModel):
    """Kalshi market data."""
    ticker: str
    title: str
    subtitle: Optional[str] = None
    status: MarketStatus
    event_ticker: str
    
    # Prices in cents (0-100)
    yes_bid: Optional[int] = None
    yes_ask: Optional[int] = None
    no_bid: Optional[int] = None
    no_ask: Optional[int] = None
    last_price: Optional[int] = None
    
    # Volume
    volume: int = 0
    volume_24h: int = 0
    open_interest: int = 0
    
    # Timestamps
    open_time: Optional[datetime] = None
    close_time: Optional[datetime] = None
    expiration_time: Optional[datetime] = None
    
    # Settlement
    result: Optional[str] = None  # "yes", "no", or None if not settled
    
    # Additional metadata
    floor_strike: Optional[float] = None
    cap_strike: Optional[float] = None
    
    @property
    def yes_price(self) -> Optional[float]:
        """Current YES price as decimal (0.0 - 1.0)."""
        if self.yes_bid is not None and self.yes_ask is not None:
            return (self.yes_bid + self.yes_ask) / 200  # Midpoint
        return None
    
    @property
    def no_price(self) -> Optional[float]:
        """Current NO price as decimal (0.0 - 1.0)."""
        if self.no_bid is not None and self.no_ask is not None:
            return (self.no_bid + self.no_ask) / 200  # Midpoint
        return None
    
    @property
    def is_tradeable(self) -> bool:
        """Check if market is currently tradeable."""
        return self.status in [MarketStatus.OPEN, MarketStatus.ACTIVE]


class Event(BaseModel):
    """Kalshi event (collection of related markets)."""
    event_ticker: str
    title: str
    subtitle: Optional[str] = None
    category: Optional[str] = None
    series_ticker: Optional[str] = None
    
    markets: List[Market] = Field(default_factory=list)
    
    @property
    def market_tickers(self) -> List[str]:
        """Get all market tickers in this event."""
        return [m.ticker for m in self.markets]
    
    @property
    def is_multi_outcome(self) -> bool:
        """Check if this is a multi-outcome event."""
        return len(self.markets) > 1


class Order(BaseModel):
    """Order data."""
    order_id: str
    client_order_id: Optional[str] = None
    ticker: str
    
    side: OrderSide
    type: OrderType
    
    # Prices in cents
    price: Optional[int] = None  # Limit price
    
    # Quantities
    quantity: int  # Original quantity
    filled_quantity: int = 0
    remaining_quantity: int = 0
    
    status: OrderStatus
    
    # Timestamps
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]
    
    @property
    def fill_percent(self) -> float:
        """Get fill percentage."""
        if self.quantity == 0:
            return 0.0
        return self.filled_quantity / self.quantity


class Position(BaseModel):
    """Portfolio position."""
    ticker: str
    market_title: Optional[str] = None
    
    # Position quantities
    yes_quantity: int = 0
    no_quantity: int = 0
    
    # Average cost in cents
    yes_avg_cost: Optional[float] = None
    no_avg_cost: Optional[float] = None
    
    # Current value
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    realized_pnl: float = 0.0
    
    @property
    def net_position(self) -> int:
        """Net position (positive = long YES, negative = long NO)."""
        return self.yes_quantity - self.no_quantity
    
    @property
    def has_position(self) -> bool:
        """Check if there's any position."""
        return self.yes_quantity > 0 or self.no_quantity > 0


class Trade(BaseModel):
    """Executed trade."""
    trade_id: str
    order_id: str
    ticker: str
    
    side: OrderSide
    price: int  # Execution price in cents
    quantity: int
    
    # Cost in cents
    total_cost: int
    
    timestamp: datetime


class PortfolioBalance(BaseModel):
    """Portfolio balance information."""
    available_balance: float  # In dollars
    total_balance: float
    pending_balance: float = 0.0
    
    @property
    def usable_balance(self) -> float:
        """Balance available for trading."""
        return self.available_balance


class ArbitrageOpportunity(BaseModel):
    """Detected arbitrage opportunity."""
    opportunity_id: str
    type: str  # "single_condition", "market_rebalancing", "combinatorial"
    
    # Markets involved
    tickers: List[str]
    event_tickers: List[str] = Field(default_factory=list)
    
    # Prices
    prices: Dict[str, float]  # ticker -> price
    
    # Calculated metrics
    theoretical_profit: float  # Maximum profit if perfect execution
    guaranteed_profit: float  # D(μ||θ) - g(μ) from Proposition 4.1
    profit_margin: float  # Profit as fraction of cost
    
    # Execution parameters
    recommended_positions: Dict[str, Dict[str, Any]]  # ticker -> {side, quantity, price}
    total_cost: float
    max_payout: float
    
    # Liquidity
    available_liquidity: float
    vwap_slippage: float
    
    # Timing
    detected_at: datetime
    expires_at: Optional[datetime] = None
    
    # Status
    is_executable: bool = True
    execution_risk_score: float = 0.0  # 0-1, lower is better
    
    @property
    def expected_profit(self) -> float:
        """Expected profit accounting for execution risk."""
        return self.guaranteed_profit * (1 - self.execution_risk_score)
