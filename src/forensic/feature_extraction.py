"""
Step 2: Feature Extraction - Extract forensic features from each 15-minute window.

For every settled BTC-15M market, we extract:
- The Entry (T-14m): Price of YES/NO at the very start
- The Mid-Point (T-7m): Price exactly halfway through
- The Peak Payout: Maximum payout reached for the losing side during the window
- The Settlement: Final outcome (0 or 100)
- Minute-by-minute snapshots for the "Point of No Return" analysis
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import structlog

from .data_acquisition import SettledMarket, Candlestick

logger = structlog.get_logger(__name__)


@dataclass
class MinuteSnapshot:
    """Price snapshot at a specific minute within the 15-minute window."""
    minute: int                 # 0 = start, 14 = end
    timestamp: int              # Unix timestamp
    yes_bid: int                # Cents
    yes_ask: int                # Cents
    yes_mid: float              # Midpoint price
    no_mid: float               # 100 - yes_mid (the "other side" price)
    volume: int                 # Volume at this candle
    
    @property
    def yes_payout(self) -> float:
        """Payout multiplier if you buy YES at this price and it settles to 100.
        E.g., buying at $0.20 (20 cents) pays $1.00 = 5x payout."""
        if self.yes_mid <= 0:
            return float('inf')
        return 100.0 / self.yes_mid
    
    @property
    def no_payout(self) -> float:
        """Payout multiplier if you buy NO at this price and YES settles to 0."""
        if self.no_mid <= 0:
            return float('inf')
        return 100.0 / self.no_mid


@dataclass
class MarketFeatures:
    """Extracted features for a single 15-minute market window."""
    ticker: str
    result: str                 # "yes" or "no"
    result_binary: int          # 1 = yes won, 0 = no won
    
    # Window metadata
    total_candles: int
    total_volume: int
    
    # --- T-0 (Entry) ---
    entry_yes_mid: float        # YES midpoint at T=0
    entry_no_mid: float         # NO midpoint at T=0
    entry_yes_payout: float     # Payout multiplier for YES at entry
    entry_no_payout: float      # Payout multiplier for NO at entry
    
    # --- T-7m (Midpoint) ---
    mid_yes_mid: Optional[float] = None
    mid_no_mid: Optional[float] = None
    mid_yes_payout: Optional[float] = None
    mid_no_payout: Optional[float] = None
    
    # --- T-10m (The "Golden Minute" test point) ---
    t10_yes_mid: Optional[float] = None
    t10_no_mid: Optional[float] = None
    t10_yes_payout: Optional[float] = None
    t10_no_payout: Optional[float] = None
    
    # --- T-12m (Late-stage test) ---
    t12_yes_mid: Optional[float] = None
    t12_no_mid: Optional[float] = None
    t12_yes_payout: Optional[float] = None
    t12_no_payout: Optional[float] = None
    
    # --- Peak Payout (for the LOSING side) ---
    # The maximum payout the losing side reached during the window.
    # This is the "trap" - how high the payout looked before collapsing.
    losing_side_peak_payout: float = 0.0
    losing_side_peak_minute: int = 0
    losing_side_peak_price: float = 0.0     # Price in cents at peak
    
    # --- Peak Payout (for ANY side reaching >2x) ---
    max_yes_payout_seen: float = 0.0
    max_no_payout_seen: float = 0.0
    max_yes_payout_minute: int = 0
    max_no_payout_minute: int = 0
    
    # --- Volatility ---
    yes_price_range: float = 0.0            # Max - Min of YES mid
    price_std: float = 0.0                  # Std dev of YES mid prices
    
    # --- Full minute-by-minute data ---
    snapshots: List[MinuteSnapshot] = field(default_factory=list)
    
    # --- Derived flags ---
    had_longshot_opportunity: bool = False   # Did any side reach >2x payout?
    longshot_side: Optional[str] = None      # "yes" or "no" - which side was longshot
    longshot_won: bool = False               # Did the longshot actually win?


def extract_features(market: SettledMarket) -> Optional[MarketFeatures]:
    """
    Extract forensic features from a single settled market.
    
    Returns None if insufficient data.
    """
    if not market.result or not market.candlesticks:
        return None
    
    candles = sorted(market.candlesticks, key=lambda c: c.end_period_ts)
    
    if len(candles) < 3:
        logger.debug("insufficient_candles", ticker=market.ticker, count=len(candles))
        return None
    
    # Build minute snapshots
    snapshots: List[MinuteSnapshot] = []
    for i, candle in enumerate(candles):
        yes_bid = candle.yes_bid or 0
        yes_ask = candle.yes_ask or 100
        
        # Handle edge cases
        if yes_bid == 0 and yes_ask == 0:
            yes_mid = 0.0
        elif yes_bid == 0:
            yes_mid = yes_ask / 2.0
        else:
            yes_mid = (yes_bid + yes_ask) / 2.0
        
        no_mid = 100.0 - yes_mid
        
        snap = MinuteSnapshot(
            minute=i,
            timestamp=candle.end_period_ts,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            yes_mid=yes_mid,
            no_mid=no_mid,
            volume=candle.volume or 0,
        )
        snapshots.append(snap)
    
    # Result
    result = market.result.lower()
    result_binary = 1 if result == "yes" else 0
    
    # --- Entry (T=0) ---
    entry = snapshots[0]
    
    # --- Mid-Point (T~7) ---
    mid_idx = len(snapshots) // 2
    mid = snapshots[mid_idx] if mid_idx < len(snapshots) else None
    
    # --- T=10 (Golden Minute test) ---
    t10 = snapshots[10] if len(snapshots) > 10 else None
    
    # --- T=12 (Late stage) ---
    t12 = snapshots[12] if len(snapshots) > 12 else None
    
    # --- Peak payouts ---
    max_yes_payout = 0.0
    max_no_payout = 0.0
    max_yes_payout_min = 0
    max_no_payout_min = 0
    
    yes_prices = []
    
    for snap in snapshots:
        yes_prices.append(snap.yes_mid)
        
        if snap.yes_mid > 0:
            yes_pay = 100.0 / snap.yes_mid
            if yes_pay > max_yes_payout:
                max_yes_payout = yes_pay
                max_yes_payout_min = snap.minute
        
        if snap.no_mid > 0:
            no_pay = 100.0 / snap.no_mid
            if no_pay > max_no_payout:
                max_no_payout = no_pay
                max_no_payout_min = snap.minute
    
    # Losing side peak
    if result == "yes":
        # NO was the losing side
        losing_peak_payout = max_no_payout
        losing_peak_minute = max_no_payout_min
        losing_peak_price = snapshots[max_no_payout_min].no_mid if max_no_payout_min < len(snapshots) else 0
    else:
        # YES was the losing side
        losing_peak_payout = max_yes_payout
        losing_peak_minute = max_yes_payout_min
        losing_peak_price = snapshots[max_yes_payout_min].yes_mid if max_yes_payout_min < len(snapshots) else 0
    
    # Volatility
    if yes_prices:
        price_range = max(yes_prices) - min(yes_prices)
        mean_price = sum(yes_prices) / len(yes_prices)
        variance = sum((p - mean_price) ** 2 for p in yes_prices) / len(yes_prices)
        price_std = variance ** 0.5
    else:
        price_range = 0.0
        price_std = 0.0
    
    # Longshot detection
    # A "longshot" is any side that traded at >2x payout (price < 50 cents)
    had_longshot = max_yes_payout > 2.0 or max_no_payout > 2.0
    
    if had_longshot:
        if max_yes_payout >= max_no_payout:
            longshot_side = "yes"
            longshot_won = (result == "yes")
        else:
            longshot_side = "no"
            longshot_won = (result == "no")
    else:
        longshot_side = None
        longshot_won = False
    
    # Total volume
    total_volume = sum(s.volume for s in snapshots)
    
    features = MarketFeatures(
        ticker=market.ticker,
        result=result,
        result_binary=result_binary,
        total_candles=len(candles),
        total_volume=total_volume,
        
        # Entry
        entry_yes_mid=entry.yes_mid,
        entry_no_mid=entry.no_mid,
        entry_yes_payout=entry.yes_payout,
        entry_no_payout=entry.no_payout,
        
        # Mid
        mid_yes_mid=mid.yes_mid if mid else None,
        mid_no_mid=mid.no_mid if mid else None,
        mid_yes_payout=mid.yes_payout if mid else None,
        mid_no_payout=mid.no_payout if mid else None,
        
        # T=10
        t10_yes_mid=t10.yes_mid if t10 else None,
        t10_no_mid=t10.no_mid if t10 else None,
        t10_yes_payout=t10.yes_payout if t10 else None,
        t10_no_payout=t10.no_payout if t10 else None,
        
        # T=12
        t12_yes_mid=t12.yes_mid if t12 else None,
        t12_no_mid=t12.no_mid if t12 else None,
        t12_yes_payout=t12.yes_payout if t12 else None,
        t12_no_payout=t12.no_payout if t12 else None,
        
        # Peaks
        losing_side_peak_payout=losing_peak_payout,
        losing_side_peak_minute=losing_peak_minute,
        losing_side_peak_price=losing_peak_price,
        max_yes_payout_seen=max_yes_payout,
        max_no_payout_seen=max_no_payout,
        max_yes_payout_minute=max_yes_payout_min,
        max_no_payout_minute=max_no_payout_min,
        
        # Volatility
        yes_price_range=price_range,
        price_std=price_std,
        
        # Snapshots
        snapshots=snapshots,
        
        # Flags
        had_longshot_opportunity=had_longshot,
        longshot_side=longshot_side,
        longshot_won=longshot_won,
    )
    
    return features


def extract_all_features(markets: List[SettledMarket]) -> List[MarketFeatures]:
    """Extract features from all settled markets."""
    features = []
    skipped = 0
    
    for market in markets:
        f = extract_features(market)
        if f:
            features.append(f)
        else:
            skipped += 1
    
    logger.info(
        "feature_extraction_complete",
        extracted=len(features),
        skipped=skipped,
    )
    
    return features
