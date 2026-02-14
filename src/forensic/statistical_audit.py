"""
Step 3: Statistical Audit - The Bias Test.

Performs:
- Bracket Analysis: Group trades by their mid-window price
- Actual vs Implied: Calculate realized win rate vs market-implied probability
- The "27/29" Validation: Search for sequences where cheaper side wins >90%
- Point of No Return: Find the minute where win probability drops to 0%
- Golden Minute Analysis: What happens at T=10 with >4x payouts?
- Expected Value ($EV$) calculation for longshot bets
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import math
import structlog

from .feature_extraction import MarketFeatures, MinuteSnapshot

logger = structlog.get_logger(__name__)


# ============================================================================
# Data Structures for Audit Results
# ============================================================================

@dataclass
class PriceBucket:
    """Analysis bucket for a price range."""
    label: str
    price_min: float            # Min price in cents
    price_max: float            # Max price in cents  
    implied_win_rate: float     # What the price implies (midpoint of range / 100)
    payout_range: str           # e.g., "2x-3x"
    
    # Observed data
    total_contracts: int = 0
    wins: int = 0
    losses: int = 0
    
    @property
    def realized_win_rate(self) -> float:
        if self.total_contracts == 0:
            return 0.0
        return self.wins / self.total_contracts
    
    @property
    def edge(self) -> float:
        """
        Negative edge = market is overpriced (bad for buyers).
        Positive edge = market is underpriced (good for buyers).
        
        edge = realized_win_rate - implied_win_rate
        """
        return self.realized_win_rate - self.implied_win_rate
    
    @property
    def expected_value_per_dollar(self) -> float:
        """
        EV per dollar bet.
        
        If you buy at the midpoint price:
        EV = (win_rate * payout) - (loss_rate * cost)
        
        For a $1 contract at price P cents:
        Cost = P/100 dollars
        Win payout = $1.00
        EV = win_rate * $1.00 - (1 - win_rate) * 0 - cost
        EV = win_rate * $1.00 - P/100
        
        Per dollar invested: EV / (P/100) = (win_rate * 100 / P) - 1
        """
        avg_price = (self.price_min + self.price_max) / 2.0
        if avg_price <= 0:
            return 0.0
        # Return per $1 invested
        return (self.realized_win_rate * 100.0 / avg_price) - 1.0
    
    @property
    def is_negative_edge(self) -> bool:
        return self.edge < -0.02  # More than 2% worse than implied


@dataclass
class MinuteAnalysis:
    """Win rate analysis for a specific minute in the window."""
    minute: int
    
    # For contracts priced below various thresholds at this minute
    # Key: price threshold (e.g., 15, 20, 25, 33, 50)
    # Value: (wins, total, win_rate)
    threshold_results: Dict[int, Tuple[int, int, float]] = field(default_factory=dict)


@dataclass
class PointOfNoReturn:
    """The minute and price where win probability effectively drops to 0%."""
    minute: int
    price_threshold: float      # Below this price at this minute = death
    observed_wins: int
    observed_total: int
    win_rate: float
    confidence: str             # "definitive", "strong", "moderate", "weak"


@dataclass 
class RollingSequence:
    """A rolling window sequence for the 27/29 validation."""
    start_index: int
    end_index: int
    window_size: int
    favorite_wins: int
    underdog_wins: int
    favorite_win_rate: float
    tickers: List[str]


@dataclass
class ForensicAuditReport:
    """Complete audit report."""
    # Summary stats
    total_markets_analyzed: int
    markets_with_longshots: int
    
    # Bracket analysis
    price_buckets: List[PriceBucket]
    
    # Minute-by-minute analysis
    minute_analysis: List[MinuteAnalysis]
    
    # Point of No Return
    points_of_no_return: List[PointOfNoReturn]
    
    # Golden Minute (T=10, >4x payout)
    golden_minute_total: int = 0
    golden_minute_wins: int = 0
    golden_minute_win_rate: float = 0.0
    golden_minute_implied_rate: float = 0.0
    
    # The 27/29 sequences
    worst_sequences: List[RollingSequence] = field(default_factory=list)
    
    # Overall EV
    longshot_ev_per_dollar: float = 0.0     # EV of all >2x payout bets
    overall_favorite_win_rate: float = 0.0   # How often the favorite wins
    
    # Time decay analysis
    # At each minute: average price of eventual losers
    loser_avg_price_by_minute: Dict[int, float] = field(default_factory=dict)


# ============================================================================
# Audit Engine
# ============================================================================

class ForensicAuditor:
    """Performs the statistical audit on extracted features."""
    
    # Price buckets for bracket analysis
    BUCKET_DEFS = [
        # (label, min_cents, max_cents, implied_rate, payout_desc)
        ("$0.01-$0.05 (20x-100x)", 1, 5, 0.03, "20x-100x"),
        ("$0.06-$0.10 (10x-16x)", 6, 10, 0.08, "10x-16x"),
        ("$0.11-$0.15 (6.7x-9x)", 11, 15, 0.13, "6.7x-9x"),
        ("$0.16-$0.20 (5x-6.25x)", 16, 20, 0.18, "5x-6.25x"),
        ("$0.21-$0.25 (4x-4.8x)", 21, 25, 0.23, "4x-4.8x"),
        ("$0.26-$0.33 (3x-3.8x)", 26, 33, 0.295, "3x-3.8x"),
        ("$0.34-$0.50 (2x-2.9x)", 34, 50, 0.42, "2x-2.9x"),
        ("$0.51-$0.65 (1.5x-2x)", 51, 65, 0.58, "1.5x-2x"),
        ("$0.66-$0.80 (1.25x-1.5x)", 66, 80, 0.73, "1.25x-1.5x"),
        ("$0.81-$0.99 (1x-1.23x)", 81, 99, 0.90, "1x-1.23x"),
    ]
    
    def __init__(self, features: List[MarketFeatures]):
        self.features = features
        self.report: Optional[ForensicAuditReport] = None
    
    def run_full_audit(self) -> ForensicAuditReport:
        """Run the complete forensic audit."""
        logger.info("starting_forensic_audit", markets=len(self.features))
        
        # Step 3a: Bracket Analysis
        buckets = self._bracket_analysis()
        
        # Step 3b: Minute-by-minute analysis
        minute_data = self._minute_by_minute_analysis()
        
        # Step 3c: Point of No Return
        ponr = self._find_points_of_no_return(minute_data)
        
        # Step 3d: Golden Minute test
        gm_total, gm_wins, gm_rate, gm_implied = self._golden_minute_test()
        
        # Step 3e: The 27/29 validation
        sequences = self._find_worst_sequences(window_size=30)
        
        # Step 3f: Overall EV calculation
        longshot_ev = self._calculate_longshot_ev()
        
        # Step 3g: Favorite win rate
        fav_rate = self._overall_favorite_win_rate()
        
        # Step 3h: Time decay / loser price trajectory
        loser_prices = self._loser_price_trajectory()
        
        self.report = ForensicAuditReport(
            total_markets_analyzed=len(self.features),
            markets_with_longshots=sum(1 for f in self.features if f.had_longshot_opportunity),
            price_buckets=buckets,
            minute_analysis=minute_data,
            points_of_no_return=ponr,
            golden_minute_total=gm_total,
            golden_minute_wins=gm_wins,
            golden_minute_win_rate=gm_rate,
            golden_minute_implied_rate=gm_implied,
            worst_sequences=sequences,
            longshot_ev_per_dollar=longshot_ev,
            overall_favorite_win_rate=fav_rate,
            loser_avg_price_by_minute=loser_prices,
        )
        
        logger.info("forensic_audit_complete")
        return self.report
    
    # --- Step 3a: Bracket Analysis ---
    
    def _bracket_analysis(self) -> List[PriceBucket]:
        """
        Group all longshot observations by price bucket.
        
        For each market, we look at ALL minute snapshots where a side's price
        fell into a longshot range (< 50 cents = >2x payout), and record
        whether that side eventually won.
        """
        buckets = [
            PriceBucket(
                label=label,
                price_min=pmin,
                price_max=pmax,
                implied_win_rate=imp,
                payout_range=pay,
            )
            for label, pmin, pmax, imp, pay in self.BUCKET_DEFS
        ]
        
        for feat in self.features:
            # Check the mid-point price (T~7m) as the canonical "entry" for this analysis
            if feat.mid_yes_mid is not None and feat.mid_yes_mid < 50:
                self._classify_into_bucket(
                    buckets, feat.mid_yes_mid,
                    won=(feat.result == "yes"),
                )
            
            if feat.mid_no_mid is not None and feat.mid_no_mid < 50:
                self._classify_into_bucket(
                    buckets, feat.mid_no_mid,
                    won=(feat.result == "no"),
                )
            
            # Also check T=10 (the "golden minute" point)
            if feat.t10_yes_mid is not None and feat.t10_yes_mid < 50:
                self._classify_into_bucket(
                    buckets, feat.t10_yes_mid,
                    won=(feat.result == "yes"),
                )
            
            if feat.t10_no_mid is not None and feat.t10_no_mid < 50:
                self._classify_into_bucket(
                    buckets, feat.t10_no_mid,
                    won=(feat.result == "no"),
                )
        
        for b in buckets:
            if b.total_contracts > 0:
                logger.info(
                    "bucket_result",
                    label=b.label,
                    total=b.total_contracts,
                    wins=b.wins,
                    win_rate=round(b.realized_win_rate, 4),
                    implied=round(b.implied_win_rate, 4),
                    edge=round(b.edge, 4),
                    ev_per_dollar=round(b.expected_value_per_dollar, 4),
                    negative_edge=b.is_negative_edge,
                )
        
        return buckets
    
    def _classify_into_bucket(
        self,
        buckets: List[PriceBucket],
        price: float,
        won: bool,
    ):
        """Classify a single observation into the appropriate price bucket."""
        for bucket in buckets:
            if bucket.price_min <= price <= bucket.price_max:
                bucket.total_contracts += 1
                if won:
                    bucket.wins += 1
                else:
                    bucket.losses += 1
                return
    
    # --- Step 3b: Minute-by-Minute Analysis ---
    
    def _minute_by_minute_analysis(self) -> List[MinuteAnalysis]:
        """
        For each minute (0-14), calculate win rates at various price thresholds.
        
        This answers: "At minute X, if the price is below Y cents, 
        what's the actual win rate?"
        """
        thresholds = [5, 10, 15, 20, 25, 33, 50]
        analyses = []
        
        for minute in range(15):
            wins_by_threshold: Dict[int, int] = {t: 0 for t in thresholds}
            total_by_threshold: Dict[int, int] = {t: 0 for t in thresholds}
            
            for feat in self.features:
                if minute >= len(feat.snapshots):
                    continue
                
                snap = feat.snapshots[minute]
                
                # Check YES side
                for threshold in thresholds:
                    if snap.yes_mid <= threshold:
                        total_by_threshold[threshold] += 1
                        if feat.result == "yes":
                            wins_by_threshold[threshold] += 1
                    
                    # Check NO side
                    if snap.no_mid <= threshold:
                        total_by_threshold[threshold] += 1
                        if feat.result == "no":
                            wins_by_threshold[threshold] += 1
            
            analysis = MinuteAnalysis(minute=minute)
            for threshold in thresholds:
                total = total_by_threshold[threshold]
                wins = wins_by_threshold[threshold]
                rate = wins / total if total > 0 else 0.0
                analysis.threshold_results[threshold] = (wins, total, rate)
            
            analyses.append(analysis)
        
        return analyses
    
    # --- Step 3c: Point of No Return ---
    
    def _find_points_of_no_return(
        self,
        minute_data: List[MinuteAnalysis],
    ) -> List[PointOfNoReturn]:
        """
        Find the exact minute and price where win probability drops to 0%.
        
        The "Point of No Return" is defined as: the first minute M where,
        for contracts priced below threshold T, the win rate is 0% (or near 0%).
        """
        ponr_list = []
        thresholds = [5, 10, 15, 20, 25, 33]
        
        for threshold in thresholds:
            for minute_analysis in minute_data:
                minute = minute_analysis.minute
                
                if minute < 5:  # Only look at minute 5+
                    continue
                
                if threshold not in minute_analysis.threshold_results:
                    continue
                
                wins, total, rate = minute_analysis.threshold_results[threshold]
                
                if total < 3:  # Need minimum sample size
                    continue
                
                if rate <= 0.02:  # Win rate effectively zero (<2%)
                    # Determine confidence
                    if total >= 20 and rate == 0.0:
                        confidence = "definitive"
                    elif total >= 10 and rate <= 0.01:
                        confidence = "strong"
                    elif total >= 5:
                        confidence = "moderate"
                    else:
                        confidence = "weak"
                    
                    ponr = PointOfNoReturn(
                        minute=minute,
                        price_threshold=threshold,
                        observed_wins=wins,
                        observed_total=total,
                        win_rate=rate,
                        confidence=confidence,
                    )
                    ponr_list.append(ponr)
                    
                    logger.info(
                        "point_of_no_return",
                        minute=minute,
                        threshold=f"${threshold/100:.2f}",
                        wins=wins,
                        total=total,
                        win_rate=round(rate, 4),
                        confidence=confidence,
                    )
                    break  # Found the earliest PONR for this threshold
        
        return ponr_list
    
    # --- Step 3d: Golden Minute Test ---
    
    def _golden_minute_test(self) -> Tuple[int, int, float, float]:
        """
        Test the hypothesis:
        'If the payout for an outcome exceeds 4x at the 10-minute mark,
        what is the mathematical probability of it winning?'
        
        4x payout = price of 25 cents or less.
        
        Returns: (total, wins, actual_win_rate, implied_win_rate)
        """
        total = 0
        wins = 0
        implied_sum = 0.0
        
        for feat in self.features:
            if len(feat.snapshots) <= 10:
                continue
            
            snap = feat.snapshots[10]
            
            # Check YES side at T=10
            if snap.yes_mid > 0 and snap.yes_mid <= 25:  # 4x+ payout
                total += 1
                implied_sum += snap.yes_mid / 100.0
                if feat.result == "yes":
                    wins += 1
            
            # Check NO side at T=10
            if snap.no_mid > 0 and snap.no_mid <= 25:  # 4x+ payout
                total += 1
                implied_sum += snap.no_mid / 100.0
                if feat.result == "no":
                    wins += 1
        
        actual_rate = wins / total if total > 0 else 0.0
        implied_rate = implied_sum / total if total > 0 else 0.0
        
        logger.info(
            "golden_minute_test",
            total=total,
            wins=wins,
            actual_rate=round(actual_rate, 4),
            implied_rate=round(implied_rate, 4),
            edge=round(actual_rate - implied_rate, 4),
        )
        
        return total, wins, actual_rate, implied_rate
    
    # --- Step 3e: The 27/29 Validation ---
    
    def _find_worst_sequences(
        self,
        window_size: int = 30,
        threshold: float = 0.90,
    ) -> List[RollingSequence]:
        """
        Search for rolling windows where the favorite wins > threshold.
        
        The "favorite" is the side with the higher mid-window price.
        If the favorite wins 27/30 times, that's a 90% rate.
        """
        if len(self.features) < window_size:
            return []
        
        # For each market, determine if the favorite won
        favorite_won_list = []
        for feat in self.features:
            if feat.mid_yes_mid is None or feat.mid_no_mid is None:
                continue
            
            # Favorite = side with higher mid-point price
            if feat.mid_yes_mid >= feat.mid_no_mid:
                # YES is favorite
                fav_won = (feat.result == "yes")
            else:
                # NO is favorite
                fav_won = (feat.result == "no")
            
            favorite_won_list.append((feat.ticker, fav_won))
        
        # Rolling window search
        worst_sequences: List[RollingSequence] = []
        
        for i in range(len(favorite_won_list) - window_size + 1):
            window = favorite_won_list[i:i + window_size]
            fav_wins = sum(1 for _, won in window if won)
            fav_rate = fav_wins / window_size
            
            if fav_rate >= threshold:
                seq = RollingSequence(
                    start_index=i,
                    end_index=i + window_size - 1,
                    window_size=window_size,
                    favorite_wins=fav_wins,
                    underdog_wins=window_size - fav_wins,
                    favorite_win_rate=fav_rate,
                    tickers=[t for t, _ in window],
                )
                worst_sequences.append(seq)
        
        # Sort by worst (highest favorite win rate)
        worst_sequences.sort(key=lambda s: -s.favorite_win_rate)
        
        # Keep top 10
        worst_sequences = worst_sequences[:10]
        
        for seq in worst_sequences:
            logger.info(
                "worst_sequence_found",
                start=seq.start_index,
                favorite_wins=seq.favorite_wins,
                underdog_wins=seq.underdog_wins,
                rate=round(seq.favorite_win_rate, 4),
            )
        
        return worst_sequences
    
    # --- Step 3f: Overall Longshot EV ---
    
    def _calculate_longshot_ev(self) -> float:
        """
        Calculate the expected value of betting on all >2x payout opportunities.
        
        EV = sum of (payout if won - cost) / number of bets
        
        For each market where a longshot existed:
        - Cost = longshot price (e.g., $0.20)
        - Win payout = $1.00
        - Net if win = $1.00 - $0.20 = $0.80
        - Net if loss = -$0.20
        """
        total_bets = 0
        total_pnl = 0.0
        
        for feat in self.features:
            if not feat.had_longshot_opportunity or feat.longshot_side is None:
                continue
            
            # Use mid-point price as the hypothetical entry
            if feat.longshot_side == "yes" and feat.mid_yes_mid is not None:
                cost = feat.mid_yes_mid / 100.0  # Convert cents to dollars
                won = feat.longshot_won
            elif feat.longshot_side == "no" and feat.mid_no_mid is not None:
                cost = feat.mid_no_mid / 100.0
                won = feat.longshot_won
            else:
                continue
            
            if cost <= 0:
                continue
            
            total_bets += 1
            if won:
                total_pnl += (1.0 - cost)  # Win: payout - cost
            else:
                total_pnl -= cost  # Loss: lose cost
        
        if total_bets == 0:
            return 0.0
        
        ev = total_pnl / total_bets
        
        logger.info(
            "longshot_ev_calculated",
            total_bets=total_bets,
            total_pnl=round(total_pnl, 4),
            ev_per_bet=round(ev, 4),
            wins=sum(1 for f in self.features if f.had_longshot_opportunity and f.longshot_won),
        )
        
        return ev
    
    # --- Step 3g: Overall Favorite Win Rate ---
    
    def _overall_favorite_win_rate(self) -> float:
        """How often does the mid-window favorite (higher-priced side) win?"""
        total = 0
        fav_wins = 0
        
        for feat in self.features:
            if feat.mid_yes_mid is None or feat.mid_no_mid is None:
                continue
            
            total += 1
            
            if feat.mid_yes_mid >= feat.mid_no_mid:
                if feat.result == "yes":
                    fav_wins += 1
            else:
                if feat.result == "no":
                    fav_wins += 1
        
        rate = fav_wins / total if total > 0 else 0.0
        
        logger.info(
            "favorite_win_rate",
            total=total,
            favorite_wins=fav_wins,
            rate=round(rate, 4),
        )
        
        return rate
    
    # --- Step 3h: Loser Price Trajectory ---
    
    def _loser_price_trajectory(self) -> Dict[int, float]:
        """
        For contracts that eventually lost, what was their average price
        at each minute?
        
        This shows the "trap" trajectory - prices that look attractive
        but were already dead.
        """
        price_sums: Dict[int, float] = {}
        price_counts: Dict[int, int] = {}
        
        for feat in self.features:
            # Look at the losing side's price at each minute
            for snap in feat.snapshots:
                minute = snap.minute
                
                if feat.result == "yes":
                    # NO was the loser - track NO prices
                    loser_price = snap.no_mid
                else:
                    # YES was the loser - track YES prices
                    loser_price = snap.yes_mid
                
                price_sums[minute] = price_sums.get(minute, 0.0) + loser_price
                price_counts[minute] = price_counts.get(minute, 0) + 1
        
        avg_prices = {}
        for minute in sorted(price_sums.keys()):
            if price_counts[minute] > 0:
                avg_prices[minute] = price_sums[minute] / price_counts[minute]
        
        return avg_prices
