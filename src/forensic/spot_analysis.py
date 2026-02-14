"""
Step 4: Manipulation Attribution - Price-to-Beat vs Spot Lag Analysis.

Compares Kalshi price movements against BTC/USD spot charts
to determine if "longshot" price spikes happen AFTER the spot price
has already moved past the point of recovery.

This module:
1. Fetches BTC/USD 1-minute klines from Binance API (free, no auth)
2. Aligns them with Kalshi candlestick timestamps
3. Detects if Kalshi "longshot" opportunities are lagging indicators
4. Quantifies the "information advantage" of market makers
"""

import asyncio
import time
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
import structlog

from .feature_extraction import MarketFeatures, MinuteSnapshot

logger = structlog.get_logger(__name__)


@dataclass
class SpotCandle:
    """1-minute BTC/USD spot candle from Binance."""
    open_time: int          # Unix ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int         # Unix ms


@dataclass
class SpotLagResult:
    """Result of spot-lag analysis for a single market."""
    ticker: str
    
    # Did we find corresponding spot data?
    has_spot_data: bool = False
    
    # Spot price metrics during the 15-minute window
    spot_open: float = 0.0
    spot_close: float = 0.0
    spot_high: float = 0.0
    spot_low: float = 0.0
    spot_range_pct: float = 0.0  # (high - low) / open * 100
    
    # Correlation between Kalshi price changes and spot price changes
    # Positive = Kalshi follows spot (expected)
    # Low/zero = Kalshi prices disconnected from spot (suspicious)
    price_correlation: float = 0.0
    
    # Lag analysis: Does the spot price move BEFORE Kalshi adjusts?
    # Measured in minutes of detected lag
    estimated_lag_minutes: float = 0.0
    
    # The "trap" detection:
    # Did the longshot price spike happen AFTER spot already moved away?
    longshot_spike_after_spot_move: bool = False
    
    # How many cents was the Kalshi price "wrong" vs where spot suggested it should be?
    max_mispricing_cents: float = 0.0
    mispricing_minute: int = 0


@dataclass
class SpotAnalysisReport:
    """Aggregate spot-lag analysis results."""
    total_analyzed: int = 0
    with_spot_data: int = 0
    
    # Average metrics
    avg_spot_range_pct: float = 0.0
    avg_price_correlation: float = 0.0
    avg_estimated_lag: float = 0.0
    
    # Trap detection
    trap_instances: int = 0         # Markets where longshot appeared after spot moved
    trap_rate: float = 0.0          # trap_instances / total
    
    avg_max_mispricing: float = 0.0
    
    # Individual results
    results: List[SpotLagResult] = field(default_factory=list)


class SpotPriceAnalyzer:
    """
    Fetches BTC spot data and compares with Kalshi price movements.
    
    Tries multiple APIs in order:
    1. Binance klines API (most granular, but geo-restricted in US)
    2. Binance US klines API (US-accessible)
    3. Coinbase candles API (US-accessible, no auth)
    """
    
    # APIs to try in order of preference
    SPOT_APIS = [
        {
            "name": "binance",
            "url": "https://api.binance.com/api/v3/klines",
            "symbol_param": "symbol",
            "symbol": "BTCUSDT",
        },
        {
            "name": "binance_us",
            "url": "https://api.binance.us/api/v3/klines",
            "symbol_param": "symbol",
            "symbol": "BTCUSD",
        },
        {
            "name": "coinbase",
            "url": "https://api.exchange.coinbase.com/products/BTC-USD/candles",
            "format": "coinbase",
        },
    ]
    
    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._spot_cache: Dict[str, List[SpotCandle]] = {}
        self._working_api: Optional[Dict] = None  # Caches which API works
    
    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=15.0)
        return self
    
    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
    
    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Use 'async with' context manager")
        return self._client
    
    async def _fetch_binance_candles(
        self,
        api_config: Dict,
        start_ts: int,
        end_ts: int,
    ) -> List[SpotCandle]:
        """Fetch from Binance or Binance US."""
        params = {
            api_config["symbol_param"]: api_config["symbol"],
            "interval": "1m",
            "startTime": start_ts * 1000,
            "endTime": end_ts * 1000,
            "limit": 20,
        }
        
        response = await self.client.get(api_config["url"], params=params)
        
        if response.status_code != 200:
            raise RuntimeError(f"{api_config['name']} returned {response.status_code}")
        
        data = response.json()
        candles = []
        for k in data:
            candle = SpotCandle(
                open_time=k[0],
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
                close_time=k[6],
            )
            candles.append(candle)
        
        return candles
    
    async def _fetch_coinbase_candles(
        self,
        start_ts: int,
        end_ts: int,
    ) -> List[SpotCandle]:
        """Fetch from Coinbase Exchange API."""
        # Coinbase candles: GET /products/{id}/candles
        # Returns: [[time, low, high, open, close, volume], ...]
        params = {
            "start": datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat(),
            "end": datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat(),
            "granularity": 60,  # 1 minute in seconds
        }
        
        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
        response = await self.client.get(url, params=params)
        
        if response.status_code != 200:
            raise RuntimeError(f"Coinbase returned {response.status_code}")
        
        data = response.json()
        candles = []
        # Coinbase format: [time, low, high, open, close, volume]
        # Returns in reverse chronological order
        for k in sorted(data, key=lambda x: x[0]):
            candle = SpotCandle(
                open_time=k[0] * 1000,
                open=float(k[3]),
                high=float(k[2]),
                low=float(k[1]),
                close=float(k[4]),
                volume=float(k[5]),
                close_time=(k[0] + 60) * 1000,
            )
            candles.append(candle)
        
        return candles
    
    async def fetch_spot_candles(
        self,
        start_ts: int,
        end_ts: int,
    ) -> List[SpotCandle]:
        """
        Fetch BTC/USD 1-minute candles from the best available API.
        
        Tries multiple sources and caches which one works.
        """
        cache_key = f"{start_ts}_{end_ts}"
        if cache_key in self._spot_cache:
            return self._spot_cache[cache_key]
        
        # If we already know which API works, use it
        if self._working_api:
            try:
                candles = await self._try_fetch(self._working_api, start_ts, end_ts)
                if candles:
                    self._spot_cache[cache_key] = candles
                    return candles
            except Exception:
                self._working_api = None  # Reset and try all
        
        # Try each API
        for api_config in self.SPOT_APIS:
            try:
                candles = await self._try_fetch(api_config, start_ts, end_ts)
                if candles:
                    self._working_api = api_config
                    self._spot_cache[cache_key] = candles
                    logger.info("spot_api_found", api=api_config.get("name", "coinbase"))
                    return candles
            except Exception as e:
                logger.debug(
                    "spot_api_failed",
                    api=api_config.get("name", "coinbase"),
                    error=str(e),
                )
                continue
        
        return []
    
    async def _try_fetch(
        self,
        api_config: Dict,
        start_ts: int,
        end_ts: int,
    ) -> List[SpotCandle]:
        """Try to fetch from a specific API."""
        if api_config.get("format") == "coinbase":
            return await self._fetch_coinbase_candles(start_ts, end_ts)
        else:
            return await self._fetch_binance_candles(api_config, start_ts, end_ts)
    
    async def analyze_single_market(
        self,
        feat: MarketFeatures,
    ) -> SpotLagResult:
        """
        Analyze spot-lag for a single market's 15-minute window.
        """
        result = SpotLagResult(ticker=feat.ticker)
        
        if len(feat.snapshots) < 5:
            return result
        
        # Get time range from snapshots
        start_ts = feat.snapshots[0].timestamp
        end_ts = feat.snapshots[-1].timestamp
        
        if start_ts <= 0 or end_ts <= 0:
            return result
        
        # Fetch spot data
        spot_candles = await self.fetch_spot_candles(
            start_ts=start_ts - 60,  # 1 minute before
            end_ts=end_ts + 60,      # 1 minute after
        )
        
        if not spot_candles:
            return result
        
        result.has_spot_data = True
        
        # Spot price metrics
        result.spot_open = spot_candles[0].open
        result.spot_close = spot_candles[-1].close
        result.spot_high = max(c.high for c in spot_candles)
        result.spot_low = min(c.low for c in spot_candles)
        
        if result.spot_open > 0:
            result.spot_range_pct = (
                (result.spot_high - result.spot_low) / result.spot_open * 100
            )
        
        # Compute correlation between Kalshi YES price changes and spot price changes
        kalshi_changes = []
        spot_changes = []
        
        for i in range(1, min(len(feat.snapshots), len(spot_candles))):
            kalshi_delta = feat.snapshots[i].yes_mid - feat.snapshots[i-1].yes_mid
            spot_delta = spot_candles[i].close - spot_candles[i-1].close
            kalshi_changes.append(kalshi_delta)
            spot_changes.append(spot_delta)
        
        if len(kalshi_changes) >= 3:
            result.price_correlation = self._correlation(kalshi_changes, spot_changes)
        
        # Lag estimation: Check if spot moves precede Kalshi moves
        result.estimated_lag_minutes = self._estimate_lag(
            kalshi_changes, spot_changes
        )
        
        # Trap detection: Did a longshot spike happen after spot moved?
        if feat.had_longshot_opportunity and feat.longshot_side:
            result.longshot_spike_after_spot_move = self._detect_trap(
                feat, spot_candles
            )
        
        # Max mispricing
        mispricing, mispricing_min = self._max_mispricing(feat, spot_candles)
        result.max_mispricing_cents = mispricing
        result.mispricing_minute = mispricing_min
        
        return result
    
    async def analyze_all(
        self,
        features: List[MarketFeatures],
        batch_size: int = 3,
        delay: float = 1.0,
    ) -> SpotAnalysisReport:
        """Analyze spot-lag for all markets."""
        report = SpotAnalysisReport()
        
        logger.info("starting_spot_analysis", total=len(features))
        
        for i in range(0, len(features), batch_size):
            batch = features[i:i + batch_size]
            tasks = [self.analyze_single_market(f) for f in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.warning("spot_analysis_error", error=str(result))
                    continue
                report.results.append(result)
            
            if i + batch_size < len(features):
                await asyncio.sleep(delay)
        
        # Aggregate
        report.total_analyzed = len(report.results)
        with_data = [r for r in report.results if r.has_spot_data]
        report.with_spot_data = len(with_data)
        
        if with_data:
            report.avg_spot_range_pct = (
                sum(r.spot_range_pct for r in with_data) / len(with_data)
            )
            report.avg_price_correlation = (
                sum(r.price_correlation for r in with_data) / len(with_data)
            )
            report.avg_estimated_lag = (
                sum(r.estimated_lag_minutes for r in with_data) / len(with_data)
            )
            
            traps = sum(1 for r in with_data if r.longshot_spike_after_spot_move)
            report.trap_instances = traps
            report.trap_rate = traps / len(with_data) if with_data else 0
            
            mispricings = [r.max_mispricing_cents for r in with_data if r.max_mispricing_cents > 0]
            report.avg_max_mispricing = (
                sum(mispricings) / len(mispricings) if mispricings else 0.0
            )
        
        logger.info(
            "spot_analysis_complete",
            analyzed=report.total_analyzed,
            with_data=report.with_spot_data,
            trap_rate=round(report.trap_rate, 4),
            avg_lag=round(report.avg_estimated_lag, 2),
        )
        
        return report
    
    # --- Statistical Helpers ---
    
    @staticmethod
    def _correlation(x: List[float], y: List[float]) -> float:
        """Pearson correlation coefficient."""
        n = len(x)
        if n < 2:
            return 0.0
        
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)
        
        denom = (var_x * var_y) ** 0.5
        if denom == 0:
            return 0.0
        
        return cov / denom
    
    @staticmethod
    def _estimate_lag(
        kalshi_changes: List[float],
        spot_changes: List[float],
    ) -> float:
        """
        Estimate lag between spot and Kalshi price changes.
        
        Uses cross-correlation: if spot at time T correlates with
        Kalshi at time T+1, there's a 1-minute lag.
        """
        if len(kalshi_changes) < 4:
            return 0.0
        
        best_lag = 0
        best_corr = 0.0
        
        for lag in range(4):
            n = len(kalshi_changes) - lag
            if n < 3:
                continue
            
            # spot[0:n] vs kalshi[lag:lag+n]
            spot_slice = spot_changes[:n]
            kalshi_slice = kalshi_changes[lag:lag + n]
            
            if len(spot_slice) != len(kalshi_slice):
                continue
            
            mean_s = sum(spot_slice) / n
            mean_k = sum(kalshi_slice) / n
            
            cov = sum(
                (s - mean_s) * (k - mean_k)
                for s, k in zip(spot_slice, kalshi_slice)
            )
            var_s = sum((s - mean_s) ** 2 for s in spot_slice)
            var_k = sum((k - mean_k) ** 2 for k in kalshi_slice)
            
            denom = (var_s * var_k) ** 0.5
            corr = cov / denom if denom > 0 else 0.0
            
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
        
        return float(best_lag)
    
    @staticmethod
    def _detect_trap(
        feat: MarketFeatures,
        spot_candles: List[SpotCandle],
    ) -> bool:
        """
        Detect if a longshot price spike happened AFTER the spot price
        already moved past the point of recovery.
        
        A "trap" is when:
        1. BTC spot moves significantly in one direction
        2. This causes the Kalshi "losing" side price to drop (payout rises)
        3. But the spot move has ALREADY happened - the Kalshi price is
           just catching up, not reflecting new opportunity
        """
        if not feat.had_longshot_opportunity:
            return False
        
        if len(spot_candles) < 5 or len(feat.snapshots) < 5:
            return False
        
        # Find the minute when the longshot peak occurred
        if feat.longshot_side == "yes":
            peak_min = feat.max_yes_payout_minute
        else:
            peak_min = feat.max_no_payout_minute
        
        if peak_min < 2 or peak_min >= len(spot_candles):
            return False
        
        # Check: did the spot price move significantly BEFORE the peak?
        # Look at spot movement from start to (peak_min - 1)
        if peak_min - 1 >= len(spot_candles):
            return False
        
        pre_peak_spot_change = abs(
            spot_candles[min(peak_min - 1, len(spot_candles) - 1)].close
            - spot_candles[0].open
        )
        total_spot_change = abs(
            spot_candles[-1].close - spot_candles[0].open
        )
        
        if total_spot_change == 0:
            return False
        
        # If >70% of the spot move happened BEFORE the Kalshi peak,
        # it's likely a lagging indicator (trap)
        pre_peak_ratio = pre_peak_spot_change / total_spot_change
        
        return pre_peak_ratio > 0.7
    
    @staticmethod
    def _max_mispricing(
        feat: MarketFeatures,
        spot_candles: List[SpotCandle],
    ) -> Tuple[float, int]:
        """
        Estimate maximum mispricing by comparing Kalshi price direction
        with spot direction.
        
        Returns (max_mispricing_cents, minute_of_max_mispricing).
        """
        if not spot_candles or len(feat.snapshots) < 3:
            return 0.0, 0
        
        max_mispricing = 0.0
        max_min = 0
        
        # Simple heuristic: if spot moved up but Kalshi YES price went down
        # (or vice versa), that's potential mispricing
        for i in range(1, min(len(feat.snapshots), len(spot_candles))):
            spot_pct = (
                (spot_candles[i].close - spot_candles[i-1].close)
                / spot_candles[i-1].close * 100
                if spot_candles[i-1].close > 0 else 0
            )
            
            kalshi_change = feat.snapshots[i].yes_mid - feat.snapshots[i-1].yes_mid
            
            # If spot moved up but Kalshi YES went down, or vice versa
            if (spot_pct > 0.01 and kalshi_change < -1) or (spot_pct < -0.01 and kalshi_change > 1):
                mispricing = abs(kalshi_change)
                if mispricing > max_mispricing:
                    max_mispricing = mispricing
                    max_min = i
        
        return max_mispricing, max_min
