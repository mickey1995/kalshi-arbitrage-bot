"""
Step 1: Data Acquisition - Fetch settled BTC-15M markets and their candlestick data.

Uses the Kalshi API to retrieve:
- Last N settled markets from the KXBTC15M series
- 1-minute candlestick data for each 15-minute window
"""

import asyncio
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict

import httpx
import structlog

from src.api.auth import KalshiAuth

logger = structlog.get_logger(__name__)

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Candlestick:
    """
    Single 1-minute candlestick from Kalshi.
    
    The Kalshi API returns OHLC dictionaries for yes_bid, yes_ask, and price.
    We extract the 'close' value (end-of-period) for each field, plus
    the open/high/low for richer analysis.
    """
    end_period_ts: int              # Unix timestamp
    
    # YES bid (close of period, in cents 0-99)
    yes_bid_close: int = 0
    yes_bid_open: int = 0
    yes_bid_high: int = 0
    yes_bid_low: int = 0
    
    # YES ask (close of period, in cents 1-100)
    yes_ask_close: int = 0
    yes_ask_open: int = 0
    yes_ask_high: int = 0
    yes_ask_low: int = 0
    
    # Trade price (close of period, in cents)
    price_close: int = 0
    price_open: int = 0
    price_high: int = 0
    price_low: int = 0
    price_mean: int = 0
    
    volume: int = 0                 # Contracts traded
    open_interest: int = 0          # Open interest
    
    # Convenience aliases used by feature extraction
    @property
    def yes_bid(self) -> int:
        return self.yes_bid_close
    
    @property
    def yes_ask(self) -> int:
        return self.yes_ask_close
    
    @property
    def price(self) -> int:
        return self.price_close


@dataclass
class SettledMarket:
    """A settled BTC-15M market with metadata."""
    ticker: str
    event_ticker: str
    series_ticker: str
    title: str
    subtitle: Optional[str]
    status: str
    result: Optional[str]       # "yes" or "no"
    
    # Prices at settlement time (cents)
    yes_bid: Optional[int]
    yes_ask: Optional[int]
    last_price: Optional[int]
    
    # Volume & interest
    volume: int
    open_interest: int
    
    # Timestamps
    open_time: Optional[str]
    close_time: Optional[str]
    expiration_time: Optional[str]
    
    # Strike info (the "price to beat")
    floor_strike: Optional[float] = None
    cap_strike: Optional[float] = None
    
    # Candlestick data (populated later)
    candlesticks: List[Candlestick] = field(default_factory=list)


# ============================================================================
# API Client (focused on forensic data needs)
# ============================================================================

class ForensicDataClient:
    """
    Kalshi API client specialized for forensic data collection.
    
    Fetches settled markets and their candlestick histories with
    rate limiting, retries, and data caching.
    """
    
    PROD_BASE = "https://api.elections.kalshi.com/trade-api/v2"
    DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"
    
    SERIES_TICKER = "KXBTC15M"
    
    def __init__(
        self,
        auth: KalshiAuth,
        base_url: Optional[str] = None,
        requests_per_minute: int = 80,  # Conservative to avoid 429s
        cache_dir: str = "data/forensic_cache",
    ):
        self.auth = auth
        self.base_url = (base_url or self.PROD_BASE).rstrip("/")
        self.rpm_limit = requests_per_minute
        self._request_times: List[float] = []
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client: Optional[httpx.AsyncClient] = None
        
    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
    
    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Use 'async with' context manager")
        return self._client
    
    # --- Rate Limiting ---
    
    async def _rate_limit(self):
        """Simple sliding-window rate limiter."""
        now = time.time()
        # Remove requests older than 60s
        self._request_times = [t for t in self._request_times if now - t < 60]
        
        if len(self._request_times) >= self.rpm_limit:
            oldest = self._request_times[0]
            wait = 60 - (now - oldest) + 0.1
            if wait > 0:
                logger.debug("rate_limit_wait", seconds=round(wait, 1))
                await asyncio.sleep(wait)
        
        self._request_times.append(time.time())
    
    # --- HTTP Requests ---
    
    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        authenticated: bool = False,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Make an API request with rate limiting and retries."""
        await self._rate_limit()
        
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"}
        
        if authenticated:
            full_path = f"/trade-api/v2{path}"
            headers.update(self.auth.get_headers(method, full_path))
        
        for attempt in range(max_retries):
            try:
                response = await self.client.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=headers,
                )
                
                if response.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    logger.warning("rate_limited", attempt=attempt, wait=wait)
                    await asyncio.sleep(wait)
                    continue
                    
                if response.status_code >= 400:
                    error_text = response.text[:500]
                    logger.error(
                        "api_error",
                        status=response.status_code,
                        path=path,
                        body=error_text,
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    raise RuntimeError(f"API {response.status_code}: {error_text}")
                
                return response.json()
                
            except httpx.HTTPError as e:
                logger.error("http_error", error=str(e), attempt=attempt)
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                raise
    
    # --- Settled Markets ---
    
    async def fetch_settled_markets(
        self,
        series_ticker: str = SERIES_TICKER,
        count: int = 200,
    ) -> List[SettledMarket]:
        """
        Fetch the last N settled markets from the BTC-15M series.
        
        Uses pagination to collect up to `count` markets.
        """
        cache_file = self.cache_dir / f"settled_markets_{series_ticker}_{count}.json"
        
        # Check cache (valid for 1 hour)
        if cache_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if age < 3600:
                logger.info("using_cached_markets", age_minutes=round(age / 60, 1))
                data = json.loads(cache_file.read_text())
                return [self._dict_to_market(m) for m in data]
        
        logger.info("fetching_settled_markets", series=series_ticker, target=count)
        
        all_markets: List[SettledMarket] = []
        cursor = None
        page = 0
        
        while len(all_markets) < count:
            params = {
                "status": "settled",
                "series_ticker": series_ticker,
                "limit": min(200, count - len(all_markets)),
            }
            if cursor:
                params["cursor"] = cursor
            
            response = await self._request("GET", "/markets", params=params)
            
            raw_markets = response.get("markets", [])
            if not raw_markets:
                logger.info("no_more_markets", fetched=len(all_markets))
                break
            
            for m in raw_markets:
                market = SettledMarket(
                    ticker=m["ticker"],
                    event_ticker=m.get("event_ticker", ""),
                    series_ticker=m.get("series_ticker", series_ticker),
                    title=m.get("title", ""),
                    subtitle=m.get("subtitle"),
                    status=m.get("status", "settled"),
                    result=m.get("result"),
                    yes_bid=m.get("yes_bid"),
                    yes_ask=m.get("yes_ask"),
                    last_price=m.get("last_price"),
                    volume=m.get("volume", 0),
                    open_interest=m.get("open_interest", 0),
                    open_time=m.get("open_time"),
                    close_time=m.get("close_time"),
                    expiration_time=m.get("expiration_time"),
                    floor_strike=m.get("floor_strike"),
                    cap_strike=m.get("cap_strike"),
                )
                all_markets.append(market)
            
            cursor = response.get("cursor")
            page += 1
            
            if not cursor:
                break
            
            logger.info(
                "fetched_page",
                page=page,
                total=len(all_markets),
                target=count,
            )
            
            await asyncio.sleep(0.3)  # Be polite
        
        # Cache the results
        cache_data = [self._market_to_dict(m) for m in all_markets]
        cache_file.write_text(json.dumps(cache_data, indent=2, default=str))
        
        logger.info("settled_markets_fetched", count=len(all_markets))
        return all_markets
    
    # --- Candlestick Data ---
    
    async def fetch_candlesticks(
        self,
        market: SettledMarket,
        period_interval: int = 1,  # 1-minute candles
    ) -> List[Candlestick]:
        """
        Fetch 1-minute candlestick data for a settled market's window.
        
        Uses the market's open_time and close_time/expiration_time to
        determine the time range.
        """
        # Determine time range
        start_ts = self._parse_ts(market.open_time)
        end_ts = self._parse_ts(market.close_time or market.expiration_time)
        
        if not start_ts or not end_ts:
            logger.warning(
                "missing_timestamps",
                ticker=market.ticker,
                open_time=market.open_time,
                close_time=market.close_time,
            )
            return []
        
        # Check candlestick cache
        cache_key = f"candles_{market.ticker}_{period_interval}.json"
        cache_file = self.cache_dir / cache_key
        
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            return [self._parse_candlestick(c) for c in data]
        
        # The candlestick endpoint requires series_ticker in the path
        series_ticker = market.series_ticker or self.SERIES_TICKER
        
        path = f"/series/{series_ticker}/markets/{market.ticker}/candlesticks"
        params = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        }
        
        try:
            response = await self._request("GET", path, params=params)
        except Exception as e:
            logger.warning(
                "candlestick_fetch_failed",
                ticker=market.ticker,
                error=str(e),
            )
            return []
        
        raw_candles = response.get("candlesticks", [])
        
        # Cache the raw JSON (preserves full OHLC data)
        cache_file.write_text(json.dumps(raw_candles, indent=2))
        
        candles = [self._parse_candlestick(c) for c in raw_candles]
        
        return candles
    
    async def fetch_all_candlesticks(
        self,
        markets: List[SettledMarket],
        batch_size: int = 5,
        delay: float = 0.5,
    ) -> List[SettledMarket]:
        """
        Fetch candlestick data for all markets, with batching.
        
        Modifies markets in-place (populates .candlesticks).
        Returns the same list for chaining.
        """
        total = len(markets)
        logger.info("fetching_all_candlesticks", total=total, batch_size=batch_size)
        
        for i in range(0, total, batch_size):
            batch = markets[i:i + batch_size]
            tasks = [self.fetch_candlesticks(m) for m in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for market, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.warning(
                        "candlestick_batch_error",
                        ticker=market.ticker,
                        error=str(result),
                    )
                    market.candlesticks = []
                else:
                    market.candlesticks = result
            
            fetched = min(i + batch_size, total)
            if fetched % 20 == 0 or fetched == total:
                logger.info("candlestick_progress", fetched=fetched, total=total)
            
            if i + batch_size < total:
                await asyncio.sleep(delay)
        
        with_data = sum(1 for m in markets if m.candlesticks)
        logger.info(
            "candlestick_fetch_complete",
            total=total,
            with_data=with_data,
            missing=total - with_data,
        )
        
        return markets
    
    # --- Helpers ---
    
    def _parse_ts(self, ts_str: Optional[str]) -> Optional[int]:
        """Parse ISO timestamp string to Unix timestamp."""
        if not ts_str:
            return None
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            return int(dt.timestamp())
        except (ValueError, AttributeError):
            return None
    
    @staticmethod
    def _extract_ohlc(field: Any) -> Tuple[int, int, int, int]:
        """
        Extract (open, high, low, close) from a candlestick field.
        
        Handles both formats:
        - Old format: int (just a single value)
        - New format: dict with {open, high, low, close, ...}
        """
        if isinstance(field, dict):
            return (
                field.get("open", 0) or 0,
                field.get("high", 0) or 0,
                field.get("low", 0) or 0,
                field.get("close", 0) or 0,
            )
        elif isinstance(field, (int, float)):
            v = int(field)
            return (v, v, v, v)
        else:
            return (0, 0, 0, 0)
    
    def _parse_candlestick(self, c: Dict[str, Any]) -> Candlestick:
        """
        Parse a single candlestick from raw API response or cache.
        
        Handles both the OHLC dict format and flattened format.
        """
        # Check if this is already in flattened format (from old cache)
        if "yes_bid_close" in c:
            return Candlestick(
                end_period_ts=c.get("end_period_ts", 0),
                yes_bid_close=c.get("yes_bid_close", 0),
                yes_bid_open=c.get("yes_bid_open", 0),
                yes_bid_high=c.get("yes_bid_high", 0),
                yes_bid_low=c.get("yes_bid_low", 0),
                yes_ask_close=c.get("yes_ask_close", 0),
                yes_ask_open=c.get("yes_ask_open", 0),
                yes_ask_high=c.get("yes_ask_high", 0),
                yes_ask_low=c.get("yes_ask_low", 0),
                price_close=c.get("price_close", 0),
                price_open=c.get("price_open", 0),
                price_high=c.get("price_high", 0),
                price_low=c.get("price_low", 0),
                price_mean=c.get("price_mean", 0),
                volume=c.get("volume", 0),
                open_interest=c.get("open_interest", 0),
            )
        
        # Parse OHLC dict format from API
        yb_o, yb_h, yb_l, yb_c = self._extract_ohlc(c.get("yes_bid", 0))
        ya_o, ya_h, ya_l, ya_c = self._extract_ohlc(c.get("yes_ask", 0))
        p_o, p_h, p_l, p_c = self._extract_ohlc(c.get("price", 0))
        
        # Extract price mean if available
        price_field = c.get("price", {})
        price_mean = 0
        if isinstance(price_field, dict):
            price_mean = price_field.get("mean", 0) or 0
        
        return Candlestick(
            end_period_ts=c.get("end_period_ts", 0),
            yes_bid_close=yb_c,
            yes_bid_open=yb_o,
            yes_bid_high=yb_h,
            yes_bid_low=yb_l,
            yes_ask_close=ya_c,
            yes_ask_open=ya_o,
            yes_ask_high=ya_h,
            yes_ask_low=ya_l,
            price_close=p_c,
            price_open=p_o,
            price_high=p_h,
            price_low=p_l,
            price_mean=price_mean,
            volume=c.get("volume", 0) or 0,
            open_interest=c.get("open_interest", 0) or 0,
        )
    
    def _market_to_dict(self, m: SettledMarket) -> Dict[str, Any]:
        """Convert SettledMarket to dict (excluding candlesticks for cache)."""
        d = asdict(m)
        d.pop("candlesticks", None)
        return d
    
    def _dict_to_market(self, d: Dict[str, Any]) -> SettledMarket:
        """Convert dict back to SettledMarket."""
        d.pop("candlesticks", None)
        return SettledMarket(**d, candlesticks=[])
