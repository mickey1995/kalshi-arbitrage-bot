"""
=============================================================================
  KALSHI BTC-15M FAVORITE BUYER BOT
=============================================================================

  Strategy: Buy the FAVORITE side during minute 8-11 of each 15-minute
  Bitcoin window. The forensic analysis of 750+ settled markets shows
  the favorite wins 93%+ of the time in this window.

  How it works:
  1. Scans for open BTC-15M markets every few seconds
  2. When a market is between minute 8-11 of its 15-minute window:
     - Identifies which side is the FAVORITE (higher price)
     - Checks the favorite price is in the sweet spot (60-85 cents)
     - Places a BUY order on the favorite side for your fixed bet amount
  3. Contract settles to $1.00 if favorite wins (93% of the time)
     - You pay ~$0.70, get back $1.00 = ~$0.30 profit per win
     - You lose ~$0.70 on the 7% that don't work out

  CONFIGURATION:
  Edit bot_config.json to change settings. Key settings:
    - bet_amount_dollars: How much to bet each time (default $1.00)
    - dry_run: Set to false to place REAL trades (default true = no real money)

  RUN:
    python run_bot.py            (uses bot_config.json)
    python run_bot.py --live     (override dry_run, place real trades)
    python run_bot.py --bet 2.50 (override bet amount to $2.50)

=============================================================================
"""

import asyncio
import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

import httpx

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.api.auth import KalshiAuth


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BotConfig:
    """Bot configuration - edit bot_config.json to change these."""
    bet_amount_dollars: float = 1.00        # Fixed bet size per trade
    max_daily_trades: int = 100             # Max trades per day
    max_daily_loss_dollars: float = 25.00   # Stop if daily loss exceeds this
    entry_minute_min: int = 8               # Earliest minute to enter (0-14)
    entry_minute_max: int = 11              # Latest minute to enter (0-14)
    max_favorite_price_cents: int = 85      # Don't buy if favorite > 85c (too expensive, thin profit)
    min_favorite_price_cents: int = 60      # Don't buy if favorite < 60c (not enough of a favorite)
    series_ticker: str = "KXBTC15M"         # Which series to trade
    scan_interval_seconds: int = 5          # How often to check for opportunities
    dry_run: bool = True                    # True = paper trading, False = real money

    @classmethod
    def load(cls, path: str = "bot_config.json") -> "BotConfig":
        config_path = Path(path)
        if config_path.exists():
            data = json.loads(config_path.read_text())
            return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
        return cls()

    def save(self, path: str = "bot_config.json"):
        from dataclasses import asdict
        Path(path).write_text(json.dumps(asdict(self), indent=4))

    def display(self):
        mode = "DRY RUN (no real money)" if self.dry_run else "*** LIVE TRADING ***"
        print(f"""
  ---------------------------------------------------------
  BOT CONFIGURATION
  ---------------------------------------------------------
  Mode:              {mode}
  Bet Amount:        ${self.bet_amount_dollars}
  Max Daily Trades:  {self.max_daily_trades}
  Max Daily Loss:    ${self.max_daily_loss_dollars}
  Entry Window:      Minute {self.entry_minute_min}-{self.entry_minute_max} of 15-min window
  Favorite Price:    {self.min_favorite_price_cents}c - {self.max_favorite_price_cents}c
  Series:            {self.series_ticker}
  Scan Interval:     {self.scan_interval_seconds}s
  ---------------------------------------------------------""")


# =============================================================================
# TRADE TRACKER
# =============================================================================

@dataclass
class TradeRecord:
    """Record of a single trade."""
    timestamp: str
    ticker: str
    side: str           # "yes" or "no"
    action: str         # "buy"
    price_cents: int
    quantity: int
    cost_dollars: float
    order_id: str
    status: str         # "placed", "filled", "settled_win", "settled_loss"
    pnl_dollars: float = 0.0


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: str = ""
    trades_placed: int = 0
    trades_won: int = 0
    trades_lost: int = 0
    trades_pending: int = 0
    total_cost: float = 0.0
    total_revenue: float = 0.0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    tickers_traded: List[str] = field(default_factory=list)

    def update_win_rate(self):
        settled = self.trades_won + self.trades_lost
        self.win_rate = self.trades_won / settled if settled > 0 else 0.0


# =============================================================================
# API CLIENT (streamlined for bot)
# =============================================================================

class BotApiClient:
    """Minimal Kalshi API client for the trading bot."""

    PROD_BASE = "https://api.elections.kalshi.com/trade-api/v2"
    DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"

    def __init__(self, auth: KalshiAuth, base_url: str):
        self.auth = auth
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None
        self._last_request = 0.0

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=15.0)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if not self._client:
            raise RuntimeError("Use async with")
        return self._client

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        authenticated: bool = False,
    ) -> Dict:
        # Simple rate limiting: min 200ms between requests
        now = time.time()
        wait = 0.2 - (now - self._last_request)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request = time.time()

        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"}

        if authenticated:
            full_path = f"/trade-api/v2{path}"
            headers.update(self.auth.get_headers(method, full_path))

        for attempt in range(3):
            try:
                response = await self.client.request(
                    method=method, url=url, params=params,
                    json=json_data, headers=headers,
                )
                if response.status_code == 429:
                    await asyncio.sleep(2 ** (attempt + 1))
                    continue
                if response.status_code >= 400:
                    return {"error": response.text, "status": response.status_code}
                return response.json()
            except Exception as e:
                if attempt == 2:
                    return {"error": str(e)}
                await asyncio.sleep(1)
        return {"error": "max retries"}

    # --- Market Data ---

    async def get_open_markets(self, series_ticker: str) -> List[Dict]:
        """Get all open markets for a series."""
        resp = await self._request("GET", "/markets", params={
            "series_ticker": series_ticker,
            "status": "open",
            "limit": 200,
        })
        return resp.get("markets", [])

    async def get_market(self, ticker: str) -> Dict:
        """Get single market details."""
        resp = await self._request("GET", f"/markets/{ticker}")
        return resp.get("market", resp)

    async def get_orderbook(self, ticker: str) -> Dict:
        """Get orderbook for a market."""
        resp = await self._request("GET", f"/markets/{ticker}/orderbook", params={"depth": 5})
        return resp.get("orderbook", resp)

    # --- Trading ---

    async def get_balance(self) -> float:
        """Get available balance in dollars."""
        resp = await self._request("GET", "/portfolio/balance", authenticated=True)
        if "error" in resp:
            return 0.0
        cents = resp.get("balance", 0)
        return cents / 100.0

    async def place_order(
        self,
        ticker: str,
        side: str,      # "yes" or "no"
        price_cents: int,
        quantity: int,
    ) -> Dict:
        """
        Place a limit buy order.

        Args:
            ticker: Market ticker
            side: "yes" or "no"
            price_cents: Price in cents (1-99)
            quantity: Number of contracts
        """
        client_order_id = str(uuid.uuid4())

        order_data = {
            "ticker": ticker,
            "action": "buy",
            "side": side,
            "type": "limit",
            "count": quantity,
            "client_order_id": client_order_id,
        }

        if side == "yes":
            order_data["yes_price"] = price_cents
        else:
            order_data["no_price"] = price_cents

        resp = await self._request(
            "POST", "/portfolio/orders",
            json_data=order_data,
            authenticated=True,
        )

        return resp

    async def get_positions(self) -> List[Dict]:
        """Get current positions."""
        resp = await self._request("GET", "/portfolio/positions", authenticated=True)
        return resp.get("market_positions", [])


# =============================================================================
# CORE BOT LOGIC
# =============================================================================

class FavoriteBot:
    """
    The main bot. Scans BTC-15M markets and buys the favorite side
    during the optimal window (minute 8-11).
    """

    def __init__(self, config: BotConfig, api: BotApiClient):
        self.config = config
        self.api = api
        self.stats = DailyStats(date=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        self.trades: List[TradeRecord] = []
        self.traded_tickers: set = set()  # Don't trade same market twice
        self._running = False
        self._trade_log_path = Path("trade_log.json")

    def _log(self, msg: str, level: str = "INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        prefix = {"INFO": " ", "TRADE": "$", "WARN": "!", "ERROR": "X", "SKIP": "-"}
        print(f"  [{ts}] {prefix.get(level, ' ')} {msg}")

    # --- Market Analysis ---

    def _get_market_minute(self, market: Dict) -> Optional[int]:
        """
        Calculate what minute (0-14) a 15-minute market is currently at.

        Returns None if we can't determine or market isn't in our window.
        """
        open_time_str = market.get("open_time")
        close_time_str = market.get("close_time") or market.get("expiration_time")

        if not open_time_str or not close_time_str:
            return None

        try:
            open_time = datetime.fromisoformat(open_time_str.replace("Z", "+00:00"))
            close_time = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

        now = datetime.now(timezone.utc)

        if now < open_time or now > close_time:
            return None

        elapsed = (now - open_time).total_seconds()
        total_duration = (close_time - open_time).total_seconds()

        if total_duration <= 0:
            return None

        # Convert to minute (0-14 for a 15-minute window)
        minute = int(elapsed / 60)
        return minute

    def _identify_favorite(self, market: Dict) -> Optional[Tuple[str, int]]:
        """
        Identify the favorite side and its price.

        Returns: (side, price_cents) or None if can't determine.

        The favorite is the side with the higher yes_bid.
        On Kalshi:
        - yes_bid = best bid for YES
        - If yes_bid is high (e.g., 75), YES is the favorite
        - If yes_bid is low (e.g., 25), NO is the favorite (NO price = 100 - 25 = 75)
        """
        yes_bid = market.get("yes_bid")
        yes_ask = market.get("yes_ask")

        if yes_bid is None or yes_ask is None:
            return None

        # Handle the new dict format from API
        if isinstance(yes_bid, dict):
            yes_bid = yes_bid.get("close", 0) or 0
        if isinstance(yes_ask, dict):
            yes_ask = yes_ask.get("close", 0) or 0

        if not yes_bid and not yes_ask:
            return None

        # Use midpoint for analysis
        yes_mid = (yes_bid + yes_ask) / 2.0 if yes_bid and yes_ask else (yes_bid or yes_ask or 50)
        no_mid = 100.0 - yes_mid

        if yes_mid >= no_mid:
            # YES is the favorite
            # To buy YES, we pay the ask price
            buy_price = yes_ask if yes_ask else int(yes_mid) + 1
            return ("yes", int(buy_price))
        else:
            # NO is the favorite
            # NO ask = 100 - yes_bid
            no_ask = 100 - yes_bid if yes_bid else int(no_mid) + 1
            return ("no", int(no_ask))

    def _should_trade(self, market: Dict, minute: int, fav_side: str, fav_price: int) -> Tuple[bool, str]:
        """
        Decide if we should trade this market.

        Returns: (should_trade, reason)
        """
        ticker = market["ticker"]

        # Already traded this market
        if ticker in self.traded_tickers:
            return False, "already traded"

        # Check minute window
        if minute < self.config.entry_minute_min or minute > self.config.entry_minute_max:
            return False, f"minute {minute} outside window {self.config.entry_minute_min}-{self.config.entry_minute_max}"

        # Check favorite price is in sweet spot
        if fav_price > self.config.max_favorite_price_cents:
            return False, f"favorite price {fav_price}c > max {self.config.max_favorite_price_cents}c"
        if fav_price < self.config.min_favorite_price_cents:
            return False, f"favorite price {fav_price}c < min {self.config.min_favorite_price_cents}c"

        # Daily limits
        if self.stats.trades_placed >= self.config.max_daily_trades:
            return False, f"daily trade limit ({self.config.max_daily_trades}) reached"

        if self.stats.total_pnl < -self.config.max_daily_loss_dollars:
            return False, f"daily loss limit (${self.config.max_daily_loss_dollars}) reached"

        # Volume check - skip very illiquid markets
        volume = market.get("volume", 0)
        if isinstance(volume, dict):
            volume = volume.get("close", 0) or 0
        if volume < 10:
            return False, f"low volume ({volume})"

        return True, "all checks passed"

    # --- Trade Execution ---

    async def _execute_trade(self, market: Dict, side: str, price_cents: int, minute: int):
        """Execute a single trade."""
        ticker = market["ticker"]

        # Calculate quantity: fixed dollar amount / price
        cost_per_contract = price_cents / 100.0
        quantity = max(1, int(self.config.bet_amount_dollars / cost_per_contract))
        actual_cost = quantity * cost_per_contract

        title = market.get("title", ticker)

        if self.config.dry_run:
            self._log(
                f"DRY RUN - Would buy {quantity}x {side.upper()} @ {price_cents}c "
                f"on {ticker} (min {minute}) - Cost: ${actual_cost:.2f}",
                "TRADE",
            )
            order_id = f"dry-{uuid.uuid4().hex[:8]}"
        else:
            # REAL TRADE
            self._log(
                f"PLACING ORDER: {quantity}x {side.upper()} @ {price_cents}c "
                f"on {ticker} (min {minute}) - Cost: ${actual_cost:.2f}",
                "TRADE",
            )

            result = await self.api.place_order(
                ticker=ticker,
                side=side,
                price_cents=price_cents,
                quantity=quantity,
            )

            if "error" in result:
                self._log(f"ORDER FAILED: {result['error']}", "ERROR")
                return

            order = result.get("order", result)
            order_id = order.get("order_id", "unknown")
            self._log(f"ORDER PLACED: {order_id}", "TRADE")

        # Record trade
        record = TradeRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            ticker=ticker,
            side=side,
            action="buy",
            price_cents=price_cents,
            quantity=quantity,
            cost_dollars=actual_cost,
            order_id=order_id,
            status="placed",
        )
        self.trades.append(record)
        self.traded_tickers.add(ticker)

        # Update stats
        self.stats.trades_placed += 1
        self.stats.total_cost += actual_cost
        self.stats.tickers_traded.append(ticker)

        # Save trade log
        self._save_trade_log()

    # --- Main Loop ---

    async def run(self):
        """Main bot loop."""
        self._running = True
        self._log("Bot starting...", "INFO")

        # Check balance
        balance = await self.api.get_balance()
        if balance > 0:
            self._log(f"Account balance: ${balance:.2f}", "INFO")
        else:
            self._log("Could not fetch balance (may need auth)", "WARN")

        self._log(
            f"Scanning {self.config.series_ticker} every {self.config.scan_interval_seconds}s "
            f"for minute {self.config.entry_minute_min}-{self.config.entry_minute_max} entries...",
            "INFO",
        )
        self._log("Press Ctrl+C to stop", "INFO")
        print()

        scan_count = 0
        last_status_time = 0

        while self._running:
            try:
                scan_count += 1

                # Fetch open markets
                markets = await self.api.get_open_markets(self.config.series_ticker)

                if not markets:
                    if scan_count % 12 == 0:  # Every ~60 seconds
                        self._log("No open markets found. Waiting...", "INFO")
                    await asyncio.sleep(self.config.scan_interval_seconds)
                    continue

                # Evaluate each market
                for market in markets:
                    ticker = market.get("ticker", "")

                    # What minute is this market at?
                    minute = self._get_market_minute(market)
                    if minute is None:
                        continue

                    # Who is the favorite?
                    fav = self._identify_favorite(market)
                    if fav is None:
                        continue

                    fav_side, fav_price = fav

                    # Should we trade?
                    should, reason = self._should_trade(market, minute, fav_side, fav_price)

                    if should:
                        await self._execute_trade(market, fav_side, fav_price, minute)
                    elif minute >= self.config.entry_minute_min and minute <= self.config.entry_minute_max:
                        # Only log skips if we're in the entry window (less noise)
                        if reason != "already traded":
                            self._log(f"Skip {ticker} (min {minute}): {reason}", "SKIP")

                # Periodic status update (every 60 seconds)
                now = time.time()
                if now - last_status_time > 60:
                    self._print_status(markets)
                    last_status_time = now

                await asyncio.sleep(self.config.scan_interval_seconds)

            except KeyboardInterrupt:
                self._running = False
                break
            except Exception as e:
                self._log(f"Error in main loop: {e}", "ERROR")
                await asyncio.sleep(5)

        self._log("Bot stopped.", "INFO")
        self._print_final_summary()

    def stop(self):
        self._running = False

    # --- Display ---

    def _print_status(self, markets: List[Dict]):
        """Print periodic status update."""
        now = datetime.now().strftime("%H:%M:%S")
        open_count = len(markets)
        self.stats.update_win_rate()

        print()
        print(f"  -- Status @ {now} -----------------------------------------")
        print(f"  Open Markets: {open_count}  |  Trades Today: {self.stats.trades_placed}  |  "
              f"P&L: ${self.stats.total_pnl:+.2f}  |  "
              f"Cost: ${self.stats.total_cost:.2f}")

        # Show markets in the window
        in_window = []
        for m in markets:
            minute = self._get_market_minute(m)
            if minute is not None:
                fav = self._identify_favorite(m)
                fav_info = f"{fav[0].upper()} @ {fav[1]}c" if fav else "?"
                traded = " [TRADED]" if m.get("ticker", "") in self.traded_tickers else ""
                in_window.append(f"    {m.get('ticker', '?'):40} min={minute:<3} fav={fav_info}{traded}")

        if in_window:
            print("  Active Markets:")
            for line in in_window[:10]:
                print(line)
        print()

    def _print_final_summary(self):
        """Print summary when bot stops."""
        self.stats.update_win_rate()
        print()
        print("  =========================================================")
        print("  FINAL SESSION SUMMARY")
        print("  =========================================================")
        print(f"  Trades Placed:    {self.stats.trades_placed}")
        print(f"  Total Cost:       ${self.stats.total_cost:.2f}")
        print(f"  Total P&L:        ${self.stats.total_pnl:+.2f}")
        print(f"  Markets Traded:   {len(self.traded_tickers)}")

        if self.trades:
            print()
            print("  Trade Log:")
            for t in self.trades:
                print(f"    {t.timestamp[:19]}  {t.ticker:40}  "
                      f"{t.side.upper():3} @ {t.price_cents}c  "
                      f"x{t.quantity}  ${t.cost_dollars:.2f}  [{t.status}]")

        print("  =========================================================")
        print()

    def _save_trade_log(self):
        """Save trades to JSON file."""
        from dataclasses import asdict
        data = {
            "stats": asdict(self.stats),
            "trades": [asdict(t) for t in self.trades],
        }
        self._trade_log_path.write_text(json.dumps(data, indent=2))


# =============================================================================
# ENVIRONMENT LOADING
# =============================================================================

def load_env():
    """Load .env file."""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


# =============================================================================
# MAIN
# =============================================================================

async def main(config: BotConfig):
    """Main entry point."""

    print()
    print("  =========================================================")
    print("  |  KALSHI BTC-15M FAVORITE BUYER BOT                     |")
    print("  |  Strategy: Buy the favorite at minute 8-11              |")
    print("  |  Win Rate: ~93% based on 750+ settled markets           |")
    print("  =========================================================")

    config.display()

    # Load credentials
    api_key = os.environ.get("KALSHI_API_KEY_ID", "")
    key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "kalshi_private_key.pem")
    environment = os.environ.get("ENVIRONMENT", "production")

    if not api_key:
        print("\n  ERROR: KALSHI_API_KEY_ID not set in .env file")
        print("  Please add your API key to the .env file")
        return

    project_root = Path(__file__).parent
    if not Path(key_path).is_absolute():
        key_path = str(project_root / key_path)

    if not Path(key_path).exists():
        print(f"\n  ERROR: Private key not found at {key_path}")
        return

    # Setup auth
    auth = KalshiAuth(api_key_id=api_key, private_key_path=key_path)

    base_url = (
        "https://api.elections.kalshi.com/trade-api/v2"
        if environment == "production"
        else "https://demo-api.kalshi.co/trade-api/v2"
    )

    print(f"\n  Environment: {environment}")
    print(f"  API Key: {api_key[:8]}...")

    if not config.dry_run:
        print()
        print("  *** WARNING: LIVE TRADING MODE ***")
        print("  Real money will be used. Press Ctrl+C within 5 seconds to cancel.")
        try:
            await asyncio.sleep(5)
        except KeyboardInterrupt:
            print("\n  Cancelled.")
            return
        print("  Starting live trading...")
    else:
        print("\n  Running in DRY RUN mode (no real money)")
        print("  To go live: set \"dry_run\": false in bot_config.json")

    print()

    # Run the bot
    async with BotApiClient(auth=auth, base_url=base_url) as api:
        bot = FavoriteBot(config=config, api=api)

        try:
            await bot.run()
        except KeyboardInterrupt:
            bot.stop()
            print("\n  Shutting down gracefully...")


def cli():
    parser = argparse.ArgumentParser(
        description="Kalshi BTC-15M Favorite Buyer Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_bot.py                  # Dry run with default settings
  python run_bot.py --live           # Real trading (careful!)
  python run_bot.py --bet 2.50       # Set bet to $2.50
  python run_bot.py --bet 5 --live   # $5 bets, real money

To change settings permanently, edit bot_config.json
        """,
    )

    parser.add_argument("--live", action="store_true", help="Enable live trading (real money)")
    parser.add_argument("--bet", type=float, help="Override bet amount in dollars")
    parser.add_argument("--config", type=str, default="bot_config.json", help="Config file path")

    args = parser.parse_args()

    # Load env
    load_env()

    # Load config
    config = BotConfig.load(args.config)

    # Apply overrides
    if args.live:
        config.dry_run = False
    if args.bet:
        config.bet_amount_dollars = args.bet

    # Save config (so changes persist)
    config.save(args.config)

    # Run
    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        print("\n  Bot stopped.")


if __name__ == "__main__":
    cli()
