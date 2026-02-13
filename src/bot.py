"""
Main Kalshi Arbitrage Bot.

Orchestrates:
- Market data ingestion
- Arbitrage detection
- Trade execution
- Risk management
- Monitoring
"""

import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import signal

import structlog

from .config import Settings, get_settings
from .api import KalshiClient, KalshiAuth, KalshiWebSocket
from .api.models import Market, Event, OrderBook, ArbitrageOpportunity
from .arbitrage import ArbitrageDetector
from .execution import ExecutionEngine, RiskManager, PositionSizer
from .execution.risk import RiskLimits

logger = structlog.get_logger(__name__)


class KalshiArbitrageBot:
    """
    Main arbitrage bot orchestrator.
    
    Lifecycle:
    1. Initialize connections and components
    2. Load market data
    3. Subscribe to real-time updates
    4. Continuously scan for arbitrage
    5. Execute profitable opportunities
    6. Monitor and report performance
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the bot.
        
        Args:
            settings: Application settings (uses global if not provided)
        """
        self.settings = settings or get_settings()
        
        # Components (initialized in start())
        self.auth: Optional[KalshiAuth] = None
        self.client: Optional[KalshiClient] = None
        self.websocket: Optional[KalshiWebSocket] = None
        self.detector: Optional[ArbitrageDetector] = None
        self.executor: Optional[ExecutionEngine] = None
        self.risk_manager: Optional[RiskManager] = None
        
        # State
        self.markets: Dict[str, Market] = {}
        self.events: Dict[str, Event] = {}
        self.orderbooks: Dict[str, OrderBook] = {}
        
        # Control
        self._running = False
        self._scan_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.scan_count = 0
        self.opportunities_found = 0
        self.trades_executed = 0
        self.start_time: Optional[datetime] = None
        
        logger.info(
            "bot_initialized",
            environment=self.settings.environment.value,
        )
    
    async def start(self):
        """Start the bot."""
        logger.info("bot_starting")
        self.start_time = datetime.utcnow()
        self._running = True
        
        # Initialize authentication
        self.auth = KalshiAuth(
            api_key_id=self.settings.kalshi_api_key_id,
            private_key_path=self.settings.kalshi_private_key_path,
        )
        
        # Initialize API client
        self.client = KalshiClient(
            auth=self.auth,
            base_url=self.settings.api_base_url,
            requests_per_minute=self.settings.api_requests_per_minute,
        )
        
        # Initialize risk management
        self.risk_manager = RiskManager(RiskLimits(
            max_drawdown=self.settings.max_drawdown_percent / 100,
            min_profit_threshold=self.settings.min_profit_threshold,
        ))
        
        position_sizer = PositionSizer(
            kelly_fraction=0.5,
            max_position_fraction=self.settings.max_position_size_fraction,
        )
        
        # Initialize detector
        self.detector = ArbitrageDetector(
            settings=self.settings,
            ip_solver_type=self.settings.ip_solver,
        )
        
        async with self.client:
            # Initialize executor
            self.executor = ExecutionEngine(
                client=self.client,
                risk_manager=self.risk_manager,
                position_sizer=position_sizer,
            )
            
            # Load initial market data
            await self._load_markets()
            
            # Get initial balance
            balance = await self.client.get_balance()
            self.risk_manager.set_starting_balance(balance.total_balance)
            logger.info("initial_balance", balance=balance.total_balance)
            
            # Initialize WebSocket
            self.websocket = KalshiWebSocket(
                auth=self.auth,
                ws_url=self.settings.ws_base_url,
                on_orderbook_update=self._on_orderbook_update,
            )
            
            # Start WebSocket in background
            ws_task = asyncio.create_task(self.websocket.connect())
            
            # Wait for connection
            await asyncio.sleep(2)
            
            # Subscribe to markets
            await self._subscribe_to_markets()
            
            # Start main loop
            self._scan_task = asyncio.create_task(self._main_loop())
            
            try:
                await self._scan_task
            except asyncio.CancelledError:
                logger.info("bot_cancelled")
            finally:
                await self.websocket.disconnect()
                ws_task.cancel()
    
    async def stop(self):
        """Stop the bot gracefully."""
        logger.info("bot_stopping")
        self._running = False
        
        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
        
        if self.websocket:
            await self.websocket.disconnect()
        
        # Log final stats
        self._log_stats()
    
    async def _load_markets(self):
        """Load open markets with activity (optimized to avoid fetching all markets)."""
        logger.info("loading_markets")
        
        # Load events first - they contain markets with actual activity
        events, _ = await self.client.get_events(status="open", with_nested_markets=True, limit=200)
        for event in events:
            self.events[event.event_ticker] = event
            # Add markets from events
            for market in event.markets:
                self.markets[market.ticker] = market
        
        # If we didn't get enough markets from events, fetch some directly
        if len(self.markets) < 100:
            markets, _ = await self.client.get_markets(status="open", limit=200)
            for market in markets:
                if market.ticker not in self.markets:
                    self.markets[market.ticker] = market
        
        logger.info(
            "markets_loaded",
            n_markets=len(self.markets),
            n_events=len(self.events),
        )
    
    async def _subscribe_to_markets(self):
        """Subscribe to order book updates for active markets."""
        # Subscribe to top markets by volume
        sorted_markets = sorted(
            self.markets.values(),
            key=lambda m: m.volume_24h,
            reverse=True
        )
        
        # Limit subscriptions to avoid overwhelming - start with fewer markets
        markets_to_track = sorted_markets[:30]
        
        logger.info("subscribing_to_markets", count=len(markets_to_track))
        
        tickers = [m.ticker for m in markets_to_track]
        await self.websocket.subscribe_markets(tickers)
        
        # Also fetch initial orderbooks
        self.orderbooks = await self.client.get_orderbooks_parallel(tickers)
    
    def _on_orderbook_update(self, ticker: str, orderbook: OrderBook):
        """Handle real-time orderbook updates."""
        self.orderbooks[ticker] = orderbook
    
    async def _main_loop(self):
        """Main scanning and trading loop."""
        logger.info("main_loop_started")
        
        scan_interval = 1.0  # Scan every second
        
        while self._running:
            try:
                await self._scan_and_execute()
                await asyncio.sleep(scan_interval)
            except Exception as e:
                logger.error("main_loop_error", error=str(e))
                await asyncio.sleep(5)  # Back off on error
    
    async def _scan_and_execute(self):
        """Single scan and execute cycle."""
        self.scan_count += 1
        
        # Quick scan for opportunities
        quick_results = self.detector.quick_scan(
            list(self.markets.values()),
            self.orderbooks,
        )
        
        if not quick_results:
            return
        
        # Found potential arbitrage, do full scan
        opportunities = await self.detector.scan_all(
            list(self.markets.values()),
            self.orderbooks,
            list(self.events.values()),
        )
        
        if not opportunities:
            return
        
        self.opportunities_found += len(opportunities)
        
        logger.info(
            "opportunities_detected",
            count=len(opportunities),
            best_profit=opportunities[0].guaranteed_profit,
        )
        
        # Execute top opportunities
        balance = await self.client.get_balance()
        positions = await self.client.get_positions()
        
        # Execute best opportunity
        best = opportunities[0]
        
        if best.guaranteed_profit >= self.settings.min_profit_threshold:
            result = await self.executor.execute(best, balance, positions)
            
            if result.success:
                self.trades_executed += 1
                logger.info(
                    "trade_executed",
                    opportunity_id=best.opportunity_id,
                    profit=result.realized_pnl,
                )
    
    def _log_stats(self):
        """Log bot statistics."""
        uptime = datetime.utcnow() - self.start_time if self.start_time else timedelta(0)
        
        stats = {
            "uptime_hours": uptime.total_seconds() / 3600,
            "scan_count": self.scan_count,
            "opportunities_found": self.opportunities_found,
            "trades_executed": self.trades_executed,
        }
        if self.executor:
            stats.update(self.executor.stats)
        
        logger.info("bot_stats", **stats)
    
    @property
    def status(self) -> Dict[str, Any]:
        """Get current bot status."""
        return {
            "running": self._running,
            "environment": self.settings.environment.value,
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0,
            "markets_tracked": len(self.markets),
            "orderbooks_live": len(self.orderbooks),
            "scan_count": self.scan_count,
            "opportunities_found": self.opportunities_found,
            "trades_executed": self.trades_executed,
            "websocket": self.websocket.stats if self.websocket else {},
            "executor": self.executor.stats if self.executor else {},
        }


async def main():
    """Entry point for running the bot."""
    import sys
    
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Create and run bot
    bot = KalshiArbitrageBot()
    
    # Handle shutdown signals
    def signal_handler():
        asyncio.create_task(bot.stop())
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass  # Windows doesn't support add_signal_handler
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
