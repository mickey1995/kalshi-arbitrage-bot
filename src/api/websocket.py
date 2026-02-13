"""
Kalshi WebSocket client for real-time market data.
"""

import asyncio
import json
from typing import Optional, Callable, Dict, Any, Set, List
from datetime import datetime
from enum import Enum

import websockets
from websockets.client import WebSocketClientProtocol
import structlog

from .auth import KalshiAuth
from .models import OrderBook, OrderBookLevel, Trade, OrderSide

logger = structlog.get_logger(__name__)


class SubscriptionType(str, Enum):
    """WebSocket subscription types."""
    ORDERBOOK_DELTA = "orderbook_delta"
    TICKER = "ticker"
    TRADE = "trade"
    FILL = "fill"
    ORDER_UPDATE = "order_update"


class KalshiWebSocket:
    """
    Kalshi WebSocket client for real-time data streaming.
    
    Supports:
    - Order book deltas
    - Ticker updates
    - Trade feed
    - Fill notifications
    - Order status updates
    """
    
    def __init__(
        self,
        auth: KalshiAuth,
        ws_url: str,
        on_orderbook_update: Optional[Callable[[str, OrderBook], None]] = None,
        on_trade: Optional[Callable[[Trade], None]] = None,
        on_fill: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_order_update: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        reconnect_delay: float = 5.0,
        max_reconnect_attempts: int = 10,
    ):
        """
        Initialize WebSocket client.
        
        Args:
            auth: KalshiAuth instance
            ws_url: WebSocket URL
            on_orderbook_update: Callback for orderbook updates
            on_trade: Callback for trades
            on_fill: Callback for fills
            on_order_update: Callback for order updates
            on_error: Callback for errors
            reconnect_delay: Delay between reconnection attempts
            max_reconnect_attempts: Maximum reconnection attempts
        """
        self.auth = auth
        self.ws_url = ws_url
        
        # Callbacks
        self.on_orderbook_update = on_orderbook_update
        self.on_trade = on_trade
        self.on_fill = on_fill
        self.on_order_update = on_order_update
        self.on_error = on_error
        
        # Connection settings
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        
        # State
        self._ws: Optional[WebSocketClientProtocol] = None
        self._running = False
        self._subscriptions: Dict[SubscriptionType, Set[str]] = {
            sub_type: set() for sub_type in SubscriptionType
        }
        self._orderbooks: Dict[str, OrderBook] = {}
        self._reconnect_count = 0
        self._message_count = 0
        self._last_heartbeat: Optional[datetime] = None
    
    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._ws is not None and self._ws.open
    
    async def connect(self):
        """Connect to WebSocket and authenticate."""
        self._running = True
        self._reconnect_count = 0
        
        while self._running and self._reconnect_count < self.max_reconnect_attempts:
            try:
                logger.info("websocket_connecting", url=self.ws_url)
                
                # Get auth headers for WebSocket handshake
                # For WebSocket, we sign the path /trade-api/ws/v2
                ws_headers = self.auth.get_headers("GET", "/trade-api/ws/v2")
                
                # Convert to list of tuples for websockets library
                extra_headers = [
                    ("KALSHI-ACCESS-KEY", ws_headers["KALSHI-ACCESS-KEY"]),
                    ("KALSHI-ACCESS-SIGNATURE", ws_headers["KALSHI-ACCESS-SIGNATURE"]),
                    ("KALSHI-ACCESS-TIMESTAMP", ws_headers["KALSHI-ACCESS-TIMESTAMP"]),
                ]
                
                self._ws = await websockets.connect(
                    self.ws_url,
                    ping_interval=30,
                    ping_timeout=10,
                    extra_headers=extra_headers,
                )
                
                logger.info("websocket_authenticated")
                self._reconnect_count = 0
                
                # Resubscribe to previous subscriptions
                await self._resubscribe()
                
                # Start message handler
                await self._handle_messages()
                
            except websockets.ConnectionClosed as e:
                logger.warning("websocket_closed", code=e.code, reason=e.reason)
                await self._handle_reconnect()
                
            except Exception as e:
                logger.error("websocket_error", error=str(e))
                if self.on_error:
                    self.on_error(e)
                await self._handle_reconnect()
    
    async def disconnect(self):
        """Disconnect from WebSocket."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("websocket_disconnected")
    
    async def _handle_reconnect(self):
        """Handle reconnection logic."""
        self._reconnect_count += 1
        if self._reconnect_count < self.max_reconnect_attempts:
            logger.info(
                "websocket_reconnecting",
                attempt=self._reconnect_count,
                delay=self.reconnect_delay,
            )
            await asyncio.sleep(self.reconnect_delay)
        else:
            logger.error("websocket_max_reconnects_exceeded")
            self._running = False
    
    async def _handle_messages(self):
        """Handle incoming WebSocket messages."""
        async for message in self._ws:
            try:
                data = json.loads(message)
                self._message_count += 1
                
                msg_type = data.get("type")
                
                if msg_type == "heartbeat":
                    self._last_heartbeat = datetime.utcnow()
                    
                elif msg_type == "orderbook_snapshot":
                    await self._handle_orderbook_snapshot(data)
                    
                elif msg_type == "orderbook_delta":
                    await self._handle_orderbook_delta(data)
                    
                elif msg_type == "trade":
                    await self._handle_trade(data)
                    
                elif msg_type == "fill":
                    if self.on_fill:
                        self.on_fill(data)
                        
                elif msg_type == "order":
                    if self.on_order_update:
                        self.on_order_update(data)
                        
                elif msg_type == "error":
                    logger.error("websocket_server_error", msg=data.get("msg"))
                    
            except json.JSONDecodeError as e:
                logger.warning("websocket_invalid_json", error=str(e))
            except Exception as e:
                logger.error("websocket_message_error", error=str(e))
    
    async def _resubscribe(self):
        """Resubscribe to all previous subscriptions after reconnect."""
        for sub_type, tickers in self._subscriptions.items():
            if tickers:
                for ticker in tickers:
                    await self._send_subscription(sub_type, ticker, subscribe=True)
    
    async def _send_subscription(
        self,
        sub_type: SubscriptionType,
        ticker: str,
        subscribe: bool = True,
    ):
        """Send subscription/unsubscription message."""
        if not self.is_connected:
            return
        
        message = {
            "type": "subscribe" if subscribe else "unsubscribe",
            "channels": [sub_type.value],
            "market_tickers": [ticker],
        }
        
        await self._ws.send(json.dumps(message))
        logger.debug(
            "subscription_sent",
            type=sub_type.value,
            ticker=ticker,
            subscribe=subscribe,
        )
    
    # ========== Subscription Methods ==========
    
    async def subscribe_orderbook(self, ticker: str):
        """Subscribe to order book updates for a market."""
        self._subscriptions[SubscriptionType.ORDERBOOK_DELTA].add(ticker)
        self._orderbooks[ticker] = OrderBook(ticker=ticker)
        await self._send_subscription(SubscriptionType.ORDERBOOK_DELTA, ticker)
    
    async def unsubscribe_orderbook(self, ticker: str):
        """Unsubscribe from order book updates."""
        self._subscriptions[SubscriptionType.ORDERBOOK_DELTA].discard(ticker)
        self._orderbooks.pop(ticker, None)
        await self._send_subscription(SubscriptionType.ORDERBOOK_DELTA, ticker, False)
    
    async def subscribe_trades(self, ticker: str):
        """Subscribe to trade feed for a market."""
        self._subscriptions[SubscriptionType.TRADE].add(ticker)
        await self._send_subscription(SubscriptionType.TRADE, ticker)
    
    async def subscribe_fills(self):
        """Subscribe to fill notifications for your orders."""
        # Fills are per-account, no ticker needed
        message = {
            "type": "subscribe",
            "channels": [SubscriptionType.FILL.value],
        }
        if self.is_connected:
            await self._ws.send(json.dumps(message))
    
    async def subscribe_order_updates(self):
        """Subscribe to order status updates."""
        message = {
            "type": "subscribe",
            "channels": [SubscriptionType.ORDER_UPDATE.value],
        }
        if self.is_connected:
            await self._ws.send(json.dumps(message))
    
    async def subscribe_markets(self, tickers: List[str]):
        """Subscribe to multiple markets at once."""
        tasks = [self.subscribe_orderbook(t) for t in tickers]
        await asyncio.gather(*tasks)
    
    # ========== Message Handlers ==========
    
    async def _handle_orderbook_snapshot(self, data: Dict[str, Any]):
        """Handle full orderbook snapshot."""
        ticker = data.get("market_ticker")
        if not ticker:
            return
        
        orderbook = OrderBook(
            ticker=ticker,
            yes_bids=[
                OrderBookLevel(price=level[0], quantity=level[1])
                for level in data.get("yes", [])
            ],
            no_bids=[
                OrderBookLevel(price=level[0], quantity=level[1])
                for level in data.get("no", [])
            ],
            timestamp=datetime.utcnow(),
        )
        
        self._orderbooks[ticker] = orderbook
        
        if self.on_orderbook_update:
            self.on_orderbook_update(ticker, orderbook)
    
    async def _handle_orderbook_delta(self, data: Dict[str, Any]):
        """Handle orderbook delta update."""
        ticker = data.get("market_ticker")
        if not ticker or ticker not in self._orderbooks:
            return
        
        orderbook = self._orderbooks[ticker]
        
        # Apply deltas
        for side in ["yes", "no"]:
            deltas = data.get(side, [])
            bids = orderbook.yes_bids if side == "yes" else orderbook.no_bids
            
            for delta in deltas:
                price, quantity = delta[0], delta[1]
                
                # Find existing level
                existing = next((b for b in bids if b.price == price), None)
                
                if quantity == 0:
                    # Remove level
                    if existing:
                        bids.remove(existing)
                elif existing:
                    # Update level
                    existing.quantity = quantity
                else:
                    # Add new level
                    bids.append(OrderBookLevel(price=price, quantity=quantity))
            
            # Sort bids by price (descending)
            bids.sort(key=lambda x: x.price, reverse=True)
        
        orderbook.timestamp = datetime.utcnow()
        
        if self.on_orderbook_update:
            self.on_orderbook_update(ticker, orderbook)
    
    async def _handle_trade(self, data: Dict[str, Any]):
        """Handle trade message."""
        if not self.on_trade:
            return
        
        trade = Trade(
            trade_id=data.get("trade_id", ""),
            order_id="",  # Not provided in trade feed
            ticker=data.get("market_ticker", ""),
            side=OrderSide(data.get("side", "yes")),
            price=data.get("price", 0),
            quantity=data.get("count", 0),
            total_cost=data.get("price", 0) * data.get("count", 0),
            timestamp=datetime.utcnow(),
        )
        
        self.on_trade(trade)
    
    # ========== State Access ==========
    
    def get_orderbook(self, ticker: str) -> Optional[OrderBook]:
        """Get current orderbook for a ticker."""
        return self._orderbooks.get(ticker)
    
    def get_all_orderbooks(self) -> Dict[str, OrderBook]:
        """Get all tracked orderbooks."""
        return self._orderbooks.copy()
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get WebSocket statistics."""
        return {
            "connected": self.is_connected,
            "message_count": self._message_count,
            "subscriptions": {
                k.value: len(v) for k, v in self._subscriptions.items()
            },
            "orderbooks_tracked": len(self._orderbooks),
            "last_heartbeat": self._last_heartbeat.isoformat() if self._last_heartbeat else None,
            "reconnect_count": self._reconnect_count,
        }
