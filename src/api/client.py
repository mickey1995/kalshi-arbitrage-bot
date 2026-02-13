"""
Kalshi REST API Client.
"""

import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

import httpx
import structlog
from asyncio_throttle import Throttler

from .auth import KalshiAuth
from .models import (
    Market,
    Event,
    Order,
    OrderBook,
    OrderBookLevel,
    Position,
    Trade,
    PortfolioBalance,
    OrderSide,
    OrderType,
    OrderStatus,
    MarketStatus,
)

logger = structlog.get_logger(__name__)


class KalshiClientError(Exception):
    """Base exception for Kalshi client errors."""
    pass


class KalshiAuthError(KalshiClientError):
    """Authentication error."""
    pass


class KalshiRateLimitError(KalshiClientError):
    """Rate limit exceeded."""
    pass


class KalshiClient:
    """
    Asynchronous Kalshi REST API client.
    
    Handles all REST API operations including:
    - Market data retrieval
    - Order management
    - Portfolio operations
    - Authentication
    """
    
    def __init__(
        self,
        auth: KalshiAuth,
        base_url: str,
        requests_per_minute: int = 100,
    ):
        """
        Initialize the Kalshi client.
        
        Args:
            auth: KalshiAuth instance for authentication
            base_url: API base URL
            requests_per_minute: Rate limit for API requests
        """
        self.auth = auth
        self.base_url = base_url.rstrip('/')
        self.throttler = Throttler(rate_limit=requests_per_minute, period=60)
        self._client: Optional[httpx.AsyncClient] = None
        
        logger.info(
            "kalshi_client_initialized",
            base_url=base_url,
            rate_limit=requests_per_minute
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get the HTTP client."""
        if self._client is None:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        return self._client
    
    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        authenticated: bool = True,
    ) -> Dict[str, Any]:
        """
        Make an API request.
        
        Args:
            method: HTTP method
            path: API path (without base URL)
            params: Query parameters
            json: JSON body
            authenticated: Whether to include auth headers
            
        Returns:
            Response JSON
        """
        async with self.throttler:
            url = f"{self.base_url}{path}"
            
            headers = {}
            if authenticated:
                # Kalshi requires signing the full path including /trade-api/v2
                full_path = f"/trade-api/v2{path}"
                headers = self.auth.get_headers(method, full_path)
            
            logger.debug(
                "api_request",
                method=method,
                path=path,
                params=params,
            )
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = await self.client.request(
                        method=method,
                        url=url,
                        params=params,
                        json=json,
                        headers=headers,
                    )
                    
                    if response.status_code == 401:
                        raise KalshiAuthError("Authentication failed")
                    elif response.status_code == 429:
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt  # Exponential backoff
                            logger.warning("rate_limit_retry", attempt=attempt, wait=wait_time)
                            await asyncio.sleep(wait_time)
                            continue
                        raise KalshiRateLimitError("Rate limit exceeded")
                    elif response.status_code >= 400:
                        error_detail = response.text
                        raise KalshiClientError(
                            f"API error {response.status_code}: {error_detail}"
                        )
                    
                    return response.json()
                    
                except httpx.HTTPError as e:
                    logger.error("http_error", error=str(e), path=path)
                    raise KalshiClientError(f"HTTP error: {e}")
    
    # ========== Market Data ==========
    
    async def get_markets(
        self,
        status: Optional[str] = None,
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> tuple[List[Market], Optional[str]]:
        """
        Get list of markets.
        
        Args:
            status: Filter by status ("open", "closed", "settled")
            event_ticker: Filter by event
            series_ticker: Filter by series
            limit: Number of results (max 200)
            cursor: Pagination cursor
            
        Returns:
            Tuple of (markets list, next cursor)
        """
        params = {"limit": limit}
        if status:
            params["status"] = status
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        
        response = await self._request("GET", "/markets", params=params, authenticated=False)
        
        markets = [self._parse_market(m) for m in response.get("markets", [])]
        next_cursor = response.get("cursor")
        
        return markets, next_cursor
    
    async def get_all_open_markets(self) -> List[Market]:
        """Get all open markets (handles pagination)."""
        all_markets = []
        cursor = None
        
        while True:
            try:
                markets, cursor = await self.get_markets(status="open", cursor=cursor, limit=200)
                all_markets.extend(markets)
                
                if not cursor:
                    break
                
                # Small delay between pagination requests to avoid rate limits
                await asyncio.sleep(0.5)
            except KalshiRateLimitError:
                logger.warning("rate_limit_hit_pagination", waiting=5)
                await asyncio.sleep(5)
                continue
        
        logger.info("fetched_all_markets", count=len(all_markets))
        return all_markets
    
    async def get_market(self, ticker: str) -> Market:
        """Get a single market by ticker."""
        response = await self._request(
            "GET", f"/markets/{ticker}", authenticated=False
        )
        return self._parse_market(response.get("market", response))
    
    async def get_orderbook(self, ticker: str, depth: int = 10) -> OrderBook:
        """
        Get order book for a market.
        
        Args:
            ticker: Market ticker
            depth: Number of price levels
            
        Returns:
            OrderBook with YES and NO bids
        """
        response = await self._request(
            "GET",
            f"/markets/{ticker}/orderbook",
            params={"depth": depth},
            authenticated=False,
        )
        
        orderbook = response.get("orderbook", response) or {}
        
        return OrderBook(
            ticker=ticker,
            yes_bids=[
                OrderBookLevel(price=level[0], quantity=level[1])
                for level in (orderbook.get("yes") or [])
            ],
            no_bids=[
                OrderBookLevel(price=level[0], quantity=level[1])
                for level in (orderbook.get("no") or [])
            ],
        )
    
    async def get_events(
        self,
        status: Optional[str] = None,
        series_ticker: Optional[str] = None,
        with_nested_markets: bool = True,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> tuple[List[Event], Optional[str]]:
        """Get list of events."""
        params = {
            "limit": limit,
            "with_nested_markets": str(with_nested_markets).lower(),
        }
        if status:
            params["status"] = status
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        
        response = await self._request("GET", "/events", params=params, authenticated=False)
        
        events = []
        for e in response.get("events", []):
            event = Event(
                event_ticker=e["event_ticker"],
                title=e.get("title", ""),
                subtitle=e.get("subtitle"),
                category=e.get("category"),
                series_ticker=e.get("series_ticker"),
            )
            if with_nested_markets and "markets" in e:
                event.markets = [self._parse_market(m) for m in e["markets"]]
            events.append(event)
        
        return events, response.get("cursor")
    
    # ========== Order Management ==========
    
    async def create_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: int,
        price: Optional[int] = None,
        order_type: OrderType = OrderType.LIMIT,
        client_order_id: Optional[str] = None,
    ) -> Order:
        """
        Create a new order.
        
        Args:
            ticker: Market ticker
            side: YES or NO
            quantity: Number of contracts
            price: Limit price in cents (required for limit orders)
            order_type: LIMIT or MARKET
            client_order_id: Client-generated ID for deduplication
            
        Returns:
            Created Order
        """
        if order_type == OrderType.LIMIT and price is None:
            raise ValueError("Price required for limit orders")
        
        if client_order_id is None:
            client_order_id = str(uuid.uuid4())
        
        order_data = {
            "ticker": ticker,
            "action": "buy",
            "side": side.value,
            "type": order_type.value,
            "count": quantity,
            "client_order_id": client_order_id,
        }
        
        if price is not None:
            order_data["yes_price" if side == OrderSide.YES else "no_price"] = price
        
        response = await self._request("POST", "/portfolio/orders", json=order_data)
        
        order = response.get("order", response)
        
        logger.info(
            "order_created",
            ticker=ticker,
            side=side.value,
            quantity=quantity,
            price=price,
            order_id=order.get("order_id"),
        )
        
        return self._parse_order(order)
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        try:
            await self._request("DELETE", f"/portfolio/orders/{order_id}")
            logger.info("order_cancelled", order_id=order_id)
            return True
        except KalshiClientError as e:
            logger.warning("order_cancel_failed", order_id=order_id, error=str(e))
            return False
    
    async def get_order(self, order_id: str) -> Order:
        """Get order by ID."""
        response = await self._request("GET", f"/portfolio/orders/{order_id}")
        return self._parse_order(response.get("order", response))
    
    async def get_orders(
        self,
        ticker: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Order]:
        """Get list of orders."""
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status
        
        response = await self._request("GET", "/portfolio/orders", params=params)
        return [self._parse_order(o) for o in response.get("orders", [])]
    
    # ========== Portfolio ==========
    
    async def get_balance(self) -> PortfolioBalance:
        """Get account balance."""
        response = await self._request("GET", "/portfolio/balance")
        
        # Response format: {'balance': cents, 'portfolio_value': cents, 'updated_ts': timestamp}
        balance_cents = response.get("balance", 0)
        portfolio_value_cents = response.get("portfolio_value", 0)
        
        return PortfolioBalance(
            available_balance=balance_cents / 100,  # Convert cents to dollars
            total_balance=(balance_cents + portfolio_value_cents) / 100,
            pending_balance=portfolio_value_cents / 100,
        )
    
    async def get_positions(self) -> List[Position]:
        """Get all portfolio positions."""
        response = await self._request("GET", "/portfolio/positions")
        
        positions = []
        for p in response.get("market_positions", []):
            positions.append(Position(
                ticker=p["ticker"],
                yes_quantity=p.get("position", 0) if p.get("position", 0) > 0 else 0,
                no_quantity=abs(p.get("position", 0)) if p.get("position", 0) < 0 else 0,
                realized_pnl=p.get("realized_pnl", 0) / 100,
            ))
        
        return positions
    
    async def get_fills(
        self,
        ticker: Optional[str] = None,
        limit: int = 100,
    ) -> List[Trade]:
        """Get recent fills/trades."""
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        
        response = await self._request("GET", "/portfolio/fills", params=params)
        
        trades = []
        for f in response.get("fills", []):
            trades.append(Trade(
                trade_id=f.get("trade_id", ""),
                order_id=f.get("order_id", ""),
                ticker=f.get("ticker", ""),
                side=OrderSide(f.get("side", "yes")),
                price=f.get("price", 0),
                quantity=f.get("count", 0),
                total_cost=f.get("price", 0) * f.get("count", 0),
                timestamp=datetime.fromisoformat(f["created_time"].replace("Z", "+00:00"))
                if f.get("created_time") else datetime.utcnow(),
            ))
        
        return trades
    
    # ========== Batch Operations ==========
    
    async def create_orders_parallel(
        self,
        orders: List[Dict[str, Any]],
    ) -> List[Order]:
        """
        Create multiple orders in parallel.
        
        Args:
            orders: List of order specs with keys: ticker, side, quantity, price
            
        Returns:
            List of created orders
        """
        tasks = [
            self.create_order(
                ticker=o["ticker"],
                side=o["side"],
                quantity=o["quantity"],
                price=o.get("price"),
            )
            for o in orders
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        created = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "parallel_order_failed",
                    order=orders[i],
                    error=str(result),
                )
            else:
                created.append(result)
        
        return created
    
    async def get_orderbooks_parallel(
        self,
        tickers: List[str],
        depth: int = 10,
        batch_size: int = 10,  # Limit concurrent requests
    ) -> Dict[str, OrderBook]:
        """Get order books for multiple markets in batches to avoid rate limits."""
        orderbooks = {}
        
        # Process in batches
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            tasks = [self.get_orderbook(t, depth) for t in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for ticker, result in zip(batch, results):
                if not isinstance(result, Exception):
                    orderbooks[ticker] = result
            
            # Small delay between batches
            if i + batch_size < len(tickers):
                await asyncio.sleep(0.5)
        
        return orderbooks
    
    # ========== Helpers ==========
    
    def _parse_market(self, data: Dict[str, Any]) -> Market:
        """Parse market data from API response."""
        return Market(
            ticker=data["ticker"],
            title=data.get("title", ""),
            subtitle=data.get("subtitle"),
            status=MarketStatus(data.get("status", "open")),
            event_ticker=data.get("event_ticker", ""),
            yes_bid=data.get("yes_bid"),
            yes_ask=data.get("yes_ask"),
            no_bid=data.get("no_bid"),
            no_ask=data.get("no_ask"),
            last_price=data.get("last_price"),
            volume=data.get("volume", 0),
            volume_24h=data.get("volume_24h", 0),
            open_interest=data.get("open_interest", 0),
            open_time=self._parse_datetime(data.get("open_time")),
            close_time=self._parse_datetime(data.get("close_time")),
            expiration_time=self._parse_datetime(data.get("expiration_time")),
            result=data.get("result"),
        )
    
    def _parse_order(self, data: Dict[str, Any]) -> Order:
        """Parse order data from API response."""
        return Order(
            order_id=data.get("order_id", ""),
            client_order_id=data.get("client_order_id"),
            ticker=data.get("ticker", ""),
            side=OrderSide(data.get("side", "yes")),
            type=OrderType(data.get("type", "limit")),
            price=data.get("yes_price") or data.get("no_price"),
            quantity=data.get("count", 0),
            filled_quantity=data.get("filled_count", 0),
            remaining_quantity=data.get("remaining_count", 0),
            status=OrderStatus(data.get("status", "pending")),
            created_at=self._parse_datetime(data.get("created_time")) or datetime.utcnow(),
            updated_at=self._parse_datetime(data.get("updated_time")),
        )
    
    def _parse_datetime(self, value: Optional[str]) -> Optional[datetime]:
        """Parse datetime string."""
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None
