"""
Arbitrage Detection Engine.

Detects three types of arbitrage:
1. Single-condition: YES + NO ≠ $1.00 in a single market
2. Market rebalancing: Sum of all outcomes ≠ $1.00 in multi-outcome events
3. Combinatorial: Cross-market dependencies create mispricing

From the research:
- 41% of conditions had single-market arbitrage
- 42% of multi-condition markets had rebalancing opportunities  
- 13 cross-market pairs had combinatorial arbitrage
"""

import asyncio
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import uuid

import numpy as np
import structlog

from ..api.models import Market, Event, OrderBook, ArbitrageOpportunity, OrderSide
from ..optimization import (
    BarrierFrankWolfe,
    LMSRBregman,
    IPSolverFactory,
    MarginalPolytope,
)
from ..config import Settings

logger = structlog.get_logger(__name__)


class ArbitrageDetector:
    """
    Main arbitrage detection engine.
    
    Scans markets for all types of arbitrage and ranks opportunities
    by guaranteed profit.
    """
    
    def __init__(
        self,
        settings: Settings,
        ip_solver_type: str = "ortools",
    ):
        """
        Initialize detector.
        
        Args:
            settings: Application settings
            ip_solver_type: Type of IP solver to use
        """
        self.settings = settings
        self.ip_solver = IPSolverFactory.create(ip_solver_type)
        
        # Thresholds
        self.min_profit = settings.min_profit_threshold
        self.alpha = settings.alpha_extraction
        
        # Cache for Frank-Wolfe optimizations
        self._fw_cache: Dict[str, Any] = {}
        
        logger.info(
            "arbitrage_detector_initialized",
            min_profit=self.min_profit,
            alpha=self.alpha,
        )
    
    async def scan_all(
        self,
        markets: List[Market],
        orderbooks: Dict[str, OrderBook],
        events: Optional[List[Event]] = None,
    ) -> List[ArbitrageOpportunity]:
        """
        Scan for all types of arbitrage.
        
        Args:
            markets: List of markets to scan
            orderbooks: Order books keyed by ticker
            events: Optional events for multi-outcome markets
            
        Returns:
            List of opportunities sorted by guaranteed profit
        """
        opportunities = []
        
        # 1. Single-condition arbitrage
        single_opps = await self.detect_single_condition(markets, orderbooks)
        opportunities.extend(single_opps)
        
        # 2. Market rebalancing
        if events:
            rebalance_opps = await self.detect_market_rebalancing(events, orderbooks)
            opportunities.extend(rebalance_opps)
        
        # Sort by guaranteed profit (descending)
        opportunities.sort(key=lambda x: x.guaranteed_profit, reverse=True)
        
        # Filter by minimum threshold
        opportunities = [o for o in opportunities if o.guaranteed_profit >= self.min_profit]
        
        logger.info(
            "arbitrage_scan_complete",
            total_opportunities=len(opportunities),
            single_condition=len(single_opps),
            rebalancing=len([o for o in opportunities if o.type == "market_rebalancing"]),
        )
        
        return opportunities
    
    async def detect_single_condition(
        self,
        markets: List[Market],
        orderbooks: Dict[str, OrderBook],
    ) -> List[ArbitrageOpportunity]:
        """
        Detect single-condition arbitrage: YES + NO ≠ $1.00.
        
        Types:
        - Buy both < $1: Buy YES and NO for less than $1, guaranteed $1 payout
        - Sell both > $1: Prices sum to > $1, arbitrage by selling overpriced side
        """
        opportunities = []
        
        for market in markets:
            if not market.is_tradeable:
                continue
            
            ticker = market.ticker
            orderbook = orderbooks.get(ticker)
            
            if not orderbook:
                continue
            
            # Get best prices
            yes_bid = orderbook.best_yes_bid  # Best price to sell YES
            no_bid = orderbook.best_no_bid    # Best price to sell NO
            yes_ask = orderbook.yes_ask       # Price to buy YES (100 - best_no_bid)
            no_ask = orderbook.no_ask         # Price to buy NO (100 - best_yes_bid)
            
            if yes_ask is None or no_ask is None:
                continue
            
            # Check BUY BOTH: cost to buy YES + NO < 100 cents
            buy_cost = yes_ask + no_ask
            if buy_cost < 100:
                profit_cents = 100 - buy_cost
                profit_dollars = profit_cents / 100
                
                if profit_dollars >= self.min_profit:
                    # Calculate liquidity-adjusted profit
                    yes_depth = orderbook.get_depth(OrderSide.YES)
                    no_depth = orderbook.get_depth(OrderSide.NO)
                    max_quantity = min(yes_depth, no_depth)
                    
                    # VWAP for realistic execution
                    vwap_yes = orderbook.get_vwap(OrderSide.YES, min(100, max_quantity))
                    vwap_no = orderbook.get_vwap(OrderSide.NO, min(100, max_quantity))
                    
                    if vwap_yes and vwap_no:
                        actual_cost = (100 - vwap_no) + (100 - vwap_yes)
                        actual_profit = (100 - actual_cost) / 100
                    else:
                        actual_profit = profit_dollars
                    
                    opp = ArbitrageOpportunity(
                        opportunity_id=str(uuid.uuid4()),
                        type="single_condition_buy",
                        tickers=[ticker],
                        prices={
                            f"{ticker}_yes_ask": yes_ask / 100,
                            f"{ticker}_no_ask": no_ask / 100,
                        },
                        theoretical_profit=profit_dollars,
                        guaranteed_profit=actual_profit * 0.95,  # 5% buffer for slippage
                        profit_margin=profit_dollars / (buy_cost / 100),
                        recommended_positions={
                            ticker: {
                                "yes_action": "buy",
                                "yes_price": yes_ask,
                                "no_action": "buy", 
                                "no_price": no_ask,
                                "quantity": min(100, max_quantity),
                            }
                        },
                        total_cost=buy_cost / 100,
                        max_payout=1.0,
                        available_liquidity=max_quantity,
                        vwap_slippage=abs(profit_dollars - actual_profit),
                        detected_at=datetime.utcnow(),
                        is_executable=max_quantity >= 10,
                        execution_risk_score=0.1 if max_quantity > 100 else 0.3,
                    )
                    opportunities.append(opp)
            
            # Check SELL BOTH: prices to sell sum > 100 cents
            if yes_bid and no_bid:
                sell_revenue = yes_bid + no_bid
                if sell_revenue > 100:
                    profit_cents = sell_revenue - 100
                    profit_dollars = profit_cents / 100
                    
                    if profit_dollars >= self.min_profit:
                        yes_depth = orderbook.get_depth(OrderSide.YES)
                        no_depth = orderbook.get_depth(OrderSide.NO)
                        max_quantity = min(yes_depth, no_depth)
                        
                        opp = ArbitrageOpportunity(
                            opportunity_id=str(uuid.uuid4()),
                            type="single_condition_sell",
                            tickers=[ticker],
                            prices={
                                f"{ticker}_yes_bid": yes_bid / 100,
                                f"{ticker}_no_bid": no_bid / 100,
                            },
                            theoretical_profit=profit_dollars,
                            guaranteed_profit=profit_dollars * 0.90,
                            profit_margin=profit_dollars,
                            recommended_positions={
                                ticker: {
                                    "yes_action": "sell",
                                    "yes_price": yes_bid,
                                    "no_action": "sell",
                                    "no_price": no_bid,
                                    "quantity": min(100, max_quantity),
                                }
                            },
                            total_cost=0,  # Selling, not buying
                            max_payout=profit_dollars,
                            available_liquidity=max_quantity,
                            vwap_slippage=0.01,
                            detected_at=datetime.utcnow(),
                            is_executable=max_quantity >= 10,
                            execution_risk_score=0.15,
                        )
                        opportunities.append(opp)
        
        return opportunities
    
    async def detect_market_rebalancing(
        self,
        events: List[Event],
        orderbooks: Dict[str, OrderBook],
    ) -> List[ArbitrageOpportunity]:
        """
        Detect market rebalancing arbitrage in multi-outcome events.
        
        If an event has N mutually exclusive outcomes, the sum of all
        YES prices should equal $1.00. Any deviation is arbitrageable.
        """
        opportunities = []
        
        for event in events:
            if not event.is_multi_outcome:
                continue
            
            # Get prices for all markets in the event
            prices = []
            valid_markets = []
            
            for market in event.markets:
                if not market.is_tradeable:
                    continue
                
                ob = orderbooks.get(market.ticker)
                if not ob or ob.yes_ask is None:
                    continue
                
                # Use ask price (cost to buy YES)
                prices.append(ob.yes_ask / 100)  # Convert to dollars
                valid_markets.append(market)
            
            if len(valid_markets) < 2:
                continue
            
            total_cost = sum(prices)
            
            # BUY ALL YES: If sum < $1, buy all YES contracts
            if total_cost < 1.0 - self.min_profit:
                profit = 1.0 - total_cost
                
                # Get minimum liquidity across all markets
                min_liquidity = float('inf')
                for m in valid_markets:
                    ob = orderbooks.get(m.ticker)
                    if ob:
                        depth = ob.get_depth(OrderSide.YES)
                        min_liquidity = min(min_liquidity, depth)
                
                if min_liquidity < 10:
                    continue
                
                positions = {}
                for i, m in enumerate(valid_markets):
                    ob = orderbooks.get(m.ticker)
                    positions[m.ticker] = {
                        "action": "buy_yes",
                        "price": ob.yes_ask if ob else int(prices[i] * 100),
                        "quantity": min(50, int(min_liquidity)),
                    }
                
                opp = ArbitrageOpportunity(
                    opportunity_id=str(uuid.uuid4()),
                    type="market_rebalancing_buy_yes",
                    tickers=[m.ticker for m in valid_markets],
                    event_tickers=[event.event_ticker],
                    prices={m.ticker: prices[i] for i, m in enumerate(valid_markets)},
                    theoretical_profit=profit,
                    guaranteed_profit=profit * 0.85,  # Conservative estimate
                    profit_margin=profit / total_cost,
                    recommended_positions=positions,
                    total_cost=total_cost,
                    max_payout=1.0,
                    available_liquidity=min_liquidity,
                    vwap_slippage=0.02,
                    detected_at=datetime.utcnow(),
                    is_executable=min_liquidity >= 10,
                    execution_risk_score=0.2 + 0.05 * len(valid_markets),  # More markets = more risk
                )
                opportunities.append(opp)
            
            # For sell all / buy all NO, similar logic would apply
            # Omitted for brevity but follows same pattern
        
        return opportunities
    
    async def detect_combinatorial(
        self,
        market_pairs: List[Tuple[Market, Market]],
        orderbooks: Dict[str, OrderBook],
        dependencies: List[Dict[str, Any]],
    ) -> List[ArbitrageOpportunity]:
        """
        Detect combinatorial arbitrage using Frank-Wolfe optimization.
        
        This is the most complex detection, using:
        1. Build MarginalPolytope with dependency constraints
        2. Run Barrier Frank-Wolfe to find Bregman projection
        3. Use Proposition 4.1 to compute guaranteed profit
        
        Args:
            market_pairs: Pairs of potentially dependent markets
            orderbooks: Current order books
            dependencies: Known dependencies between markets
        """
        opportunities = []
        
        for market_a, market_b in market_pairs:
            # Build polytope for this pair
            polytope = MarginalPolytope()
            
            # Add markets (simplified: assume binary for now)
            polytope.add_market(market_a.ticker, ["yes", "no"])
            polytope.add_market(market_b.ticker, ["yes", "no"])
            
            # Find applicable dependencies
            pair_deps = [
                d for d in dependencies
                if market_a.ticker in str(d) and market_b.ticker in str(d)
            ]
            
            if not pair_deps:
                continue  # No dependency, no combinatorial arbitrage
            
            # Add dependency constraints
            for dep in pair_deps:
                if dep.get("type") == "mutual_exclusion":
                    # E.g., both can't be YES
                    polytope.add_mutual_exclusion([0, 2])  # indices of YES outcomes
            
            # Get current prices
            ob_a = orderbooks.get(market_a.ticker)
            ob_b = orderbooks.get(market_b.ticker)
            
            if not ob_a or not ob_b:
                continue
            
            # Build theta (market state) from prices
            # Using simple log-odds transformation
            def price_to_theta(price_cents):
                p = max(0.01, min(0.99, price_cents / 100))
                return np.log(p / (1 - p))
            
            theta = np.array([
                price_to_theta(ob_a.yes_ask or 50),
                price_to_theta(ob_a.no_ask or 50),
                price_to_theta(ob_b.yes_ask or 50),
                price_to_theta(ob_b.no_ask or 50),
            ])
            
            # Run Frank-Wolfe
            bregman = LMSRBregman(liquidity=100.0)
            fw = BarrierFrankWolfe(
                bregman=bregman,
                ip_solver=self.ip_solver,
                polytope=polytope,
                alpha=self.alpha,
                max_iterations=50,  # Limit for speed
                time_limit=30.0,
            )
            
            result = fw.run(theta)
            
            if result.is_profitable and result.guaranteed_profit >= self.min_profit:
                opp = ArbitrageOpportunity(
                    opportunity_id=str(uuid.uuid4()),
                    type="combinatorial",
                    tickers=[market_a.ticker, market_b.ticker],
                    prices={
                        market_a.ticker: (ob_a.yes_ask or 0) / 100,
                        market_b.ticker: (ob_b.yes_ask or 0) / 100,
                    },
                    theoretical_profit=result.divergence,
                    guaranteed_profit=result.guaranteed_profit,
                    profit_margin=result.extraction_ratio,
                    recommended_positions={},  # Would need trade calculation
                    total_cost=0,  # Depends on specific trade
                    max_payout=result.guaranteed_profit,
                    available_liquidity=min(
                        ob_a.get_depth(OrderSide.YES),
                        ob_b.get_depth(OrderSide.YES),
                    ),
                    vwap_slippage=0.03,
                    detected_at=datetime.utcnow(),
                    is_executable=True,
                    execution_risk_score=0.25,
                )
                opportunities.append(opp)
                
                logger.info(
                    "combinatorial_arbitrage_found",
                    markets=[market_a.ticker, market_b.ticker],
                    profit=result.guaranteed_profit,
                    iterations=result.iterations,
                )
        
        return opportunities
    
    def quick_scan(
        self,
        markets: List[Market],
        orderbooks: Dict[str, OrderBook],
    ) -> List[Tuple[str, float]]:
        """
        Quick synchronous scan for single-condition arbitrage.
        
        Returns list of (ticker, profit) tuples for markets with arbitrage.
        Useful for real-time monitoring without async overhead.
        """
        results = []
        
        for market in markets:
            if not market.is_tradeable:
                continue
            
            ob = orderbooks.get(market.ticker)
            if not ob:
                continue
            
            yes_ask = ob.yes_ask
            no_ask = ob.no_ask
            
            if yes_ask is None or no_ask is None:
                continue
            
            buy_cost = yes_ask + no_ask
            if buy_cost < 100:
                profit = (100 - buy_cost) / 100
                if profit >= self.min_profit:
                    results.append((market.ticker, profit))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
