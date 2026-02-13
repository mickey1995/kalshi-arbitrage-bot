"""
Arbitrage Strategy Implementations.

Provides specialized classes for each arbitrage type:
1. SingleConditionArbitrage - YES + NO ≠ $1.00
2. MarketRebalancingArbitrage - Multi-outcome sum ≠ $1.00
3. CombinatorialArbitrage - Cross-market dependencies
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid
import numpy as np

import structlog

from ..api.models import Market, Event, OrderBook, ArbitrageOpportunity, OrderSide

logger = structlog.get_logger(__name__)


class ArbitrageStrategy(ABC):
    """Base class for arbitrage strategies."""
    
    def __init__(self, min_profit: float = 0.05):
        """
        Initialize strategy.
        
        Args:
            min_profit: Minimum profit threshold in dollars
        """
        self.min_profit = min_profit
    
    @abstractmethod
    def detect(
        self,
        markets: List[Market],
        orderbooks: Dict[str, OrderBook],
        **kwargs,
    ) -> List[ArbitrageOpportunity]:
        """
        Detect arbitrage opportunities.
        
        Args:
            markets: Markets to scan
            orderbooks: Current order books
            **kwargs: Strategy-specific arguments
            
        Returns:
            List of detected opportunities
        """
        pass
    
    def _create_opportunity(
        self,
        type_: str,
        tickers: List[str],
        prices: Dict[str, float],
        profit: float,
        positions: Dict[str, Dict],
        **kwargs,
    ) -> ArbitrageOpportunity:
        """Helper to create opportunity objects."""
        return ArbitrageOpportunity(
            opportunity_id=str(uuid.uuid4()),
            type=type_,
            tickers=tickers,
            prices=prices,
            theoretical_profit=profit,
            guaranteed_profit=profit * 0.90,  # Conservative
            profit_margin=kwargs.get("margin", 0.0),
            recommended_positions=positions,
            total_cost=kwargs.get("cost", 0.0),
            max_payout=kwargs.get("payout", 1.0),
            available_liquidity=kwargs.get("liquidity", 0),
            vwap_slippage=kwargs.get("slippage", 0.01),
            detected_at=datetime.utcnow(),
            is_executable=kwargs.get("executable", True),
            execution_risk_score=kwargs.get("risk", 0.1),
        )


class SingleConditionArbitrage(ArbitrageStrategy):
    """
    Detects single-condition arbitrage.
    
    Opportunity exists when:
    - Buy both: YES_ask + NO_ask < 100 cents
    - Sell both: YES_bid + NO_bid > 100 cents
    
    From the research: 41% of conditions had this type of arbitrage
    with median mispricing of $0.60 per dollar.
    """
    
    def detect(
        self,
        markets: List[Market],
        orderbooks: Dict[str, OrderBook],
        **kwargs,
    ) -> List[ArbitrageOpportunity]:
        """Detect single-condition arbitrage."""
        opportunities = []
        
        for market in markets:
            if not market.is_tradeable:
                continue
            
            ob = orderbooks.get(market.ticker)
            if not ob:
                continue
            
            # Check buy both
            opp = self._check_buy_both(market, ob)
            if opp:
                opportunities.append(opp)
            
            # Check sell both
            opp = self._check_sell_both(market, ob)
            if opp:
                opportunities.append(opp)
        
        return opportunities
    
    def _check_buy_both(
        self,
        market: Market,
        ob: OrderBook,
    ) -> Optional[ArbitrageOpportunity]:
        """Check if buying both YES and NO is profitable."""
        yes_ask = ob.yes_ask  # Price to buy YES
        no_ask = ob.no_ask    # Price to buy NO
        
        if yes_ask is None or no_ask is None:
            return None
        
        cost = yes_ask + no_ask
        if cost >= 100:
            return None
        
        profit = (100 - cost) / 100
        if profit < self.min_profit:
            return None
        
        # Calculate liquidity
        yes_depth = ob.get_depth(OrderSide.YES)
        no_depth = ob.get_depth(OrderSide.NO)
        max_qty = min(yes_depth, no_depth)
        
        if max_qty < 10:
            return None
        
        return self._create_opportunity(
            type_="single_condition_buy",
            tickers=[market.ticker],
            prices={
                f"{market.ticker}_yes": yes_ask / 100,
                f"{market.ticker}_no": no_ask / 100,
            },
            profit=profit,
            positions={
                market.ticker: {
                    "yes_action": "buy",
                    "yes_price": yes_ask,
                    "no_action": "buy",
                    "no_price": no_ask,
                    "quantity": min(100, max_qty),
                }
            },
            cost=cost / 100,
            payout=1.0,
            liquidity=max_qty,
            margin=profit / (cost / 100),
            executable=True,
            risk=0.1 if max_qty > 100 else 0.25,
        )
    
    def _check_sell_both(
        self,
        market: Market,
        ob: OrderBook,
    ) -> Optional[ArbitrageOpportunity]:
        """Check if selling both YES and NO is profitable."""
        yes_bid = ob.best_yes_bid
        no_bid = ob.best_no_bid
        
        if yes_bid is None or no_bid is None:
            return None
        
        revenue = yes_bid + no_bid
        if revenue <= 100:
            return None
        
        profit = (revenue - 100) / 100
        if profit < self.min_profit:
            return None
        
        yes_depth = ob.get_depth(OrderSide.YES)
        no_depth = ob.get_depth(OrderSide.NO)
        max_qty = min(yes_depth, no_depth)
        
        if max_qty < 10:
            return None
        
        return self._create_opportunity(
            type_="single_condition_sell",
            tickers=[market.ticker],
            prices={
                f"{market.ticker}_yes_bid": yes_bid / 100,
                f"{market.ticker}_no_bid": no_bid / 100,
            },
            profit=profit,
            positions={
                market.ticker: {
                    "yes_action": "sell",
                    "yes_price": yes_bid,
                    "no_action": "sell",
                    "no_price": no_bid,
                    "quantity": min(100, max_qty),
                }
            },
            cost=0,  # Selling, not buying
            payout=profit,
            liquidity=max_qty,
            margin=1.0,
            executable=True,
            risk=0.15,
        )


class MarketRebalancingArbitrage(ArbitrageStrategy):
    """
    Detects market rebalancing arbitrage in multi-outcome events.
    
    For mutually exclusive outcomes:
    - Sum of all YES prices should equal $1.00
    - Deviation creates arbitrage
    
    From the research: 42% of multi-condition markets had rebalancing
    opportunities. $29M+ was extracted this way.
    """
    
    def detect(
        self,
        markets: List[Market],
        orderbooks: Dict[str, OrderBook],
        events: List[Event] = None,
        **kwargs,
    ) -> List[ArbitrageOpportunity]:
        """Detect market rebalancing arbitrage."""
        if not events:
            return []
        
        opportunities = []
        
        for event in events:
            if not event.is_multi_outcome:
                continue
            
            opp = self._check_event(event, orderbooks)
            if opp:
                opportunities.append(opp)
        
        return opportunities
    
    def _check_event(
        self,
        event: Event,
        orderbooks: Dict[str, OrderBook],
    ) -> Optional[ArbitrageOpportunity]:
        """Check a single event for rebalancing arbitrage."""
        prices = []
        valid_markets = []
        
        for market in event.markets:
            if not market.is_tradeable:
                continue
            
            ob = orderbooks.get(market.ticker)
            if not ob or ob.yes_ask is None:
                continue
            
            prices.append(ob.yes_ask / 100)
            valid_markets.append(market)
        
        if len(valid_markets) < 2:
            return None
        
        total = sum(prices)
        
        # Buy all YES if sum < 1
        if total < 1.0 - self.min_profit:
            profit = 1.0 - total
            
            # Calculate liquidity
            min_liquidity = float('inf')
            for m in valid_markets:
                ob = orderbooks.get(m.ticker)
                if ob:
                    min_liquidity = min(min_liquidity, ob.get_depth(OrderSide.YES))
            
            if min_liquidity < 10:
                return None
            
            positions = {}
            for i, m in enumerate(valid_markets):
                ob = orderbooks.get(m.ticker)
                positions[m.ticker] = {
                    "action": "buy_yes",
                    "price": ob.yes_ask if ob else int(prices[i] * 100),
                    "quantity": min(50, int(min_liquidity)),
                }
            
            return self._create_opportunity(
                type_="market_rebalancing_buy",
                tickers=[m.ticker for m in valid_markets],
                prices={m.ticker: prices[i] for i, m in enumerate(valid_markets)},
                profit=profit,
                positions=positions,
                cost=total,
                payout=1.0,
                liquidity=min_liquidity,
                margin=profit / total,
                risk=0.2 + 0.03 * len(valid_markets),
                event_tickers=[event.event_ticker],
            )
        
        return None


class CombinatorialArbitrage(ArbitrageStrategy):
    """
    Detects combinatorial arbitrage using Frank-Wolfe optimization.
    
    This is the most sophisticated type, using:
    1. Integer programming for constraint modeling
    2. Bregman projection for optimal pricing
    3. Frank-Wolfe for tractable computation
    
    From the research: $95K+ was extracted from 13 cross-market pairs.
    The potential is larger with more dependency discovery.
    """
    
    def __init__(
        self,
        min_profit: float = 0.05,
        ip_solver_type: str = "ortools",
        max_iterations: int = 50,
    ):
        """
        Initialize combinatorial strategy.
        
        Args:
            min_profit: Minimum profit threshold
            ip_solver_type: IP solver to use
            max_iterations: Max Frank-Wolfe iterations
        """
        super().__init__(min_profit)
        self.ip_solver_type = ip_solver_type
        self.max_iterations = max_iterations
        
        # Lazy import to avoid circular dependencies
        self._ip_solver = None
        self._bregman = None
    
    def _init_solver(self):
        """Initialize optimization components."""
        if self._ip_solver is None:
            from ..optimization import IPSolverFactory, LMSRBregman
            self._ip_solver = IPSolverFactory.create(self.ip_solver_type)
            self._bregman = LMSRBregman(liquidity=100.0)
    
    def detect(
        self,
        markets: List[Market],
        orderbooks: Dict[str, OrderBook],
        dependencies: List[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[ArbitrageOpportunity]:
        """
        Detect combinatorial arbitrage.
        
        Args:
            markets: Markets to scan
            orderbooks: Order books
            dependencies: Known market dependencies
            
        Returns:
            List of opportunities
        """
        if not dependencies:
            return []
        
        self._init_solver()
        
        opportunities = []
        
        # Group dependencies by market pairs
        from itertools import combinations
        
        market_dict = {m.ticker: m for m in markets}
        
        for dep in dependencies:
            tickers = dep.get("tickers", [])
            if len(tickers) < 2:
                continue
            
            # Get markets
            dep_markets = [market_dict.get(t) for t in tickers]
            if not all(dep_markets):
                continue
            
            # Check this dependency
            opp = self._check_dependency(dep_markets, orderbooks, dep)
            if opp:
                opportunities.append(opp)
        
        return opportunities
    
    def _check_dependency(
        self,
        markets: List[Market],
        orderbooks: Dict[str, OrderBook],
        dependency: Dict[str, Any],
    ) -> Optional[ArbitrageOpportunity]:
        """Check a specific dependency for arbitrage."""
        from ..optimization import BarrierFrankWolfe, MarginalPolytope
        
        # Build polytope
        polytope = MarginalPolytope()
        
        for market in markets:
            polytope.add_market(market.ticker, ["yes", "no"])
        
        # Add dependency constraint
        dep_type = dependency.get("type")
        indices = dependency.get("indices", [])
        
        if dep_type == "mutual_exclusion" and indices:
            polytope.add_mutual_exclusion(indices)
        
        # Build theta from prices
        theta = []
        for market in markets:
            ob = orderbooks.get(market.ticker)
            if not ob:
                return None
            
            yes_price = (ob.yes_ask or 50) / 100
            no_price = (ob.no_ask or 50) / 100
            
            # Log-odds transformation
            yes_price = max(0.01, min(0.99, yes_price))
            no_price = max(0.01, min(0.99, no_price))
            
            theta.extend([
                self._bregman.b * np.log(yes_price / (1 - yes_price)),
                self._bregman.b * np.log(no_price / (1 - no_price)),
            ])
        
        theta = np.array(theta)
        
        # Run Frank-Wolfe
        fw = BarrierFrankWolfe(
            bregman=self._bregman,
            ip_solver=self._ip_solver,
            polytope=polytope,
            max_iterations=self.max_iterations,
            time_limit=30.0,
        )
        
        result = fw.run(theta)
        
        if not result.is_profitable or result.guaranteed_profit < self.min_profit:
            return None
        
        return self._create_opportunity(
            type_="combinatorial",
            tickers=[m.ticker for m in markets],
            prices={m.ticker: (orderbooks.get(m.ticker).yes_ask or 0) / 100 for m in markets},
            profit=result.guaranteed_profit,
            positions={},  # Would need trade calculation
            cost=0,
            payout=result.guaranteed_profit,
            liquidity=min(
                orderbooks.get(m.ticker).get_depth(OrderSide.YES)
                for m in markets
                if orderbooks.get(m.ticker)
            ),
            margin=result.extraction_ratio,
            risk=0.25,
        )
