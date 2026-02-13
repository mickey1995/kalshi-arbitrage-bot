# Kalshi Arbitrage Bot

A sophisticated arbitrage detection and execution system for Kalshi prediction markets, implementing the mathematical frameworks from cutting-edge research.

## Overview

This bot implements the algorithms described in:
- "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets" (arXiv:2508.03474v1)
- "Arbitrage-Free Combinatorial Market Making via Integer Programming" (arXiv:1606.02825v2)

The research showed that sophisticated traders extracted **$40 million in guaranteed arbitrage profits** from Polymarket in one year using these exact techniques. This implementation adapts them for Kalshi.

## Features

### Arbitrage Detection

1. **Single-Condition Arbitrage**
   - Detects when YES + NO ≠ $1.00
   - From research: 41% of conditions had this type

2. **Market Rebalancing**
   - Multi-outcome events where sum ≠ $1.00
   - From research: $29M+ extracted this way

3. **Combinatorial Arbitrage**
   - Cross-market dependencies using Frank-Wolfe optimization
   - Integer Programming for constraint modeling
   - Bregman projection for optimal pricing

### Core Algorithms

- **Frank-Wolfe Algorithm**: Iterative optimization over marginal polytope
- **Barrier Frank-Wolfe**: Handles LMSR gradient explosion via adaptive contraction
- **InitFW**: Proper initialization with valid vertices and interior points
- **Integer Programming**: Google OR-Tools (free) or Gurobi (commercial)
- **Proposition 4.1**: Guaranteed profit calculation: `D(μ||θ) - g(μ)`

### Execution Engine

- Parallel order submission
- VWAP-based position sizing
- Modified Kelly Criterion
- Risk management with drawdown limits
- Real-time WebSocket order book tracking

## Installation

```bash
# Clone the repository
cd kalshi-arbitrage-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Generate API keys:
```bash
python run.py generate-keys
```

3. Upload the PUBLIC key to Kalshi: https://kalshi.com/account/api-keys

4. Update `.env` with your API key ID and private key path

## Usage

### Quick Scan
```bash
python run.py scan --top 10
```

### Check Status
```bash
python run.py status
```

### Test Optimization
```bash
python run.py test
```

### Start Trading Bot
```bash
# Demo environment (recommended for testing)
python run.py start --demo

# Production (real money)
python run.py start
```

## Architecture

```
kalshi-arbitrage-bot/
├── src/
│   ├── api/              # Kalshi API client
│   │   ├── auth.py       # RSA-PSS authentication
│   │   ├── client.py     # REST API client
│   │   ├── websocket.py  # Real-time data
│   │   └── models.py     # Data models
│   │
│   ├── optimization/     # Mathematical core
│   │   ├── bregman.py    # Bregman divergence (LMSR)
│   │   ├── frank_wolfe.py # FW + Barrier FW algorithms
│   │   ├── ip_solver.py  # Integer programming
│   │   └── marginal_polytope.py # Constraint modeling
│   │
│   ├── arbitrage/        # Detection strategies
│   │   ├── detector.py   # Main detection engine
│   │   └── strategies.py # Strategy implementations
│   │
│   ├── execution/        # Trade execution
│   │   ├── engine.py     # Order management
│   │   └── risk.py       # Risk/position sizing
│   │
│   ├── config.py         # Configuration
│   └── bot.py            # Main orchestrator
│
├── run.py                # CLI entry point
├── requirements.txt      # Dependencies
└── .env.example         # Config template
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_PROFIT_THRESHOLD` | $0.05 | Minimum profit to execute |
| `ALPHA_EXTRACTION` | 0.9 | Stop when capturing 90% of profit |
| `MAX_DRAWDOWN_PERCENT` | 15% | Halt trading if exceeded |
| `FW_MAX_ITERATIONS` | 150 | Max Frank-Wolfe iterations |
| `IP_SOLVER` | ortools | Integer programming solver |

## Mathematical Foundation

### Bregman Divergence (LMSR)

For LMSR cost functions, the Bregman divergence is KL divergence:

```
D(μ||θ) = Σ μ_i ln(μ_i / p_i(θ))
```

This measures how "wrong" current prices are.

### Frank-Wolfe Gap

The gap measures how suboptimal the current solution is:

```
g(μ) = ∇F(μ)·(μ - z*)
```

Where z* is the descent vertex from the IP solver.

### Profit Guarantee (Proposition 4.1)

```
Guaranteed Profit ≥ D(μ||θ) - g(μ)
```

This is the key insight: we can compute exact guaranteed profit before executing any trade.

### Stopping Conditions

1. **α-Extraction**: Stop when `g(μ) ≤ (1-α) × D(μ||θ)`
2. **Near-Arbitrage-Free**: Stop when `D(μ||θ) < ε_D`
3. **Time Limit**: Return best iterate found

## Risk Warnings

⚠️ **This is experimental software for educational purposes.**

- Trading involves substantial risk of loss
- Past performance does not guarantee future results
- Start with the demo environment
- Never risk more than you can afford to lose
- The authors are not responsible for any financial losses

## Performance Expectations

Based on Kalshi's market structure (simpler than Polymarket):

| Strategy | Estimated Potential |
|----------|-------------------|
| Single-condition | $1-3M/year |
| Market rebalancing | $2-5M/year |
| Combinatorial | $10-50K/year |

These are rough estimates. Actual results depend on market conditions, execution speed, and competition.

## Future Enhancements

- [ ] LLM-based dependency detection
- [ ] Prometheus metrics dashboard
- [ ] Database persistence for trade history
- [ ] Backtesting framework
- [ ] Multi-account support
- [ ] Advanced market making

## License

MIT License - See LICENSE file

## References

1. Kroer et al. "Arbitrage-Free Combinatorial Market Making via Integer Programming" (arXiv:1606.02825v2)
2. Research paper: "Unravelling the Probabilistic Forest" (arXiv:2508.03474v1)
3. Kalshi API Documentation: https://docs.kalshi.com/
