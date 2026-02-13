# Kalshi Arbitrage Bot - Quick Start Guide

## What This Bot Does

This bot implements the **exact mathematical framework** from the research paper "Unravelling the Probabilistic Forest" that showed how traders extracted **$40 million in guaranteed arbitrage profits**.

### The Math (exactly as described):

1. **Bregman Divergence** - Measures price mispricing using KL divergence
2. **Frank-Wolfe Algorithm** - Iteratively finds optimal arbitrage trades
3. **Barrier Frank-Wolfe** - Handles LMSR gradient explosion with adaptive contraction
4. **Integer Programming** - Solves for valid payoff vectors (uses OR-Tools)
5. **Proposition 4.1** - Calculates guaranteed profit: `D(μ||θ) - g(μ)`

### Learning Mode

The bot is configured to make **hundreds of small $1 bets** so you can:
- Learn how the market behaves
- See the math in action
- Understand execution dynamics
- Build confidence before scaling up

---

## Step-by-Step Setup

### Step 1: Open Terminal in the Bot Directory

```bash
cd C:\Users\micka\OneDrive\Documents\affilate\kalashi\kalshi-arbitrage-bot
```

### Step 2: Create Python Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Run Setup Script

```bash
python setup.py
```

This will:
- Install all dependencies
- Generate your API keys
- Test the math algorithms
- Create your `.env` file

### Step 4: Upload Public Key to Kalshi

1. Go to: https://kalshi.com/account/api-keys
2. Click "Add API Key"
3. Give it a name (e.g., "Arbitrage Bot")
4. Upload the file: `kalshi_keys_public.pem`
5. **Copy the API Key ID** that Kalshi gives you

### Step 5: Configure Your .env File

Open `.env` and add your API key:

```
KALSHI_API_KEY_ID=your_key_id_from_step_4
```

### Step 6: Add Funds to Kalshi

- Deposit funds at https://kalshi.com
- For learning: Start with $50-100
- The bot will make $1 bets

### Step 7: Test the Connection

```bash
python run.py status
```

You should see your balance and account info.

### Step 8: Scan for Opportunities

```bash
python run.py scan --top 20
```

This shows current arbitrage opportunities without trading.

### Step 9: Start Learning Mode

```bash
# Start with demo mode first (no real money)
python run.py start --demo

# When ready for real trading:
python run.py start
```

---

## What Happens When Running

The bot continuously:

1. **Fetches market data** via WebSocket (real-time)
2. **Scans for arbitrage** every second
3. **Detects 3 types**:
   - Single-condition: YES + NO ≠ $1.00
   - Market rebalancing: Sum of outcomes ≠ $1.00
   - Combinatorial: Cross-market dependencies
4. **Calculates guaranteed profit** using Proposition 4.1
5. **Executes trades** in parallel if profit > threshold
6. **Logs everything** for your review

---

## Understanding the Output

```
[INFO] opportunities_detected count=3 best_profit=0.0234
```
- Found 3 opportunities
- Best one guarantees $0.0234 profit (2.34 cents)

```
[INFO] trade_executed profit=0.0189
```
- Made a trade
- Guaranteed profit: $0.0189

```
[INFO] fw_alpha_extracted alpha=0.9 extraction=0.92
```
- Frank-Wolfe stopped at 92% profit extraction
- This matches the paper's recommendation

---

## Configuration Options

Edit `.env` to change:

| Setting | Default | Description |
|---------|---------|-------------|
| `LEARNING_MODE` | true | Fixed $1 bets for learning |
| `LEARNING_BET_SIZE` | 1.00 | Size of each bet |
| `MIN_PROFIT_THRESHOLD` | 0.01 | Min profit to trade ($0.01) |
| `MAX_DAILY_TRADES` | 500 | Max trades per day |
| `ALPHA_EXTRACTION` | 0.9 | Stop at 90% profit capture |

---

## Monitoring Your Trades

### View Live Status
```bash
python run.py status
```

### View Logs
```bash
type logs\arbitrage_bot.log
```

### Stop the Bot
Press `Ctrl+C` in the terminal

---

## Scaling Up

Once you're comfortable with the math:

1. Set `LEARNING_MODE=false` in `.env`
2. Increase `LEARNING_BET_SIZE` or disable learning mode
3. The bot will use Kelly Criterion for optimal sizing
4. Start with small amounts and scale gradually

---

## Troubleshooting

### "Authentication failed"
- Check your API Key ID in `.env`
- Make sure you uploaded the PUBLIC key to Kalshi
- The private key should be `kalshi_private_key.pem`

### "No opportunities found"
- Markets may be efficiently priced
- Try during high-volume periods
- Lower `MIN_PROFIT_THRESHOLD` to see smaller opportunities

### "IP solver timeout"
- Normal for complex multi-market analysis
- The bot will use best result found

---

## The Math Reference

From the paper, the key formula is **Proposition 4.1**:

```
Guaranteed Profit ≥ D(μ||θ) - g(μ)
```

Where:
- `D(μ||θ)` = Bregman divergence (max profit if perfect)
- `g(μ)` = Frank-Wolfe gap (how suboptimal current solution is)
- The difference = **guaranteed minimum profit**

The bot calculates this for every opportunity before trading.

---

## Questions?

The code is fully documented. Key files:
- `src/optimization/bregman.py` - Bregman divergence math
- `src/optimization/frank_wolfe.py` - Frank-Wolfe algorithm
- `src/arbitrage/detector.py` - Opportunity detection
- `src/execution/engine.py` - Trade execution
