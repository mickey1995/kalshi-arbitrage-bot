"""
Forensic Pattern-Finder for Kalshi BTC-15M Series.

Main entry point. Run with:
    python run_forensic.py

Or with options:
    python run_forensic.py --count 200 --skip-spot --output report.txt
"""

import asyncio
import argparse
import sys
import os
import time
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

# Configure structlog BEFORE any imports that use it
import structlog
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger(__name__)


def load_env():
    """Load .env file manually (avoid pydantic-settings auto-load issues)."""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


async def run_forensic_analysis(
    count: int = 200,
    skip_spot: bool = False,
    output_file: str = "forensic_report.txt",
    series_ticker: str = "KXBTC15M",
):
    """
    Execute the full forensic analysis pipeline.
    
    Steps:
    1. Data Acquisition: Fetch settled markets + candlesticks
    2. Feature Extraction: Extract forensic features per window
    3. Statistical Audit: Bracket analysis, bias tests, PONR, Golden Minute
    4. Spot Analysis: Compare against BTC spot (optional)
    5. Report Generation: Produce final report
    """
    from src.api.auth import KalshiAuth
    from src.forensic.data_acquisition import ForensicDataClient
    from src.forensic.feature_extraction import extract_all_features
    from src.forensic.statistical_audit import ForensicAuditor
    from src.forensic.spot_analysis import SpotPriceAnalyzer
    from src.forensic.report_generator import generate_report
    
    start_time = time.time()
    
    print("\n" + "=" * 70)
    print("  FORENSIC PATTERN-FINDER: Kalshi BTC-15M Series")
    print("  Investigating Structural Pricing Bias")
    print("=" * 70)
    
    # --- Auth Setup ---
    api_key = os.environ.get("KALSHI_API_KEY_ID", "")
    key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "kalshi_private_key.pem")
    environment = os.environ.get("ENVIRONMENT", "production")
    
    if not api_key:
        print("\n  ERROR: KALSHI_API_KEY_ID not found in .env")
        print("  Please set your API key in the .env file.")
        return
    
    # Resolve key path relative to project root
    project_root = Path(__file__).parent
    if not Path(key_path).is_absolute():
        key_path = str(project_root / key_path)
    
    if not Path(key_path).exists():
        print(f"\n  ERROR: Private key not found at {key_path}")
        return
    
    print(f"\n  Environment: {environment}")
    print(f"  API Key: {api_key[:8]}...")
    print(f"  Series: {series_ticker}")
    print(f"  Target Markets: {count}")
    print(f"  Spot Analysis: {'Enabled' if not skip_spot else 'Disabled'}")
    
    auth = KalshiAuth(
        api_key_id=api_key,
        private_key_path=key_path,
    )
    
    base_url = (
        "https://api.elections.kalshi.com/trade-api/v2"
        if environment == "production"
        else "https://demo-api.kalshi.co/trade-api/v2"
    )
    
    # ========================================================================
    # STEP 1: DATA ACQUISITION
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("  STEP 1: Fetching settled markets and candlestick data...")
    print("-" * 70)
    
    async with ForensicDataClient(auth=auth, base_url=base_url) as client:
        # 1a. Fetch settled markets
        markets = await client.fetch_settled_markets(
            series_ticker=series_ticker,
            count=count,
        )
        
        print(f"\n  Fetched {len(markets)} settled markets")
        
        if not markets:
            print("\n  ERROR: No settled markets found for series '{series_ticker}'")
            print("  This could mean:")
            print("    - The series ticker is wrong (try KXBTC15M)")
            print("    - The API key doesn't have access")
            print("    - There are no settled markets yet")
            return
        
        # Show sample
        print(f"  Sample: {markets[0].ticker} - Result: {markets[0].result}")
        if len(markets) > 1:
            results = {"yes": 0, "no": 0, "unknown": 0}
            for m in markets:
                r = (m.result or "unknown").lower()
                results[r] = results.get(r, 0) + 1
            print(f"  Results distribution: {dict(results)}")
        
        # 1b. Fetch candlestick data
        print(f"\n  Fetching 1-minute candlestick data for {len(markets)} markets...")
        print("  (This may take a few minutes with rate limiting)")
        
        markets = await client.fetch_all_candlesticks(markets)
        
        with_candles = sum(1 for m in markets if m.candlesticks)
        print(f"\n  Markets with candlestick data: {with_candles}/{len(markets)}")
    
    # ========================================================================
    # STEP 2: FEATURE EXTRACTION
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("  STEP 2: Extracting forensic features...")
    print("-" * 70)
    
    features = extract_all_features(markets)
    
    print(f"\n  Features extracted: {len(features)}")
    
    if not features:
        print("\n  ERROR: No features could be extracted.")
        print("  Markets may not have candlestick data.")
        return
    
    # Quick stats
    longshots = sum(1 for f in features if f.had_longshot_opportunity)
    longshot_wins = sum(1 for f in features if f.longshot_won)
    print(f"  Markets with longshot opportunities: {longshots}")
    print(f"  Longshots that actually won: {longshot_wins}")
    if longshots > 0:
        print(f"  Longshot win rate: {longshot_wins/longshots:.1%}")
    
    # ========================================================================
    # STEP 3: STATISTICAL AUDIT
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("  STEP 3: Running statistical audit...")
    print("-" * 70)
    
    auditor = ForensicAuditor(features)
    audit_report = auditor.run_full_audit()
    
    print(f"\n  Audit complete.")
    print(f"  Golden Minute (T=10, >4x): {audit_report.golden_minute_wins}/{audit_report.golden_minute_total} wins")
    print(f"  Points of No Return found: {len(audit_report.points_of_no_return)}")
    print(f"  '27/29' sequences found: {len(audit_report.worst_sequences)}")
    print(f"  Overall favorite win rate: {audit_report.overall_favorite_win_rate:.1%}")
    print(f"  Longshot EV per $1: ${audit_report.longshot_ev_per_dollar:+.4f}")
    
    # ========================================================================
    # STEP 4: SPOT ANALYSIS (optional)
    # ========================================================================
    
    spot_report = None
    
    if not skip_spot:
        print("\n" + "-" * 70)
        print("  STEP 4: Running spot-lag analysis (Binance BTC/USD)...")
        print("-" * 70)
        
        try:
            async with SpotPriceAnalyzer() as spot_analyzer:
                # Only analyze markets that had longshot opportunities
                longshot_features = [f for f in features if f.had_longshot_opportunity]
                
                if longshot_features:
                    print(f"\n  Analyzing {len(longshot_features)} markets with longshot opportunities...")
                    spot_report = await spot_analyzer.analyze_all(longshot_features)
                    
                    print(f"\n  Spot analysis complete.")
                    print(f"  Markets with spot data: {spot_report.with_spot_data}")
                    print(f"  Trap instances: {spot_report.trap_instances} ({spot_report.trap_rate:.0%})")
                    print(f"  Avg price lag: {spot_report.avg_estimated_lag:.1f} minutes")
                else:
                    print("\n  No longshot opportunities to analyze against spot.")
        
        except Exception as e:
            print(f"\n  Spot analysis failed (non-critical): {e}")
            print("  Continuing without spot data...")
    else:
        print("\n  STEP 4: Spot analysis skipped (--skip-spot)")
    
    # ========================================================================
    # STEP 5: GENERATE REPORT
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("  STEP 5: Generating forensic report...")
    print("-" * 70)
    
    report_text = generate_report(audit_report, spot_report)
    
    # Save to file
    output_path = project_root / output_file
    output_path.write_text(report_text)
    
    elapsed = time.time() - start_time
    
    # Print report to console
    print("\n")
    print(report_text)
    
    print(f"\n\n  Report saved to: {output_path}")
    print(f"  Total analysis time: {elapsed:.1f} seconds")
    print(f"  Markets analyzed: {len(features)}")
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Forensic Pattern-Finder for Kalshi BTC-15M Series",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_forensic.py                          # Full analysis, 200 markets
  python run_forensic.py --count 500              # Analyze 500 markets
  python run_forensic.py --skip-spot              # Skip Binance comparison
  python run_forensic.py --output my_report.txt   # Custom output file
  python run_forensic.py --series KXBTC5M         # Different series
        """,
    )
    
    parser.add_argument(
        "--count", type=int, default=200,
        help="Number of settled markets to analyze (default: 200)",
    )
    parser.add_argument(
        "--skip-spot", action="store_true",
        help="Skip Binance spot-lag analysis",
    )
    parser.add_argument(
        "--output", type=str, default="forensic_report.txt",
        help="Output file for the report (default: forensic_report.txt)",
    )
    parser.add_argument(
        "--series", type=str, default="KXBTC15M",
        help="Series ticker to analyze (default: KXBTC15M)",
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_env()
    
    # Run the analysis
    asyncio.run(
        run_forensic_analysis(
            count=args.count,
            skip_spot=args.skip_spot,
            output_file=args.output,
            series_ticker=args.series,
        )
    )


if __name__ == "__main__":
    main()
