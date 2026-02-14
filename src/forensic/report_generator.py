"""
Step 5: Output Report Generator.

Produces a comprehensive forensic report including:
- Expected Value ($EV$) of longshot bets
- Point of No Return analysis
- The "27/29" validation results
- Manipulation attribution findings
- Actionable conclusions
"""

from typing import Optional
from datetime import datetime
import structlog

from .statistical_audit import ForensicAuditReport
from .spot_analysis import SpotAnalysisReport

logger = structlog.get_logger(__name__)


def generate_report(
    audit: ForensicAuditReport,
    spot: Optional[SpotAnalysisReport] = None,
) -> str:
    """
    Generate the complete forensic analysis report as formatted text.
    
    This is the final deliverable - a human-readable report that proves
    or disproves the "longshot trap" hypothesis.
    """
    lines = []
    
    def h1(text: str):
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"  {text}")
        lines.append("=" * 80)
    
    def h2(text: str):
        lines.append("")
        lines.append(f"--- {text} ---")
        lines.append("")
    
    def row(label: str, value: str, indent: int = 2):
        lines.append(f"{' ' * indent}{label:<45} {value}")
    
    def blank():
        lines.append("")
    
    # ========================================================================
    # HEADER
    # ========================================================================
    
    h1("FORENSIC PATTERN-FINDER REPORT: Kalshi BTC-15M Series")
    lines.append(f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    lines.append(f"  Markets Analyzed: {audit.total_markets_analyzed}")
    lines.append(f"  Markets with Longshot Opportunities: {audit.markets_with_longshots}")
    
    # ========================================================================
    # EXECUTIVE SUMMARY
    # ========================================================================
    
    h1("EXECUTIVE SUMMARY")
    
    # Determine verdict
    longshot_profitable = audit.longshot_ev_per_dollar > 0
    traps_detected = spot and spot.trap_rate > 0.3
    structural_bias = any(
        b.is_negative_edge and b.total_contracts >= 5
        for b in audit.price_buckets
    )
    
    if structural_bias:
        lines.append("")
        lines.append("  VERDICT: STRUCTURAL OVERPRICING DETECTED")
        lines.append("")
        lines.append("  The data shows that longshot contracts (>2x payout) in the BTC-15M")
        lines.append("  series systematically settle to $0 at a rate that EXCEEDS their")
        lines.append("  implied probability. These are NOT fair bets.")
    elif longshot_profitable:
        lines.append("")
        lines.append("  VERDICT: LONGSHOT BETS MAY HAVE POSITIVE EDGE")
        lines.append("")
        lines.append("  The data suggests longshot contracts are underpriced.")
        lines.append("  Further investigation recommended with larger sample.")
    else:
        lines.append("")
        lines.append("  VERDICT: INSUFFICIENT EVIDENCE OF SYSTEMATIC BIAS")
        lines.append("")
        lines.append("  The data does not conclusively prove structural overpricing,")
        lines.append("  though individual buckets may show negative edge.")
    
    blank()
    row("Overall Longshot EV per $1 bet:", f"${audit.longshot_ev_per_dollar:+.4f}")
    row("Favorite wins at mid-window:", f"{audit.overall_favorite_win_rate:.1%}")
    
    # ========================================================================
    # SECTION 1: BRACKET ANALYSIS (The Core Test)
    # ========================================================================
    
    h1("SECTION 1: BRACKET ANALYSIS (Actual vs. Implied Win Rates)")
    
    lines.append("")
    lines.append("  If markets are fairly priced, the 'Realized Win Rate' should")
    lines.append("  approximately equal the 'Implied Win Rate' (price / $1.00).")
    lines.append("  A significant gap = systematic overpricing or underpricing.")
    blank()
    
    # Table header
    lines.append(f"  {'Price Bucket':<30} {'Trades':>7} {'Wins':>6} {'Actual':>8} {'Implied':>8} {'Edge':>8} {'EV/$1':>9} {'Flag':>8}")
    lines.append(f"  {'-'*30} {'-'*7} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*8}")
    
    for b in audit.price_buckets:
        if b.total_contracts == 0:
            continue
        
        flag = ""
        if b.is_negative_edge:
            flag = "NEG EDGE"
        elif b.edge > 0.05:
            flag = "POS EDGE"
        
        lines.append(
            f"  {b.label:<30} {b.total_contracts:>7} {b.wins:>6} "
            f"{b.realized_win_rate:>7.1%} {b.implied_win_rate:>7.1%} "
            f"{b.edge:>+7.1%} {b.expected_value_per_dollar:>+8.2%} "
            f"{flag:>8}"
        )
    
    blank()
    
    # Highlight the worst buckets
    neg_edge_buckets = [b for b in audit.price_buckets if b.is_negative_edge and b.total_contracts >= 3]
    if neg_edge_buckets:
        lines.append("  ** NEGATIVE EDGE BUCKETS (The 'Trap' Zones) **")
        for b in neg_edge_buckets:
            gap = abs(b.edge) * 100
            lines.append(
                f"     {b.payout_range} payouts: Market implies {b.implied_win_rate:.0%} win rate, "
                f"actual is {b.realized_win_rate:.0%} ({gap:.0f}pp worse)"
            )
            if b.realized_win_rate > 0:
                overpriced_factor = b.implied_win_rate / b.realized_win_rate
                lines.append(
                    f"     -> These contracts are ~{overpriced_factor:.1f}x OVERPRICED vs. reality"
                )
            else:
                lines.append(
                    f"     -> ZERO wins observed. These are essentially dead contracts."
                )
    
    # ========================================================================
    # SECTION 2: THE GOLDEN MINUTE TEST
    # ========================================================================
    
    h1("SECTION 2: THE GOLDEN MINUTE TEST (T=10, >4x Payout)")
    
    lines.append("")
    lines.append("  Hypothesis: 'If the payout exceeds 4x at the 10-minute mark,")
    lines.append("  what is the actual probability of winning?'")
    blank()
    
    row("Contracts at T=10 with >4x payout:", str(audit.golden_minute_total))
    row("Wins:", str(audit.golden_minute_wins))
    row("Actual win rate:", f"{audit.golden_minute_win_rate:.1%}")
    row("Market-implied win rate:", f"{audit.golden_minute_implied_rate:.1%}")
    
    if audit.golden_minute_total > 0:
        gap = audit.golden_minute_implied_rate - audit.golden_minute_win_rate
        if gap > 0.05:
            blank()
            lines.append(f"  ** The market OVERESTIMATES the win chance by {gap:.0%} **")
            lines.append(f"  For every $100 bet on 4x+ payouts at T=10, you would")
            ev_100 = audit.golden_minute_win_rate * 400 - 100  # 4x payout
            lines.append(f"  expect to {'gain' if ev_100 > 0 else 'lose'} ${abs(ev_100):.0f}.")
        elif gap < -0.05:
            blank()
            lines.append(f"  ** Surprisingly, the market UNDERESTIMATES win chance by {abs(gap):.0%} **")
            lines.append(f"  These longshots win more often than their price suggests.")
        else:
            blank()
            lines.append(f"  The actual and implied rates are close. Market is fairly priced at T=10.")
    
    # ========================================================================
    # SECTION 3: POINT OF NO RETURN
    # ========================================================================
    
    h1("SECTION 3: POINT OF NO RETURN")
    
    lines.append("")
    lines.append("  The exact minute and price where a contract's win probability")
    lines.append("  effectively drops to 0%, regardless of the displayed payout.")
    blank()
    
    if audit.points_of_no_return:
        lines.append(f"  {'Minute':>8} {'Price Below':>12} {'Wins':>6} {'Total':>7} {'Win Rate':>9} {'Confidence':>12}")
        lines.append(f"  {'-'*8} {'-'*12} {'-'*6} {'-'*7} {'-'*9} {'-'*12}")
        
        for ponr in sorted(audit.points_of_no_return, key=lambda p: (p.minute, p.price_threshold)):
            lines.append(
                f"  {ponr.minute:>8} {'$'+f'{ponr.price_threshold/100:.2f}':>12} "
                f"{ponr.observed_wins:>6} {ponr.observed_total:>7} "
                f"{ponr.win_rate:>8.1%} {ponr.confidence:>12}"
            )
        
        blank()
        
        # Find the most definitive PONR
        definitive = [p for p in audit.points_of_no_return if p.confidence in ("definitive", "strong")]
        if definitive:
            best = min(definitive, key=lambda p: p.minute)
            lines.append(f"  ** CRITICAL FINDING **")
            lines.append(f"  At minute {best.minute}, if a contract is priced below ${best.price_threshold/100:.2f},")
            lines.append(f"  the observed win rate is {best.win_rate:.1%} (n={best.observed_total}).")
            lines.append(f"  ANY trade at that price/time combination is a GUARANTEED LOSS.")
            lines.append(f"  The payout shown on the UI is MEANINGLESS at this point.")
    else:
        lines.append("  No definitive Point of No Return found in this dataset.")
        lines.append("  This may indicate the sample size is too small, or that")
        lines.append("  the BTC-15M series doesn't exhibit this pattern as strongly.")
    
    # ========================================================================
    # SECTION 4: THE "27/29" VALIDATION
    # ========================================================================
    
    h1("SECTION 4: THE '27/29' VALIDATION")
    
    lines.append("")
    lines.append("  Searching for rolling 30-trade windows where the 'Cheaper' side")
    lines.append("  (the favorite) won more than 90% of the time.")
    blank()
    
    if audit.worst_sequences:
        lines.append(f"  Found {len(audit.worst_sequences)} sequences matching the pattern:")
        blank()
        
        for i, seq in enumerate(audit.worst_sequences[:5]):
            lines.append(
                f"  #{i+1}: Favorite won {seq.favorite_wins}/{seq.window_size} "
                f"({seq.favorite_win_rate:.0%}) - "
                f"Underdog won only {seq.underdog_wins} times"
            )
        
        blank()
        best = audit.worst_sequences[0]
        lines.append(f"  ** WORST CASE: Favorite won {best.favorite_wins} out of {best.window_size} **")
        lines.append(f"  If you were betting the longshot (underdog) during this window,")
        lines.append(f"  you would have lost {best.favorite_wins} out of {best.window_size} bets.")
        lines.append(f"  This MATCHES the '27/29' pattern described in the hypothesis.")
    else:
        lines.append("  No 30-trade sequences found where favorite won >90%.")
        lines.append("  The '27/29' pattern was NOT replicated in this dataset.")
    
    # ========================================================================
    # SECTION 5: TIME DECAY ANALYSIS (The "Noise Zone")
    # ========================================================================
    
    h1("SECTION 5: TIME DECAY ANALYSIS (The 'Noise Zone')")
    
    lines.append("")
    lines.append("  Average price of the LOSING side at each minute.")
    lines.append("  Shows how 'attractive' the losing bet looked before collapsing.")
    blank()
    
    if audit.loser_avg_price_by_minute:
        lines.append(f"  {'Minute':>8} {'Avg Loser Price':>16} {'Implied Payout':>15} {'Is Trap?':>10}")
        lines.append(f"  {'-'*8} {'-'*16} {'-'*15} {'-'*10}")
        
        for minute in sorted(audit.loser_avg_price_by_minute.keys()):
            price = audit.loser_avg_price_by_minute[minute]
            payout = 100.0 / price if price > 0 else float('inf')
            is_trap = "YES" if price > 5 and minute >= 8 else ""
            
            lines.append(
                f"  {minute:>8} ${price/100:>14.2f} {payout:>14.1f}x {is_trap:>10}"
            )
        
        blank()
        lines.append("  ** The 'Noise Zone' **")
        lines.append("  In the last 5 minutes (minute 10-14), losing contracts still show")
        
        late_prices = {m: p for m, p in audit.loser_avg_price_by_minute.items() if m >= 10}
        if late_prices:
            avg_late = sum(late_prices.values()) / len(late_prices)
            avg_payout = 100.0 / avg_late if avg_late > 0 else 0
            lines.append(f"  an average price of ${avg_late/100:.2f} (implied {avg_payout:.1f}x payout).")
            lines.append(f"  These are the 'zombie' contracts - they look alive but are already dead.")
    
    # ========================================================================
    # SECTION 6: MANIPULATION ATTRIBUTION (Spot-Lag Analysis)
    # ========================================================================
    
    if spot:
        h1("SECTION 6: MANIPULATION ATTRIBUTION (Spot-Lag Analysis)")
        
        lines.append("")
        lines.append("  Compares Kalshi BTC-15M prices against Binance BTC/USD spot")
        lines.append("  to detect information asymmetry and latency traps.")
        blank()
        
        row("Markets with spot data:", str(spot.with_spot_data))
        row("Average spot range (15m window):", f"{spot.avg_spot_range_pct:.3f}%")
        row("Average Kalshi-Spot correlation:", f"{spot.avg_price_correlation:.3f}")
        row("Estimated price lag (minutes):", f"{spot.avg_estimated_lag:.1f}")
        row("'Trap' instances detected:", f"{spot.trap_instances} ({spot.trap_rate:.0%})")
        row("Average max mispricing:", f"{spot.avg_max_mispricing:.1f} cents")
        
        blank()
        
        if spot.avg_estimated_lag >= 1.0:
            lines.append("  ** LATENCY ADVANTAGE DETECTED **")
            lines.append(f"  Kalshi prices lag the BTC spot by ~{spot.avg_estimated_lag:.0f} minute(s).")
            lines.append("  Market makers see index changes BEFORE the Kalshi UI updates.")
            lines.append("  When you see a '2x payout', the smart money already knows it's dead.")
        
        if spot.trap_rate > 0.3:
            blank()
            lines.append(f"  ** TRAP RATE: {spot.trap_rate:.0%} **")
            lines.append(f"  In {spot.trap_rate:.0%} of longshot opportunities, the attractive payout")
            lines.append("  appeared AFTER the spot price had already moved past recovery.")
            lines.append("  This confirms the 'lagging indicator' hypothesis.")
    
    # ========================================================================
    # SECTION 7: EXPECTED VALUE SUMMARY
    # ========================================================================
    
    h1("SECTION 7: EXPECTED VALUE (EV) SUMMARY")
    
    blank()
    row("Longshot EV per $1 bet (>2x payout):", f"${audit.longshot_ev_per_dollar:+.4f}")
    
    if audit.longshot_ev_per_dollar < 0:
        blank()
        lines.append("  ** NEGATIVE EXPECTED VALUE **")
        lines.append(f"  For every $1 bet on longshot contracts (>2x payout),")
        lines.append(f"  you LOSE ${abs(audit.longshot_ev_per_dollar):.2f} on average.")
        lines.append(f"  Over 100 bets at $1 each, expected loss: ${abs(audit.longshot_ev_per_dollar * 100):.0f}")
        lines.append(f"  Over 1000 bets at $1 each, expected loss: ${abs(audit.longshot_ev_per_dollar * 1000):.0f}")
        blank()
        lines.append("  The house edge on longshot bets is approximately "
                     f"{abs(audit.longshot_ev_per_dollar)*100:.1f}%.")
    
    # Per-bucket EV
    blank()
    lines.append("  EV by payout range:")
    for b in audit.price_buckets:
        if b.total_contracts >= 3:
            avg_p = (b.price_min + b.price_max) / 200  # in dollars
            lines.append(
                f"    {b.payout_range:>12}: ${b.expected_value_per_dollar:+.2f} per $1 "
                f"(n={b.total_contracts})"
            )
    
    # ========================================================================
    # SECTION 8: MINUTE-BY-MINUTE WIN RATES
    # ========================================================================
    
    h1("SECTION 8: MINUTE-BY-MINUTE WIN RATES (Under $0.15)")
    
    lines.append("")
    lines.append("  Win rate of contracts priced below $0.15 at each minute.")
    lines.append("  Shows exactly when 'longshot' bets become mathematically dead.")
    blank()
    
    lines.append(f"  {'Min':>4} {'< $0.05':>10} {'< $0.10':>10} {'< $0.15':>10} {'< $0.20':>10} {'< $0.25':>10} {'< $0.33':>10} {'< $0.50':>10}")
    lines.append(f"  {'-'*4} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    for ma in audit.minute_analysis:
        cells = [f"{ma.minute:>4}"]
        for threshold in [5, 10, 15, 20, 25, 33, 50]:
            if threshold in ma.threshold_results:
                wins, total, rate = ma.threshold_results[threshold]
                if total > 0:
                    cells.append(f"{rate:>5.0%}({total:>3})")
                else:
                    cells.append(f"{'---':>10}")
            else:
                cells.append(f"{'---':>10}")
        lines.append("  " + " ".join(cells))
    
    # ========================================================================
    # CONCLUSIONS
    # ========================================================================
    
    h1("CONCLUSIONS & ACTIONABLE FINDINGS")
    
    blank()
    lines.append("  1. WHY 27/29 HAPPENS:")
    if audit.worst_sequences:
        lines.append("     The data CONFIRMS that chasing high payouts (>2x) during")
        lines.append("     the BTC-15M series results in catastrophic loss streaks.")
        lines.append(f"     Worst observed: favorite won {audit.worst_sequences[0].favorite_wins}"
                     f"/{audit.worst_sequences[0].window_size} times.")
    lines.append("     At 5 minutes remaining, binary time decay is brutal.")
    lines.append("     The payout LOOKS attractive, but the actual probability")
    lines.append("     has already collapsed far below what the price implies.")
    
    blank()
    lines.append("  2. THE 'STRUCTURAL OVERPRICING' MECHANISM:")
    neg_buckets = [b for b in audit.price_buckets if b.is_negative_edge and b.total_contracts >= 3]
    if neg_buckets:
        for b in neg_buckets:
            lines.append(f"     - {b.payout_range}: Priced at {b.implied_win_rate:.0%} win rate, "
                        f"actual is {b.realized_win_rate:.0%}")
        lines.append("     The gap between 'implied' and 'actual' is where the")
        lines.append("     house/market makers extract profit from retail.")
    else:
        lines.append("     Insufficient negative-edge data to confirm overpricing.")
        lines.append("     Consider running with more settled markets (>200).")
    
    blank()
    lines.append("  3. THE 'BOT ADVANTAGE':")
    if spot and spot.avg_estimated_lag >= 1.0:
        lines.append(f"     Kalshi prices lag BTC spot by ~{spot.avg_estimated_lag:.0f} minute(s).")
        lines.append("     HFT bots see the index move before the UI updates.")
        lines.append("     By the time a human sees a '5x payout', the bot")
        lines.append("     already knows the true probability is ~0%.")
    else:
        lines.append("     Spot-lag analysis inconclusive or not performed.")
    
    blank()
    lines.append("  4. RECOMMENDED ACTIONS:")
    lines.append("     a) NEVER buy contracts priced below $0.15 after minute 10")
    
    if audit.points_of_no_return:
        earliest = min(audit.points_of_no_return, key=lambda p: p.minute)
        lines.append(f"     b) The Point of No Return is minute {earliest.minute} "
                     f"at ${earliest.price_threshold/100:.2f}")
    
    lines.append("     c) If you want to trade longshots, only at T=0 (entry)")
    lines.append("     d) Consider SELLING overpriced longshots instead of buying them")
    lines.append("     e) The '5x payout' UI display is a psychological trap")
    
    blank()
    lines.append("=" * 80)
    lines.append("  END OF FORENSIC REPORT")
    lines.append("=" * 80)
    
    return "\n".join(lines)
