#!/usr/bin/env python3
"""
Kalshi Arbitrage Bot - Command Line Interface

Usage:
    python run.py start          # Start the bot
    python run.py scan           # One-time scan for arbitrage
    python run.py status         # Show current status
    python run.py generate-keys  # Generate new API keys
"""

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

console = Console()


@click.group()
def cli():
    """Kalshi Arbitrage Bot - Sophisticated prediction market trading."""
    pass


@cli.command()
@click.option('--demo', is_flag=True, help='Use demo environment')
def start(demo: bool):
    """Start the arbitrage bot."""
    from dotenv import load_dotenv
    load_dotenv()
    
    if demo:
        import os
        os.environ['ENVIRONMENT'] = 'demo'
    
    console.print(Panel.fit(
        "[bold green]Starting Kalshi Arbitrage Bot[/bold green]\n"
        "Press Ctrl+C to stop",
        title="Arbitrage Bot"
    ))
    
    from src.bot import main
    asyncio.run(main())


@cli.command()
@click.option('--top', default=10, help='Number of top opportunities to show')
def scan(top: int):
    """Run a one-time scan for arbitrage opportunities."""
    from dotenv import load_dotenv
    load_dotenv()
    
    async def do_scan():
        from src.config import get_settings
        from src.api import KalshiClient, KalshiAuth
        from src.arbitrage import ArbitrageDetector
        
        settings = get_settings()
        
        console.print("[yellow]Initializing...[/yellow]")
        
        auth = KalshiAuth(
            api_key_id=settings.kalshi_api_key_id,
            private_key_path=settings.kalshi_private_key_path,
        )
        
        client = KalshiClient(
            auth=auth,
            base_url=settings.api_base_url,
        )
        
        detector = ArbitrageDetector(settings)
        
        async with client:
            console.print("[yellow]Fetching top markets by volume...[/yellow]")
            # Fetch only top markets (limited fetch to avoid timeout)
            markets, _ = await client.get_markets(status="open", limit=200)
            
            console.print(f"[green]Fetched {len(markets)} markets[/green]")
            
            # Get orderbooks for top markets by volume
            top_markets = sorted(markets, key=lambda m: m.volume_24h, reverse=True)[:50]
            tickers = [m.ticker for m in top_markets]
            
            console.print("[yellow]Fetching order books...[/yellow]")
            orderbooks = await client.get_orderbooks_parallel(tickers)
            
            console.print("[yellow]Scanning for arbitrage...[/yellow]")
            opportunities = await detector.scan_all(top_markets, orderbooks)
            
            if not opportunities:
                console.print("[red]No arbitrage opportunities found[/red]")
                return
            
            # Display results
            table = Table(title=f"Top {min(top, len(opportunities))} Arbitrage Opportunities")
            table.add_column("Type", style="cyan")
            table.add_column("Ticker(s)", style="magenta")
            table.add_column("Profit", style="green")
            table.add_column("Margin", style="yellow")
            table.add_column("Liquidity", style="blue")
            table.add_column("Risk", style="red")
            
            for opp in opportunities[:top]:
                table.add_row(
                    opp.type,
                    ", ".join(opp.tickers[:2]),
                    f"${opp.guaranteed_profit:.4f}",
                    f"{opp.profit_margin:.1%}",
                    str(int(opp.available_liquidity)),
                    f"{opp.execution_risk_score:.2f}",
                )
            
            console.print(table)
            console.print(f"\n[bold]Total opportunities: {len(opportunities)}[/bold]")
    
    asyncio.run(do_scan())


@cli.command()
def status():
    """Show current bot and market status."""
    from dotenv import load_dotenv
    load_dotenv()
    
    async def show_status():
        from src.config import get_settings
        from src.api import KalshiClient, KalshiAuth
        
        settings = get_settings()
        
        auth = KalshiAuth(
            api_key_id=settings.kalshi_api_key_id,
            private_key_path=settings.kalshi_private_key_path,
        )
        
        client = KalshiClient(
            auth=auth,
            base_url=settings.api_base_url,
        )
        
        async with client:
            balance = await client.get_balance()
            positions = await client.get_positions()
            markets, _ = await client.get_markets(status="open", limit=10)
            
            # Balance panel
            console.print(Panel(
                f"Available: ${balance.available_balance:.2f}\n"
                f"Total: ${balance.total_balance:.2f}",
                title="Account Balance"
            ))
            
            # Positions
            if positions:
                table = Table(title="Open Positions")
                table.add_column("Ticker")
                table.add_column("YES Qty")
                table.add_column("NO Qty")
                table.add_column("P&L")
                
                for pos in positions[:10]:
                    if pos.has_position:
                        table.add_row(
                            pos.ticker,
                            str(pos.yes_quantity),
                            str(pos.no_quantity),
                            f"${pos.realized_pnl:.2f}",
                        )
                
                console.print(table)
            else:
                console.print("[dim]No open positions[/dim]")
            
            # Environment info
            console.print(f"\n[dim]Environment: {settings.environment.value}[/dim]")
            console.print(f"[dim]API URL: {settings.api_base_url}[/dim]")
    
    asyncio.run(show_status())


@cli.command('generate-keys')
@click.option('--output', default='kalshi_keys', help='Output path prefix')
def generate_keys(output: str):
    """Generate RSA key pair for API authentication."""
    from src.api.auth import generate_key_pair
    
    private_path, public_path = generate_key_pair(output)
    
    console.print(Panel.fit(
        f"[green]Keys generated successfully![/green]\n\n"
        f"Private key: {private_path}\n"
        f"Public key: {public_path}\n\n"
        f"[yellow]Upload the PUBLIC key to Kalshi:[/yellow]\n"
        f"https://kalshi.com/account/api-keys\n\n"
        f"[red]Keep the PRIVATE key secure![/red]",
        title="RSA Key Generation"
    ))


@cli.command()
def test():
    """Run a quick test of the optimization algorithms."""
    import numpy as np
    from src.optimization import (
        BarrierFrankWolfe,
        LMSRBregman,
        IPSolverFactory,
        MarginalPolytope,
    )
    
    console.print("[yellow]Testing optimization components...[/yellow]")
    
    # Create simple test polytope (2-outcome market)
    polytope = MarginalPolytope()
    polytope.add_market("TEST", ["yes", "no"])
    
    # Create IP solver
    ip_solver = IPSolverFactory.create("ortools")
    console.print("[green]✓ IP Solver created[/green]")
    
    # Create Bregman divergence
    bregman = LMSRBregman(liquidity=100.0)
    console.print("[green]✓ Bregman divergence created[/green]")
    
    # Create Frank-Wolfe
    fw = BarrierFrankWolfe(
        bregman=bregman,
        ip_solver=ip_solver,
        polytope=polytope,
        max_iterations=20,
    )
    console.print("[green]✓ Barrier Frank-Wolfe created[/green]")
    
    # Test with mispriced market
    # Price YES at 40 cents, NO at 40 cents (sum = 80, arbitrage!)
    theta = np.array([
        bregman.b * np.log(0.4),  # YES
        bregman.b * np.log(0.4),  # NO (mispriced, should sum to 1)
    ])
    
    console.print("\n[yellow]Running Frank-Wolfe on mispriced market...[/yellow]")
    console.print(f"Initial prices: YES=${0.4:.2f}, NO=${0.4:.2f} (sum=${0.8:.2f})")
    
    result = fw.run(theta)
    
    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Iterations: {result.iterations}")
    console.print(f"  Divergence (max profit): ${result.divergence:.4f}")
    console.print(f"  Gap: ${result.gap:.6f}")
    console.print(f"  Guaranteed profit: ${result.guaranteed_profit:.4f}")
    console.print(f"  Extraction ratio: {result.extraction_ratio:.1%}")
    console.print(f"  Time: {result.total_time:.2f}s")
    console.print(f"  Status: {result.status}")
    
    if result.is_profitable:
        console.print("\n[green]✓ Arbitrage detected and quantified![/green]")
    else:
        console.print("\n[red]✗ No profitable arbitrage found[/red]")


if __name__ == "__main__":
    cli()
