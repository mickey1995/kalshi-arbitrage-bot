#!/usr/bin/env python3
"""
Quick Setup Script for Kalshi Arbitrage Bot

Run this first to:
1. Check dependencies
2. Generate API keys
3. Test the connection
4. Verify the math works
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and show status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {result.stderr}")
        return False
    print(result.stdout)
    return True


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║           KALSHI ARBITRAGE BOT - SETUP WIZARD                 ║
    ║                                                               ║
    ║  Based on: "Unravelling the Probabilistic Forest"             ║
    ║  Implements: Frank-Wolfe, Bregman Projection, IP Solver       ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Check Python version
    print("\n[1/6] Checking Python version...")
    if sys.version_info < (3, 10):
        print(f"[ERROR] Python 3.10+ required. You have {sys.version}")
        return
    print(f"[OK] Python {sys.version}")
    
    # Step 2: Install dependencies
    print("\n[2/6] Installing dependencies...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
        capture_output=True
    )
    if result.returncode != 0:
        print("[ERROR] Failed to install dependencies")
        print(result.stderr.decode())
        return
    print("[OK] Dependencies installed")
    
    # Step 3: Check for .env file
    print("\n[3/6] Checking configuration...")
    env_file = Path(".env")
    if not env_file.exists():
        print("[INFO] Creating .env from template...")
        import shutil
        shutil.copy(".env.example", ".env")
        print("[OK] Created .env - YOU NEED TO EDIT THIS!")
    else:
        print("[OK] .env file exists")
    
    # Step 4: Check for API key
    print("\n[4/6] Checking API keys...")
    key_file = Path("kalshi_private_key.pem")
    if not key_file.exists():
        print("[INFO] No API key found. Generating new keypair...")
        
        # Generate keys
        from src.api.auth import generate_key_pair
        private_path, public_path = generate_key_pair("kalshi_keys")
        
        # Rename to expected name
        os.rename(private_path, "kalshi_private_key.pem")
        
        print(f"""
[OK] Keys generated!

IMPORTANT - Next steps:
1. Go to https://kalshi.com/account/api-keys
2. Click "Add API Key"  
3. Upload the PUBLIC key: {public_path}
4. Copy the API Key ID and paste it in .env as KALSHI_API_KEY_ID
        """)
    else:
        print("[OK] Private key exists")
    
    # Step 5: Test OR-Tools
    print("\n[5/6] Testing Integer Programming solver...")
    try:
        from ortools.linear_solver import pywraplp
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if solver:
            print("[OK] OR-Tools SCIP solver working")
        else:
            print("[WARN] SCIP not available, trying CBC...")
            solver = pywraplp.Solver.CreateSolver('CBC')
            if solver:
                print("[OK] OR-Tools CBC solver working")
    except ImportError:
        print("[ERROR] OR-Tools not installed properly")
        return
    
    # Step 6: Test the math
    print("\n[6/6] Testing optimization algorithms...")
    try:
        import numpy as np
        from src.optimization import (
            BarrierFrankWolfe,
            LMSRBregman,
            IPSolverFactory,
            MarginalPolytope,
        )
        
        # Create test case: mispriced market
        polytope = MarginalPolytope()
        polytope.add_market("TEST", ["yes", "no"])
        
        ip_solver = IPSolverFactory.create("ortools")
        bregman = LMSRBregman(liquidity=100.0)
        
        fw = BarrierFrankWolfe(
            bregman=bregman,
            ip_solver=ip_solver,
            polytope=polytope,
            alpha=0.9,
            max_iterations=20,
        )
        
        # Mispriced: YES=40c, NO=40c (should sum to 100c)
        theta = np.array([
            bregman.b * np.log(0.4),
            bregman.b * np.log(0.4),
        ])
        
        result = fw.run(theta)
        
        print(f"""
[OK] Math test passed!
    - Iterations: {result.iterations}
    - Divergence D(μ||θ): ${result.divergence:.4f}
    - Gap g(μ): ${result.gap:.6f}
    - Guaranteed Profit: ${result.guaranteed_profit:.4f}
    - Extraction Ratio: {result.extraction_ratio:.1%}
        """)
        
    except Exception as e:
        print(f"[ERROR] Math test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    SETUP COMPLETE!                            ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║                                                               ║
    ║  Before running the bot:                                      ║
    ║                                                               ║
    ║  1. Edit .env and add your KALSHI_API_KEY_ID                  ║
    ║  2. Upload kalshi_keys_public.pem to Kalshi                   ║
    ║  3. Add funds to your Kalshi account                          ║
    ║                                                               ║
    ║  Commands:                                                    ║
    ║    python run.py scan          - Scan for opportunities       ║
    ║    python run.py start --demo  - Start in demo mode           ║
    ║    python run.py start         - Start live trading           ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
