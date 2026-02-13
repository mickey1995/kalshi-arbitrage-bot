"""
Configuration management for Kalshi Arbitrage Bot.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr
from typing import Optional
from enum import Enum
import os


class Environment(str, Enum):
    """Environment types."""
    DEMO = "demo"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Environment
    environment: Environment = Field(
        default=Environment.DEMO,
        description="Trading environment (demo or production)"
    )
    
    # Kalshi API Credentials
    kalshi_api_key_id: str = Field(
        ...,
        description="Kalshi API Key ID"
    )
    kalshi_private_key_path: str = Field(
        default="kalshi_private_key.pem",
        description="Path to RSA private key for API authentication"
    )
    
    # API URLs
    @property
    def api_base_url(self) -> str:
        if self.environment == Environment.DEMO:
            return "https://demo-api.kalshi.co/trade-api/v2"
        return "https://api.elections.kalshi.com/trade-api/v2"
    
    @property
    def ws_base_url(self) -> str:
        if self.environment == Environment.DEMO:
            return "wss://demo-api.kalshi.co/trade-api/ws/v2"
        return "wss://api.elections.kalshi.com/trade-api/ws/v2"
    
    # Trading Parameters
    min_profit_threshold: float = Field(
        default=0.01,
        description="Minimum profit threshold in dollars (default $0.01 for learning)"
    )
    alpha_extraction: float = Field(
        default=0.9,
        description="Alpha parameter for profit extraction (stop when capturing 90% of arbitrage)"
    )
    max_position_size_fraction: float = Field(
        default=0.1,
        description="Maximum fraction of order book depth for position sizing"
    )
    max_drawdown_percent: float = Field(
        default=20.0,
        description="Maximum drawdown percentage before halting"
    )
    
    # Learning Mode (for making hundreds of small bets)
    learning_mode: bool = Field(
        default=True,
        description="Enable learning mode with fixed small bet sizes"
    )
    learning_bet_size: float = Field(
        default=1.0,
        description="Fixed bet size in dollars for learning mode"
    )
    max_daily_trades: int = Field(
        default=500,
        description="Maximum trades per day in learning mode"
    )
    
    # Frank-Wolfe Parameters
    fw_initial_epsilon: float = Field(
        default=0.1,
        description="Initial contraction parameter for Barrier Frank-Wolfe"
    )
    fw_convergence_threshold: float = Field(
        default=1e-6,
        description="Convergence threshold for Frank-Wolfe gap"
    )
    fw_max_iterations: int = Field(
        default=150,
        description="Maximum Frank-Wolfe iterations"
    )
    fw_time_limit_seconds: float = Field(
        default=1800,
        description="Time limit for Frank-Wolfe optimization (30 minutes)"
    )
    
    # Integer Programming
    ip_solver: str = Field(
        default="ortools",
        description="IP solver to use: 'ortools', 'gurobi', or 'scip'"
    )
    ip_time_limit_seconds: float = Field(
        default=30,
        description="Time limit per IP solve"
    )
    
    # Rate Limiting
    api_requests_per_minute: int = Field(
        default=100,
        description="Maximum API requests per minute"
    )
    
    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///kalshi_arbitrage.db",
        description="Database connection URL"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_file: Optional[str] = Field(
        default="logs/arbitrage_bot.log",
        description="Log file path"
    )
    
    # LLM for Dependency Detection (optional)
    openai_api_key: Optional[SecretStr] = Field(
        default=None,
        description="OpenAI API key for dependency detection"
    )
    anthropic_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Anthropic API key for dependency detection"
    )
    
    # Monitoring
    enable_prometheus: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )
    prometheus_port: int = Field(
        default=9090,
        description="Prometheus metrics port"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
