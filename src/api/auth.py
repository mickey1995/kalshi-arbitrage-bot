"""
Kalshi API Authentication using RSA-PSS signatures.
"""

import base64
import time
from pathlib import Path
from typing import Optional
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend
import structlog

logger = structlog.get_logger(__name__)


class KalshiAuth:
    """
    Handles Kalshi API authentication using RSA-PSS signatures.
    
    Kalshi requires:
    1. KALSHI-ACCESS-KEY header with your API key ID
    2. KALSHI-ACCESS-SIGNATURE header with RSA-PSS signature
    3. KALSHI-ACCESS-TIMESTAMP header with millisecond timestamp
    
    The signature is created over: timestamp + method + path
    """
    
    def __init__(
        self,
        api_key_id: str,
        private_key_path: Optional[str] = None,
        private_key_pem: Optional[str] = None,
    ):
        """
        Initialize authentication.
        
        Args:
            api_key_id: Your Kalshi API key ID
            private_key_path: Path to RSA private key PEM file
            private_key_pem: RSA private key as PEM string (alternative to path)
        """
        self.api_key_id = api_key_id
        self._private_key = self._load_private_key(private_key_path, private_key_pem)
        logger.info("kalshi_auth_initialized", api_key_id=api_key_id[:8] + "...")
    
    def _load_private_key(
        self,
        key_path: Optional[str],
        key_pem: Optional[str],
    ) -> rsa.RSAPrivateKey:
        """Load RSA private key from file or PEM string."""
        if key_pem:
            key_bytes = key_pem.encode('utf-8')
        elif key_path:
            path = Path(key_path)
            if not path.exists():
                raise FileNotFoundError(f"Private key file not found: {key_path}")
            key_bytes = path.read_bytes()
        else:
            raise ValueError("Either private_key_path or private_key_pem must be provided")
        
        private_key = serialization.load_pem_private_key(
            key_bytes,
            password=None,
            backend=default_backend()
        )
        
        if not isinstance(private_key, rsa.RSAPrivateKey):
            raise ValueError("Private key must be RSA")
        
        return private_key
    
    def create_signature(
        self,
        timestamp: str,
        method: str,
        path: str,
    ) -> str:
        """
        Create RSA-PSS signature for API request.
        
        Args:
            timestamp: Millisecond timestamp as string
            method: HTTP method (GET, POST, etc.)
            path: API path (e.g., /trade-api/v2/markets)
            
        Returns:
            Base64-encoded signature
        """
        # Message to sign: timestamp + method + path
        message = f"{timestamp}{method.upper()}{path}"
        message_bytes = message.encode('utf-8')
        
        # Sign using RSA-PSS with SHA-256
        signature = self._private_key.sign(
            message_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )
        
        # Return base64-encoded signature
        return base64.b64encode(signature).decode('utf-8')
    
    def get_headers(self, method: str, path: str) -> dict:
        """
        Get authentication headers for API request.
        
        Args:
            method: HTTP method
            path: API path (including /trade-api/v2 prefix)
            
        Returns:
            Dictionary of headers
        """
        # Timestamp in milliseconds
        timestamp = str(int(time.time() * 1000))
        
        # Create signature
        signature = self.create_signature(timestamp, method, path)
        
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }
    
    def get_ws_auth_message(self) -> dict:
        """
        Get WebSocket authentication message.
        
        Returns:
            Dictionary to send as first WebSocket message
        """
        timestamp = str(int(time.time() * 1000))
        
        # For WebSocket, sign the path used for connection
        path = "/trade-api/ws/v2"
        signature = self.create_signature(timestamp, "GET", path)
        
        return {
            "type": "auth",
            "api_key": self.api_key_id,
            "timestamp": timestamp,
            "signature": signature,
        }


def generate_key_pair(output_path: str = "kalshi_keys") -> tuple[str, str]:
    """
    Generate a new RSA key pair for Kalshi API.
    
    Args:
        output_path: Base path for output files (will create .pem and .pub files)
        
    Returns:
        Tuple of (private_key_path, public_key_path)
    """
    # Generate 4096-bit RSA key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=4096,
        backend=default_backend()
    )
    
    # Serialize private key
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    # Serialize public key
    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    # Write to files
    private_path = f"{output_path}_private.pem"
    public_path = f"{output_path}_public.pem"
    
    Path(private_path).write_bytes(private_pem)
    Path(public_path).write_bytes(public_pem)
    
    logger.info(
        "generated_key_pair",
        private_key_path=private_path,
        public_key_path=public_path
    )
    
    return private_path, public_path
