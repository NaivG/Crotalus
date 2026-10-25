"""
Minecraft Protocol Encryption Helper

Provides RSA key generation and shared secret decryption for Minecraft server encryption.
Uses cryptography library for secure operations.
"""

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import secrets


class CryptoHelper:
    """Helper class for Minecraft protocol encryption."""
    
    def __init__(self):
        """Generate RSA key pair (1024 bits as per Minecraft protocol)."""
        self._private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=1024,
            backend=default_backend()
        )
        self._public_key = self._private_key.public_key()
    
    @property
    def public_key_bytes(self) -> bytes:
        """Get public key in DER format (SubjectPublicKeyInfo)."""
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    
    def decrypt_shared_secret(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt shared secret from client using RSA private key.
        
        Args:
            encrypted_data: Encrypted shared secret
            
        Returns:
            Decrypted shared secret
        """
        return self._private_key.decrypt(
            encrypted_data,
            padding.PKCS1v15()
        )
    
    @staticmethod
    def generate_verify_token(length: int = 4) -> bytes:
        """
        Generate random verify token.
        
        Args:
            length: Token length in bytes (default: 4)
            
        Returns:
            Random bytes
        """
        return secrets.token_bytes(length)
    
    @staticmethod
    def generate_server_id(length: int = 4) -> str:
        """
        Generate random server ID (hexadecimal string).
        
        Args:
            length: Length in bytes (default: 4, produces 8-char hex string)
            
        Returns:
            Hexadecimal string
        """
        return secrets.token_hex(length)
