"""
Database factory for creating appropriate database instances based on connection URLs.
"""

from typing import Optional
from src.core.database.base import DatabaseBase


class DatabaseFactory:
    """
    Factory class for creating database instances based on connection URL schemes.
    
    This class keeps the base DatabaseBase class independent of specific implementations.
    """
    
    @classmethod
    def create(cls, url: str, logger, **kwargs) -> DatabaseBase:
        """
        Create the appropriate database instance based on URL scheme.
        
        Args:
            url: Database connection URL (e.g., 'mariadb://...', 'qdrant://...')
            logger: Logger instance to use
            **kwargs: Additional connection parameters
            
        Returns:
            DatabaseBase: Appropriate database instance
            
        Raises:
            ValueError: If the URL scheme is not supported
        """
        if url.startswith('mariadb://'):
            from .mariadb import MariaDatabase
            return MariaDatabase(url, logger=logger, **kwargs)
        # elif url.startswith('mysql://'):
        #     from .mysql import MySQLDatabase
        #     return MySQLDatabase(url, logger=logger, **kwargs)
        elif url.startswith('qdrant://'):
            from .qdrant import QdrantDatabase
            return QdrantDatabase(url, logger=logger, **kwargs)
        elif url.startswith('postgresql://'):
            from .pgsql import PostgreSQLDatabase
            return PostgreSQLDatabase(url, logger=logger, **kwargs)
        else:
            raise ValueError(f"Unsupported database URL scheme: {url.split('://')[0] if '://' in url else 'unknown'}. Supported: mariadb://, qdrant://, postgresql://")
