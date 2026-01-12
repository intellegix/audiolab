"""
AudioLab Database Connection Manager
Async database connections for PostgreSQL and Redis
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import asyncpg
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.orm import declarative_base

from ..core.config import get_settings
from ..core.logging import performance_logger

settings = get_settings()

# SQLAlchemy base for models
Base = declarative_base()


class DatabaseManager:
    """Manages database connections and sessions"""

    def __init__(self):
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._redis_pool: Optional[redis.ConnectionPool] = None
        self._redis: Optional[redis.Redis] = None

    async def initialize(self) -> None:
        """Initialize database connections"""

        # Initialize PostgreSQL
        self._engine = create_async_engine(
            settings.DATABASE_URL,
            pool_size=settings.DATABASE_POOL_SIZE,
            max_overflow=settings.DATABASE_MAX_OVERFLOW,
            echo=settings.is_development,  # Log SQL queries in dev
            pool_pre_ping=True,  # Verify connections before use
        )

        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

        # Initialize Redis
        self._redis_pool = redis.ConnectionPool.from_url(
            settings.REDIS_URL,
            max_connections=20,
            retry_on_timeout=True
        )
        self._redis = redis.Redis(connection_pool=self._redis_pool)

        # Test connections
        await self.check_health()

    async def close(self) -> None:
        """Close database connections"""
        if self._engine:
            await self._engine.dispose()

        if self._redis_pool:
            await self._redis_pool.disconnect()

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session"""
        if not self._session_factory:
            raise RuntimeError("Database not initialized")

        async with self._session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    def get_redis(self) -> redis.Redis:
        """Get Redis client"""
        if not self._redis:
            raise RuntimeError("Redis not initialized")
        return self._redis

    async def check_health(self) -> bool:
        """Check database health"""
        try:
            # Check PostgreSQL
            async with self.get_session() as session:
                result = await session.execute("SELECT 1")
                result.fetchone()

            # Check Redis
            await self._redis.ping()

            return True

        except Exception as e:
            performance_logger.logger.error(f"Database health check failed: {e}")
            return False


# Global database manager instance
database_manager = DatabaseManager()