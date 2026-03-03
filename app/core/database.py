"""
MySQL connection pool (SQLAlchemy 2.0 + aiomysql).

Usage:
    from app.core.database import get_db, async_session_maker, init_db, close_db

    # In FastAPI route:
    @router.get("/")
    async def index(db: AsyncSession = Depends(get_db)):
        ...

    # In lifespan: init_db() on startup, close_db() on shutdown.
"""
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings

logger = logging.getLogger(__name__)

_async_engine = None
async_session_maker = None


def get_engine():
    """Return the global async engine (create on first use)."""
    global _async_engine, async_session_maker
    if _async_engine is not None:
        return _async_engine

    url = settings.get_database_url()
    _async_engine = create_async_engine(
        url,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_recycle=settings.database_pool_recycle,
        echo=settings.database_echo,
        pool_pre_ping=True,
    )

    async_session_maker = async_sessionmaker(
        _async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
    logger.info("MySQL connection pool and session factory created")
    return _async_engine


async def init_db() -> None:
    """Initialize MySQL connection pool (call on app startup)."""
    get_engine()
    # Optional: create tables if using Base.metadata
    # async with _async_engine.begin() as conn:
    #     await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Dispose MySQL connection pool (call on app shutdown)."""
    global _async_engine, async_session_maker
    if _async_engine is not None:
        await _async_engine.dispose()
        _async_engine = None
        async_session_maker = None
        logger.info("MySQL connection pool disposed")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency: yield an async session, ensure it is closed after request.
    """
    if async_session_maker is None:
        get_engine()
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_session_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for use outside FastAPI (e.g. background tasks, scripts).
    """
    if async_session_maker is None:
        get_engine()
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
