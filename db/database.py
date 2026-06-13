from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.config import get_settings

settings = get_settings()

# The engine is the actual connection to PostgreSQL
# pool_pre_ping=True — tests connection before using it, handles dropped connections
engine = create_async_engine(
    settings.database_url,
    pool_pre_ping=True,
    echo=settings.debug,
    connect_args={"ssl": "require"},
)

# Session factory — creates new DB sessions on demand
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """
    All SQLAlchemy models inherit from this.
    It knows about all your tables and creates them automatically.
    """
    pass


async def init_db():
    """Creates all tables on startup if they don't exist."""
    async with engine.begin() as conn:
        from db import models   # import here to ensure models are registered
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """
    FastAPI dependency — yields a DB session for each request.
    Automatically closes the session when the request finishes.
    The try/finally ensures cleanup even if the request crashes.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()