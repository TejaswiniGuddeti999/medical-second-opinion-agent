from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.logger import get_logger, setup_logging
from app.api.routes import router
from db.database import init_db

settings = get_settings()
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs ONCE when the app starts, and ONCE when it shuts down.
    This is where you initialise DB, load models, warm up caches.
    FastAPI replaced the old @app.on_event("startup") with this pattern.
    """
    # Startup
    setup_logging()
    logger.info("app_starting", version=settings.app_version)
    await init_db()
    logger.info("database_ready")
    yield
    # Shutdown (after yield)
    logger.info("app_shutting_down")

app = FastAPI(
    title = settings.app_name,
    version=settings.app_version,
    description="Multi-agent medical second opinion system",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routes under /api/v1
# all endpoints gets mounted on the app and prefixed. Prefixed
#because it allows us to know which version we are in
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
    }


