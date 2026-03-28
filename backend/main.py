"""
Tree Base - Universal Knowledge Graph
Graph navigation + entity extraction for any content type.
No embeddings, no vector search — pure graph intelligence.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .api.routes import router, set_storage
from .graph.storage import Neo4jStorage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    neo4j = Neo4jStorage(uri=settings.neo4j_uri, user=settings.neo4j_user, password=settings.neo4j_password)
    if neo4j.connect():
        logger.info("Neo4j connected")
        set_storage(neo4j)
    else:
        logger.warning("Neo4j connection failed")
    yield
    neo4j.close()


app = FastAPI(
    title="Tree Base",
    description="Universal Knowledge Graph — graph navigation + entity extraction for any content.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(router)

import pathlib
_static_dir = pathlib.Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/")
async def root():
    index = _static_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"name": "Tree Base", "version": "2.0.0", "docs": "/docs"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": str(exc)})
