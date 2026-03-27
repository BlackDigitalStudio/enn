"""
Agentic GraphRAG - Entry Point
python -m backend
"""

import uvicorn
from .main import app
from .config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
