"""
Entry point for the FastAPI application.
Import the app from the app package to maintain backward compatibility with deployment configs.
"""

from app.main import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
