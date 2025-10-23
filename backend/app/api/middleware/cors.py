"""
CORS Middleware Configuration

Handles Cross-Origin Resource Sharing (CORS) for the application.
Includes both FastAPI CORS middleware and custom middleware for OPTIONS requests.
"""

import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def setup_cors(app: FastAPI) -> None:
    """
    Configure CORS middleware for the FastAPI application.
    
    Sets up:
    1. FastAPI's built-in CORSMiddleware
    2. Custom middleware to handle OPTIONS preflight requests
    
    Args:
        app: FastAPI application instance
    """
    # CORS configuration
    origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        f"http://{os.getenv('HOST_IP', '0.0.0.0')}:3000",
        "http://192.168.8.205:3000",
        "*"  # Allow all origins in development
    ]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex=".*",
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["*", "Authorization", "Content-Type", "X-Requested-With"],
        expose_headers=["*"],
        max_age=86400,
    )
    
    # Custom middleware to handle CORS issues
    @app.middleware("http")
    async def cors_middleware(request: Request, call_next):
        """Handle CORS for OPTIONS requests"""
        if request.method == "OPTIONS":
            logger.info(f"Handling OPTIONS request for {request.url.path}")
            headers = {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
                "Access-Control-Allow-Headers": "Authorization, Content-Type, Accept, X-Requested-With",
                "Access-Control-Max-Age": "86400",
                "Access-Control-Allow-Credentials": "true",
            }
            return JSONResponse(
                content={},
                status_code=200,
                headers=headers
            )
        
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response
    
    logger.info("CORS middleware configured successfully")
