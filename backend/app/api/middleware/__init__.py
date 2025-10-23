"""
API Middleware Package

This package contains custom middleware for the FastAPI application.
"""

from .cors import setup_cors

__all__ = ['setup_cors']
