"""
Churn Prediction API package.

This package exposes the FastAPI application used for serving
the churn prediction model.
"""

from .main import app

__all__ = ["app"]