"""
AudioLab Database Module
Exports database models, connection management, and Base
"""

from .connection import Base, DatabaseManager, database_manager
from .models import (
    Project,
    Track,
    Clip,
    Effect,
    StemSeparation
)

__all__ = [
    "Base",
    "DatabaseManager",
    "database_manager",
    "Project",
    "Track",
    "Clip",
    "Effect",
    "StemSeparation"
]