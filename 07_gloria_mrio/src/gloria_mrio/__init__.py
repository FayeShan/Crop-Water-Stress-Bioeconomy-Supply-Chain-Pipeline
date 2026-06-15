"""GLORIA MRIO supply-chain footprint decomposition (Cabernard, Pfister & Hellweg, 2019)."""

from .config import Config
from .calculator import FootprintCalculator
from .index_builder import IndexBuilder
from .mrio_loader import MRIOLoader
from .runner import GloriaMRIO

__version__ = "0.1.0"
__all__ = [
    "Config",
    "FootprintCalculator", 
    "IndexBuilder",
    "MRIOLoader",
    "GloriaMRIO",
]
