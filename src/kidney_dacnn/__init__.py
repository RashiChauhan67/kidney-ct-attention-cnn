"""Dual-attention CNN for kidney ailment classification from CT images."""

from .model import NOTEBOOK_DENSE_UNITS, PAPER_FIGURE_DENSE_UNITS, build_model, compile_model

__all__ = ["build_model", "compile_model", "NOTEBOOK_DENSE_UNITS", "PAPER_FIGURE_DENSE_UNITS"]
