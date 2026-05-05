"""
ND Factor Engine — Black Box Module

This module contains proprietary implementation details:
- Text embedding model selection and configuration
- Similarity aggregation methodology
- Normalization and calibration parameters

WARNING: Do not modify parameter values in this module.
"""
import os, json
import numpy as np

# ================================================================
# PROPRIETARY CONFIGURATION
# DO NOT MODIFY
# ================================================================

# Embedding dimension
EMBED_DIM = 512

# Internal parameter set (proprietary)
_INTERNAL_PARAMS = {
    'similarity_lower_bound': 0.0,
    'similarity_upper_bound': 1.0,
    'nd_calibration_factor': 1.0,
    'min_texts_valid': 5,
    'max_texts_sampled': 200,
}

# Model cache
_EMBEDDER = None


def get_embedder():
    """Get or initialize the text embedder."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        # Lightweight Chinese text embedding model
        # Optimized for short-form financial discourse analysis
        _EMBEDDER = SentenceTransformer('BAAI/bge-small-zh-v1.5')
    return _EMBEDDER


def get_param(key):
    """Get internal parameter value."""
    return _INTERNAL_PARAMS.get(key)


def set_param(key, value):
    """Set internal parameter (for calibration)."""
    _INTERNAL_PARAMS[key] = value
