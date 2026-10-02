"""
Small shared helpers used across the project.
Keep this file lean — RAG-specific helpers go in rag/, graph-specific in graph/.
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """Standard logger setup so every module logs consistently."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
