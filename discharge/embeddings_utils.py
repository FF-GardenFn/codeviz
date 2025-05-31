#!/usr/bin/env python3
"""
embeddings_utils.py

Utilities for generating and working with embeddings.
"""
import os
import logging
import sys
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('embeddings_utils')

# Try to import optional dependencies
try:
    import numpy as np
    USE_NUMPY = True
except ImportError:
    USE_NUMPY = False
    import math
    logger.warning("NumPy not available, using slower math library fallback")

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    # Create a session with retry logic
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    session = None
    logger.warning("Requests library not available, embedding features will use fallback")


def get_embedding(text: str, api_key: Optional[str] = None) -> List[float]:
    """
    Get embedding for text using OpenAI API or fallback.

    Args:
        text: Text to get embedding for
        api_key: OpenAI API key

    Returns:
        Embedding vector
    """
    if not text.strip():
        return []

    # Truncate if too long
    max_length = 8191
    if len(text) > max_length:
        text = text[:max_length]

    # Use the API if available
    if api_key and OPENAI_AVAILABLE:
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            payload = {
                "input": text,
                "model": "text-embedding-3-small"
            }

            response = session.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                return response.json()["data"][0]["embedding"]
            else:
                logger.warning(f"OpenAI API error: {response.status_code}")
        except Exception as e:
            logger.warning(f"Error getting OpenAI embedding: {str(e)}")

    # Fallback to basic embedding
    return basic_embedding(text)


def basic_embedding(text: str) -> List[float]:
    """
    Create a basic embedding for text when OpenAI API is not available.

    Args:
        text: Text to embed

    Returns:
        A simple embedding vector
    """
    # Use a hash-based approach for more consistency
    char_counts = {}
    text = text.lower()

    # Count character frequencies
    for char in text:
        if char in char_counts:
            char_counts[char] += 1
        else:
            char_counts[char] = 1

    # Use common ASCII characters as dimensions
    dimensions = [ord(c) for c in 'abcdefghijklmnopqrstuvwxyz0123456789.,!?:;()[]{}\'"-+*/&^%$#@~<>=\\|']

    # Create a vector with frequency counts
    embedding = [char_counts.get(chr(d), 0) for d in dimensions]

    # Normalize
    total = sum(embedding) or 1
    embedding = [v / total for v in embedding]

    # Add hash-based components for more entropy
    import hashlib
    hash_obj = hashlib.sha256(text.encode())
    hash_bytes = hash_obj.digest()

    for i in range(min(32, len(dimensions))):
        if i < len(embedding):
            embedding[i] = (embedding[i] + hash_bytes[i] / 255.0) / 2

    # Pad to 1024 dimensions
    embedding = embedding + [0] * (1024 - len(embedding))

    return embedding


def compute_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Args:
        vec1: First embedding vector
        vec2: Second embedding vector

    Returns:
        Cosine similarity score
    """
    if not vec1 or not vec2:
        return 0.0

    if len(vec1) != len(vec2):
        # Pad to make equal length
        if len(vec1) < len(vec2):
            vec1 = vec1 + [0] * (len(vec2) - len(vec1))
        else:
            vec2 = vec2 + [0] * (len(vec1) - len(vec2))

    if USE_NUMPY:
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1_np, vec2_np) / (norm1 * norm2))
    else:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in text.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        pass

    # Fallback estimation
    words = text.split()
    chars = len(text)

    # You can use your own word and character count for estimation
    word_estimate = len(words) * 1.33  # ~1.33 tokens per word
    char_estimate = chars / 4  # ~4 chars per token

    return int((word_estimate + char_estimate) / 2)