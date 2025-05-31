import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Any, Union
import requests
import structlog
from ..config import settings

logger = structlog.get_logger(__name__)

# Database path for caching embeddings
DB_PATH = settings.cache_dir / "embeddings.sqlite"


def _setup_db() -> sqlite3.Connection:
    """
    Set up the SQLite database for caching embeddings.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cache (digest TEXT PRIMARY KEY, vec TEXT, created_at INTEGER)"
    )
    return conn


def _db() -> sqlite3.Connection:
    """
    Get a database connection, ensuring table exists.
    """
    return _setup_db()


def embed(text: str, api_key: str = None) -> List[float]:
    """
    Generate OpenAI embeddings for text with caching.

    Args:
        text: The text to generate embeddings for
        api_key: OpenAI API key (falls back to settings if not provided)

    Returns:
        List of floats representing the embedding vector
    """
    if not text:
        logger.warning("Empty text provided for embedding")
        return []

    # Use API key from function or fall back to settings
    api_key = api_key or settings.openai_api_key
    if not api_key:
        logger.error("No OpenAI API key provided")
        raise ValueError("OpenAI API key is required for embeddings")

    # Calculate hash of input for caching
    digest = hashlib.sha256(text.encode()).hexdigest()

    # Check cache first
    with _db() as conn:
        row = conn.execute("SELECT vec FROM cache WHERE digest = ?", (digest,)).fetchone()
        if row:
            logger.debug("Cache hit for embedding")
            return json.loads(row[0])

    logger.info("Generating embedding via OpenAI API")

    # Prepare API request
    payload = {"input": text, "model": "text-embedding-3-small"}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Attempt API call with retries for rate limiting
    for attempt in range(5):
        try:
            resp = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=payload,
                timeout=30
            )

            if resp.ok:
                vec = resp.json()["data"][0]["embedding"]

                # Store in cache
                with _db() as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO cache VALUES (?, ?, ?)",
                        (digest, json.dumps(vec), int(time.time()))
                    )

                return vec

            elif resp.status_code == 429:  # Rate limit
                wait_time = 2 ** attempt
                logger.warning(f"Rate limited by OpenAI API, waiting {wait_time}s")
                time.sleep(wait_time)
                continue

            else:
                logger.error("OpenAI API error",
                             status_code=resp.status_code,
                             response=resp.text)
                raise RuntimeError(f"OpenAI API error: {resp.text}")

        except requests.RequestException as e:
            logger.error("Request exception calling OpenAI API",
                         attempt=attempt + 1,
                         error=str(e))
            if attempt < 4:  # Don't sleep on last attempt
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Failed to call OpenAI API: {str(e)}")

    raise RuntimeError("Max retries reached for OpenAI API")


def clear_cache() -> int:
    """
    Clear the embeddings cache.

    Returns:
        Number of entries removed
    """
    try:
        with _db() as conn:
            cursor = conn.execute("DELETE FROM cache")
            return cursor.rowcount
    except sqlite3.Error as e:
        logger.error("Error clearing cache", error=str(e))
        return 0