import os
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, AuthenticationError, OpenAI, RateLimitError

MODEL_NAME = "text-embedding-3-small"

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


def _create_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to the project's .env file."
        )

    return OpenAI(api_key=api_key)


def create_embedding(text):
    """Create one embedding vector for a non-empty string."""
    if not isinstance(text, str):
        raise ValueError("text must be a string.")
    if not text.strip():
        raise ValueError("Cannot create an embedding for empty text.")

    try:
        response = _create_client().embeddings.create(
            model=MODEL_NAME,
            input=text,
        )
    except ValueError:
        raise
    except RateLimitError as error:
        raise RuntimeError(
            "OpenAI rejected the request because the account has no available "
            "credits or has reached its usage limit."
        ) from error
    except AuthenticationError as error:
        raise RuntimeError(
            "OpenAI rejected the API key. Check OPENAI_API_KEY in .env."
        ) from error
    except APIConnectionError as error:
        raise RuntimeError(
            "Could not connect to the OpenAI API. Check your internet connection."
        ) from error
    except Exception as error:
        raise RuntimeError("The OpenAI embedding request failed.") from error

    return response.data[0].embedding


def create_embeddings(chunks):
    """Create embedding vectors for chunks in their original order."""
    if not isinstance(chunks, list):
        raise ValueError("chunks must be a list of strings.")
    if not chunks:
        raise ValueError("Cannot create embeddings for an empty chunk list.")
    if any(not isinstance(chunk, str) or not chunk.strip() for chunk in chunks):
        raise ValueError("Every chunk must contain non-empty text.")

    try:
        response = _create_client().embeddings.create(
            model=MODEL_NAME,
            input=chunks,
        )
    except ValueError:
        raise
    except RateLimitError as error:
        raise RuntimeError(
            "OpenAI rejected the request because the account has no available "
            "credits or has reached its usage limit."
        ) from error
    except AuthenticationError as error:
        raise RuntimeError(
            "OpenAI rejected the API key. Check OPENAI_API_KEY in .env."
        ) from error
    except APIConnectionError as error:
        raise RuntimeError(
            "Could not connect to the OpenAI API. Check your internet connection."
        ) from error
    except Exception as error:
        raise RuntimeError("The OpenAI embedding request failed.") from error

    ordered_data = sorted(response.data, key=lambda item: item.index)
    embeddings = [item.embedding for item in ordered_data]

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            "OpenAI returned an unexpected number of embeddings.")

    return embeddings
