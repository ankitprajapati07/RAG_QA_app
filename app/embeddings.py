from typing import List
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

_client = None


def get_openai_client() -> OpenAI:
	global _client
	if _client is None:
		_client = OpenAI(api_key=OPENAI_API_KEY)
	return _client


def embed_texts(texts: List[str], batch_size: int = 128) -> np.ndarray:
	client = get_openai_client()
	embeddings: List[List[float]] = []
	for i in range(0, len(texts), batch_size):
		chunk = texts[i:i + batch_size]
		resp = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=chunk)
		embeddings.extend([e.embedding for e in resp.data])
	return np.array(embeddings, dtype=np.float32)


def embed_text(text: str) -> np.ndarray:
	return embed_texts([text], batch_size=1)[0]
