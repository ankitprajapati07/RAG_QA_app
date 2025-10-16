from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import faiss
import numpy as np

from .embeddings import embed_texts, embed_text


@dataclass
class Document:
	id: str
	question: str
	description: str


class FaissRetriever:
	def __init__(self, index_dir: str = "./index"):
		self.index_dir = index_dir
		self.index_path = os.path.join(index_dir, "faiss.index")
		self.meta_path = os.path.join(index_dir, "meta.json")
		self.index: faiss.Index | None = None
		self.meta: List[Document] = []

	def build(self, docs: List[Document], batch_size: int = 128) -> None:
		texts = [f"{d.question}\n\n{d.description}" for d in docs]
		vectors = embed_texts(texts, batch_size=batch_size)
		dim = vectors.shape[1]
		index = faiss.IndexFlatIP(dim)
		faiss.normalize_L2(vectors)
		index.add(vectors)
		self.index = index
		self.meta = docs
		os.makedirs(self.index_dir, exist_ok=True)
		faiss.write_index(self.index, self.index_path)
		with open(self.meta_path, "w", encoding="utf-8") as f:
			json.dump([doc.__dict__ for doc in docs], f, ensure_ascii=False)

	def load(self) -> None:
		if not os.path.exists(self.index_path):
			raise FileNotFoundError(f"FAISS index not found at {self.index_path}")
		if not os.path.exists(self.meta_path):
			raise FileNotFoundError(f"Metadata not found at {self.meta_path}")
		self.index = faiss.read_index(self.index_path)
		with open(self.meta_path, "r", encoding="utf-8") as f:
			raw = json.load(f)
		self.meta = [Document(**item) for item in raw]

	def query(self, question: str, top_k: int = 10) -> List[Tuple[Document, float]]:
		if self.index is None:
			self.load()
		q_vec = embed_text(question).astype(np.float32).reshape(1, -1)
		faiss.normalize_L2(q_vec)
		scores, indices = self.index.search(q_vec, top_k)
		idxs = indices[0]
		scrs = scores[0]
		results: List[Tuple[Document, float]] = []
		for i, s in zip(idxs, scrs):
			if i == -1:
				continue
			results.append((self.meta[int(i)], float(s)))
		return results

	def top1(self, question: str) -> Tuple[Document, float]:
		results = self.query(question, top_k=1)
		if not results:
			raise ValueError("No results from retriever")
		return results[0]
