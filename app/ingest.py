import argparse
import json
from typing import List, Union

from .retriever import Document, FaissRetriever


def _coerce_item(item: Union[dict, str]) -> Union[dict, None]:
	# If item is a string, try to parse JSON; otherwise skip
	if isinstance(item, str):
		try:
			parsed = json.loads(item)
			if isinstance(parsed, dict):
				return parsed
		except Exception:
			return None
	elif isinstance(item, dict):
		return item
	return None


def load_dataset(path: str) -> List[Document]:
	# Try to load as a JSON array first
	data: List[Union[dict, str]] = []
	try:
		with open(path, "r", encoding="utf-8") as f:
			content = f.read()
			obj = json.loads(content)
			if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
				data = obj["data"]
			elif isinstance(obj, list):
				data = obj
			else:
				data = []
	except json.JSONDecodeError:
		# Fallback to JSONL (one JSON per line)
		with open(path, "r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				try:
					obj = json.loads(line)
					data.append(obj)
				except Exception:
					# keep raw line for string-parse attempt below
					data.append(line)

	docs: List[Document] = []
	for raw in data:
		item = _coerce_item(raw)
		if not item:
			continue
		_id = str(item.get("id", item.get("_id", "")))
		q = str(item.get("question", "")).strip()
		d = str(item.get("description", "")).strip()
		if not _id or not q:
			continue
		docs.append(Document(id=_id, question=q, description=d))
	return docs


def main():
	parser = argparse.ArgumentParser(description="Build FAISS index from dataset")
	parser.add_argument("--dataset", required=True, help="Path to dataset JSON/JSONL")
	parser.add_argument("--out_dir", required=True, help="Output directory for index")
	parser.add_argument("--batch_size", type=int, default=128, help="Embedding batch size")
	args = parser.parse_args()

	docs = load_dataset(args.dataset)
	if not docs:
		raise SystemExit("No valid records parsed from dataset. Ensure fields id, question, description exist.")
	retriever = FaissRetriever(index_dir=args.out_dir)
	retriever.build(docs, batch_size=args.batch_size)
	print(f"Indexed {len(docs)} documents → {args.out_dir}")


if __name__ == "__main__":
	main()
