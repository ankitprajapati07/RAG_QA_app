from typing import List
import os
from dotenv import load_dotenv
from .embeddings import get_openai_client

load_dotenv()
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini")

_SYSTEM_PROMPT = (
	"You are a medical Q&A assistant. Answer the user's question using the provided context snippets. "
	"Synthesize a clear, concise answer; do not copy verbatim. If the context is insufficient, say you are unsure."
)


def _format_context(snippets: List[str]) -> str:
	parts = []
	for i, s in enumerate(snippets, 1):
		parts.append(f"[Snippet {i}]\n{s}")
	return "\n\n".join(parts)


def _fallback_answer(question: str, context_snippets: List[str]) -> str:
	if not context_snippets:
		return "I couldn't find enough context to answer confidently."
	joined = " \n".join(context_snippets)
	# Trim a long fallback to keep response short
	fallback = joined.strip()
	if len(fallback) > 500:
		fallback = fallback[:500].rsplit(" ", 1)[0] + "..."
	return f"Based on the retrieved context, the likely answer is: {fallback}"


def generate_answer(question: str, context_snippets: List[str]) -> str:
	client = get_openai_client()
	messages = [
		{"role": "system", "content": _SYSTEM_PROMPT},
		{"role": "user", "content": f"Question: {question}\n\nContext:\n{_format_context(context_snippets)}"},
	]
	try:
		resp = client.chat.completions.create(
			model=OPENAI_CHAT_MODEL,
			messages=messages,
			max_completion_tokens=256,
		)
		content = resp.choices[0].message.content if resp and resp.choices else ""
		content = (content or "").strip()
		return content if content else _fallback_answer(question, context_snippets)
	except Exception:
		# Fall back to a simple synthesized answer from context
		return _fallback_answer(question, context_snippets)
