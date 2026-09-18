# agent/prompts.py

GRADE_EVIDENCE_PROMPT = """
Determine whether the documentation contains enough evidence to
answer the user's question.

Do not answer the question.
Only grade the available evidence.
Do not use outside knowledge.

Original question:
{question}

Current search query:
{search_query}

Documentation:
{context}
"""


REWRITE_QUERY_PROMPT = """
Rewrite the search query so it is more likely to retrieve relevant
Singed documentation.

Return one concise search query.
Do not answer the question.
Preserve the technical meaning of the original question.

Original question:
{question}

Previous search query:
{search_query}

Reason the evidence was insufficient:
{evidence_reason}

Previously retrieved text:
{context_summary}
"""


ANSWER_PROMPT = """
You answer questions using only the supplied Singed documentation.

Rules:
- Answer the original question.
- Cite supported claims using [S1], [S2], and similar labels.
- Do not use outside knowledge.
- Do not invent information.
- If the evidence is unexpectedly insufficient, say so.

Documentation:
{context}

Original question:
{question}
"""