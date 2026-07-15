"""
query_expander.py
------------------
Job: take one student query and rewrite it a few different ways, so
semantic search catches segments that use different wording for the
same concept (e.g. "pointers" vs "memory addresses" vs "referencing variables").

Uses a small, fast Groq model (llama-3.1-8b-instant) since this is a
cheap rewriting task, not something that needs the big model.
"""

from groq import Groq

from config import GROQ_API_KEY

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def expand_query(student_prompt: str, n_variants: int = 4) -> list[str]:
    """
    Returns a list of query variants, always including the original
    prompt last so it's guaranteed to be part of the search even if
    the LLM call fails or returns something unparseable.
    """
    prompt = f"""A student is searching for a specific part of a lecture video.
Their question is: "{student_prompt}"

Rewrite this question {n_variants} different ways that ask for the same thing,
using different vocabulary and phrasing each time (e.g. synonyms, more
formal/informal phrasing, different sentence structure).

Return ONLY a numbered list, one variant per line. No explanations.
"""

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Query expansion failed, falling back to original query only: {e}")
        return [student_prompt]

    expanded = []
    for line in raw.split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            cleaned = line.split(".", 1)[-1].strip()
            if cleaned:
                expanded.append(cleaned)

    if not expanded:
        expanded = [student_prompt]
    elif student_prompt not in expanded:
        expanded.append(student_prompt)

    return expanded