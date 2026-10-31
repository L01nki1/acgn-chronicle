"""FastAPI host + static frontend for the WP6 query page.

Owner: C | Spec: contracts/KnowledgePort.md §4, docs/ROADMAP.md G1a-G2

G1a serves a fixed mock backend (`mock_port.MockKnowledgePort`) behind the contract routes, so the
page, its two clipping layers and its empty-vs-refused states can be built and checked before WP2/
WP4 land. The real host replaces the backend at W2; routes and static files stay unchanged.

Run locally: `uv run python -m acgn_rag.web` then open http://127.0.0.1:8000
"""

__all__: list[str] = []
