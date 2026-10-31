"""KnowledgePort HTTP host for the G1a query page.

Owner: C | Spec: contracts/KnowledgePort.md §4, docs/ROADMAP.md G1a

Routes are exactly the contract reads: `GET /snapshots`, `GET /snapshot`, `POST /retrieve`,
`POST /ask`; `GET /daily` answers 501 until the conditional G3 stage. G1a serves a mock backend so
the page can be built before retrieval/generation land. The page never imports a backend directly:
it talks to these routes (or an in-process implementation of the same port).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from acgn_rag.core.ports import (
    ChunkExcerpt,
    CorpusSnapshot,
    GroundedAnswer,
    RetrievalQuery,
    SnapshotInventory,
)
from acgn_rag.web.backend import KnowledgePortBackend
from acgn_rag.web.errors import PortError
from acgn_rag.web.mock_port import MockKnowledgePort

STATIC_DIR = Path(__file__).with_name("static")
_API_PATHS = frozenset({"/snapshots", "/snapshot", "/retrieve", "/ask", "/daily"})


def create_app(backend: KnowledgePortBackend | None = None) -> FastAPI:
    """Build the host around a backend; G1a defaults to the fixed mock."""
    port = MockKnowledgePort() if backend is None else backend
    app = FastAPI(
        title="ACGN Chronicle KnowledgePort",
        version="0.1.0",
        description="G1a mock host for the WP6 query page. Responses follow the wire contract.",
    )

    @app.exception_handler(PortError)
    async def _port_error(_request: Request, exc: PortError) -> JSONResponse:
        return JSONResponse(
            {"detail": exc.detail},
            status_code=exc.status_code,
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _malformed_body(_request: Request, _exc: RequestValidationError) -> JSONResponse:
        # KnowledgePort §5: a malformed request body is 400, while an invalid filter is 422.
        return JSONResponse({"detail": "Malformed request body."}, status_code=400)

    @app.middleware("http")
    async def _no_store_for_reads(request: Request, call_next):
        response = await call_next(request)
        if request.url.path in _API_PATHS:
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/snapshots", response_model=SnapshotInventory)
    def get_snapshots() -> SnapshotInventory:
        return port.inventory()

    @app.get("/snapshot", response_model=CorpusSnapshot)
    def get_snapshot(
        snapshot_id: Annotated[str | None, Query(alias="id")] = None,
    ) -> CorpusSnapshot:
        return port.snapshot(snapshot_id)

    @app.post("/retrieve", response_model=list[ChunkExcerpt])
    def retrieve(query: RetrievalQuery) -> list[ChunkExcerpt]:
        return list(port.retrieve(query))

    @app.post("/ask", response_model=GroundedAnswer)
    def ask(query: RetrievalQuery) -> GroundedAnswer:
        return port.ask(query)

    @app.get("/daily")
    def daily(snapshot_id: str, date: str) -> JSONResponse:
        return JSONResponse(
            {
                "detail": (
                    "GET /daily is the conditional G3 stage and is not served in G1a "
                    f"(snapshot_id={snapshot_id}, date={date})."
                )
            },
            status_code=501,
        )

    return app


app = create_app()

__all__ = ["STATIC_DIR", "app", "create_app"]
