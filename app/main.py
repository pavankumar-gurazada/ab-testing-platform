"""FastAPI application entrypoint.

"Features as services": each domain is a router (flags, assign, metrics,
experiments, analysis, simulator) — logical service boundaries in one process.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from . import db, seed

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def sim_clock() -> dict:
    """Current simulated-clock state, rendered in the navbar on every page."""
    row = db.query_one("SELECT * FROM sim_state WHERE id = 1")
    return dict(row) if row else {"current_sim_day": 0, "population_size": 0, "anomaly_mode": "none"}


def render(request: Request, template: str, **ctx) -> HTMLResponse:
    ctx.update(request=request, clock=sim_clock())
    return templates.TemplateResponse(request, template, ctx)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    seed.seed_if_empty()
    yield


app = FastAPI(title="A/B Testing Platform", lifespan=lifespan)

from .routers import analysis, assign, experiments, flags, metrics, simulator  # noqa: E402

app.include_router(flags.router)
app.include_router(assign.router)
app.include_router(metrics.router)
app.include_router(simulator.router)
app.include_router(experiments.router)
app.include_router(analysis.router)


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    flag_rows = db.query(
        "SELECT f.*, a.name AS app_name, "
        "  (SELECT COUNT(*) FROM assignments WHERE flag_id = f.id) AS n_assigned "
        "FROM flags f JOIN applications a ON a.id = f.application_id ORDER BY f.id")
    experiments = db.query(
        "SELECT e.*, f.name AS flag_name FROM experiments e "
        "JOIN flags f ON f.id = e.flag_id ORDER BY e.id DESC")
    return render(request, "dashboard.html", flags=flag_rows, experiments=experiments)
