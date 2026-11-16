"""Controlled sequential preparation of persisted Kazakhstan screening cells.

This is deliberately a local operator tool, not a public API endpoint. Start
with --dry-run; a pilot requires an explicit --limit 3.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path
from typing import Sequence

from fastapi import HTTPException

# Allow `python scripts/prepare_national_grid.py` from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.routes_monitoring import _perform_analysis
from app.services import monitoring_service, national_grid_service
from app.services.kazakhstan_grid import DEFAULT_GRID_VERSION, configured_grid_cells, grid_plan_summary

RATE_LIMIT_SECONDS = 1.5
PROVENANCE_VERSION = "national-grid-preparation-v1"


def _bool(value: str) -> bool:
    normalized = value.lower()
    if normalized not in {"true", "false"}:
        raise argparse.ArgumentTypeError("must be true or false")
    return normalized == "true"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare persisted Kazakhstan national screening cells sequentially.")
    parser.add_argument("--year", type=int, required=True)
    period = parser.add_mutually_exclusive_group(required=True)
    period.add_argument("--month", type=int, choices=range(1, 13))
    period.add_argument("--all-months", action="store_true", help="Prepare all 12 months sequentially.")
    parser.add_argument("--grid-version", default=DEFAULT_GRID_VERSION)
    parser.add_argument("--limit", type=int, default=None, help="Maximum selected cells; use 3 for the initial pilot.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without DB writes or external calls.")
    parser.add_argument("--resume", action="store_true", help="Skip cells that already have a ready observation.")
    parser.add_argument("--cell-id", action="append", default=[], help="Repeat to select individual cells.")
    parser.add_argument("--enrich", type=_bool, default=False, metavar="true|false")
    parser.add_argument("--force", action="store_true", help="Explicitly replace a ready observation; use only after review.")
    return parser


def planned_cells(grid_version: str, cell_ids: Sequence[str], limit: int | None) -> list[dict]:
    cells = configured_grid_cells(grid_version)
    if cell_ids:
        requested = set(cell_ids)
        cells = [cell for cell in cells if cell["cell_id"] in requested]
        missing = requested - {cell["cell_id"] for cell in cells}
        if missing:
            raise ValueError(f"Unknown or out-of-bound cell IDs: {', '.join(sorted(missing))}")
    return cells[:limit] if limit is not None else cells


def _progress_line(
    completed: int,
    total: int,
    ready: int,
    skipped: int,
    missing: int,
    failed: int,
    current: str,
) -> str:
    width = 24
    percent = completed / total if total else 1.0
    filled = round(width * percent)
    bar = "#" * filled + "-" * (width - filled)
    return (
        f"[{bar}] {completed}/{total} ({percent:.1%}) "
        f"ready={ready} skipped={skipped} missing={missing} failed={failed} | {current}"
    )


def _show_progress(line: str, finished: bool = False) -> None:
    if sys.stdout.isatty():
        sys.stdout.write(f"\r{line}{' ' * 8}")
        if finished:
            sys.stdout.write("\n")
        sys.stdout.flush()
    else:
        print(line, flush=True)


def run(args: argparse.Namespace) -> int:
    if args.year < 2017:
        raise ValueError("year must be 2017 or later")
    all_months = getattr(args, "all_months", False)
    months = list(range(1, 13)) if all_months else [args.month]
    cells = planned_cells(args.grid_version, args.cell_id, args.limit)
    total = len(cells) * len(months)
    summary = grid_plan_summary(args.grid_version)
    print(f"Grid {args.grid_version}: {summary['inside_kazakhstan_count']} cells inside Kazakhstan from {summary['candidate_count']} bounding-box candidates.")
    period_label = f"all months of {args.year}" if all_months else f"{args.year:04d}-{args.month:02d}"
    print(f"Planned for {period_label}: {total} cell-months; sequential rate limit {RATE_LIMIT_SECONDS:.1f}s between requests.")
    for cell in cells:
        print(f"  {cell['cell_id']}: {cell['latitude']:.3f}, {cell['longitude']:.3f}")
    if args.dry_run:
        print("DRY RUN: zero database writes, zero Earth Engine calls, and zero NASA POWER calls were made.")
        return 0
    if args.force:
        print("WARNING: --force may replace ready observations for the selected cells only.")

    persisted = {cell["cell_id"]: cell for cell in national_grid_service.ensure_grid_cells(args.grid_version)}
    created = skipped = missing = failed = 0
    completed = 0
    for month in months:
        for index, plan in enumerate(cells):
            cell = persisted[plan["cell_id"]]
            existing = national_grid_service.grid_results(args.year, month, args.grid_version, has_observation=True)
            existing_ids = {item["cell_id"] for item in existing if item["status"] == "ready"}
            current = f"{args.year:04d}-{month:02d} {cell['cell_id']}"
            if args.resume and cell["cell_id"] in existing_ids and not args.force:
                skipped += 1
                completed += 1
                _show_progress(_progress_line(completed, total, created, skipped, missing, failed, f"{current} SKIP"))
                continue
            _show_progress(_progress_line(completed, total, created, skipped, missing, failed, f"{current} processing"))
            try:
                location = monitoring_service.get_location(cell["location_id"])
                _perform_analysis(
                    location, date(args.year, month, 1), args.enrich,
                    {"source_type": "national_grid", "grid_version": args.grid_version, "config_version": PROVENANCE_VERSION},
                    month, force=args.force,
                )
                national_grid_service.record_status(args.grid_version, cell["cell_id"], args.year, month, "ready", provenance={"config_version": PROVENANCE_VERSION})
                created += 1
                outcome = "READY"
            except HTTPException as error:
                outcome = "missing" if error.status_code == 404 else "failed"
                national_grid_service.record_status(args.grid_version, cell["cell_id"], args.year, month, outcome, str(error.detail), {"config_version": PROVENANCE_VERSION})
                missing += outcome == "missing"
                failed += outcome == "failed"
                print(f"{outcome.upper()} {current}: {error.detail}")
            except Exception as error:  # keep a single failed cell from aborting the batch
                outcome = "failed"
                national_grid_service.record_status(args.grid_version, cell["cell_id"], args.year, month, outcome, str(error), {"config_version": PROVENANCE_VERSION})
                failed += 1
                print(f"FAILED {current}: {error}")
            completed += 1
            _show_progress(_progress_line(completed, total, created, skipped, missing, failed, f"{current} {outcome.upper()}"))
            if completed < total:
                time.sleep(RATE_LIMIT_SECONDS)
    _show_progress(_progress_line(completed, total, created, skipped, missing, failed, "complete"), finished=True)
    print(f"Summary: ready={created}, skipped={skipped}, missing={missing}, failed={failed}.")
    return 0 if failed == 0 else 1


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
