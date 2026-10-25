"""
Server Log Maintenance

Startup log hygiene. A vanilla server gzips its previous runs' logs on
startup, so `logs/` holds one uncompressed `.log` for the run in progress
and `.log.gz` archives for everything before it. Crotalus writes one
timestamped `.log` per run, so there is no single file to exempt -- the
active one is passed in by name and everything else is stale.
"""

import gzip
import os
import shutil
from pathlib import Path
from typing import Optional

from loguru import logger


def find_compressible_logs(logs_dir: Path, current_name: Optional[str] = None) -> list[Path]:
    """Select the files in `logs_dir` that this startup should gzip.

    Only `*.log` counts. Widening the pattern to catch `debug.log` or
    `server.log.1` style leftovers would be more tidying than it is worth:
    those names belong to other tools' conventions, and archiving one on
    their behalf is likelier to surprise an operator than to help.

    Args:
        logs_dir: The `logs/` directory to scan.
        current_name: File name of the log this run is writing right now.
            It must never be compressed, or the running server loses its
            log. `None` means the caller is not writing into `logs_dir`.

    Returns:
        Stale logs, sorted by name so compression order is reproducible.
    """
    return sorted(
        p for p in logs_dir.glob("*.log")
        if p.is_file() and p.name != current_name
    )


def _archive_path(source: Path) -> Path:
    """First unused `name.log.gz`, or `name-1.log.gz`, `name-2.log.gz`, ...

    An existing archive is never clobbered -- a half-written one left by an
    interrupted run, or an older archive the operator kept on purpose, both
    outrank the log we are about to archive.
    """
    candidate = source.with_name(f"{source.name}.gz")
    if not candidate.exists():
        return candidate

    num = 1
    while candidate.exists():
        candidate = source.with_name(f"{source.stem}-{num}{source.suffix}.gz")
        num += 1
    return candidate


def _compress_file(source: Path, target: Path) -> None:
    """Gzip `source` into `target`, then drop the original.

    The archive is built under a `.tmp` name and moved into place, so a
    crash mid-compress never leaves a truncated `.gz` as the only copy of
    a log. The `.tmp` name cannot itself be picked up by the `*.log` scan.
    """
    staging = target.with_name(f"{target.name}.tmp")

    try:
        with open(source, "rb") as fin, gzip.open(staging, "wb") as fout:
            shutil.copyfileobj(fin, fout, length=1 << 20)
        os.replace(staging, target)
    except OSError:
        staging.unlink(missing_ok=True)
        raise

    os.remove(source)


def compress_old_logs(logs_dir: Path, current_name: Optional[str] = None) -> int:
    """Gzip every stale log in `logs_dir`. Returns how many were archived.

    One unreadable or locked file must not stop the rest, so failures are
    reported per file and the scan carries on.
    """
    if not logs_dir.is_dir():
        return 0

    stale = find_compressible_logs(logs_dir, current_name)
    if not stale:
        return 0

    logger.info(f"Found {len(stale)} old logs to compress.")

    compressed = 0
    for source in stale:
        target = _archive_path(source)
        logger.debug(f"Compressing {source.name} to {target.name}.")
        try:
            _compress_file(source, target)
        except OSError as exc:
            logger.error(f"Couldn't compress {source.name}: {exc}")
            continue
        compressed += 1

    if compressed:
        plural = "" if compressed == 1 else "s"
        logger.info(f"Compressed {compressed} log file{plural}.")
    return compressed

