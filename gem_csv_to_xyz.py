#!/usr/bin/env python3
"""
Convert GEM2 frequency-domain survey CSV exports to whitespace-column XYZ-style text
Whitespace column layout matches AEM-style .xyz files: spaced columns, '*' for missing,
and header tokens prefixed with '/' (e.g. /Line, /X) unless already present.

X, Y and the auto-detected Z / GPS altitude column are written in fixed decimal notation
(metres or degrees), never scientific notation, so UTM coordinates stay exact for GIS /
Workbench. Use --no-header if the importer treats the first line as data.

Skips leading lines whose first CSV cell starts with '#' (e.g. GPX export banners with UTMZone).

Example:
  python gem_csv_to_xyz.py ^
    "C:\\...\\023_7015_gem.csv" ^
    "C:\\...\\023_7015_gem.xyz"
"""

from __future__ import annotations

import argparse
import csv
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _row_first_significant_cell(row: Sequence[str]) -> str:
    if not row:
        return ""
    return (row[0] or "").strip().lstrip("\ufeff")


def _is_comment_row(row: Sequence[str]) -> bool:
    """GPX/GEM exporters often prefix metadata with '#' in the first field."""
    return _row_first_significant_cell(row).startswith("#")


def _read_header_row_skip_comments(reader: Iterable[List[str]]) -> List[str]:
    """Return first non-empty, non-comment row (tabular header)."""
    for row in reader:
        if not row or all(not (c or "").strip() for c in row):
            continue
        if _is_comment_row(row):
            continue
        return list(row)
    raise SystemExit("No tabular header row found (file empty or only comments/blank lines)")


def _normalize_header(name: str) -> str:
    return name.strip()


def _slash_header_label(name: str) -> str:
    """AEM-style headers often use a leading '/' (e.g. /Line)."""
    n = name.strip()
    if n.startswith("/"):
        return n
    return "/" + n


def _is_missing_token(s: str) -> bool:
    t = s.strip()
    return t == "" or t == "*"


def _format_projected_coord(raw: str) -> str:
    """
    Easting/northing/height (metres or degrees): fixed decimal only, never scientific
    notation, so UTM values stay readable and import tools see true map coordinates.
    """
    if _is_missing_token(raw):
        return "*"
    t = raw.strip().replace(",", ".")
    try:
        d = Decimal(t)
    except InvalidOperation:
        return raw.strip()
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def _format_metre_scalar(raw: str) -> str:
    """Small distances (e.g. sensor height): fixed decimal, no scientific notation."""
    return _format_projected_coord(raw)


def _format_cell(raw: str, as_int: bool) -> str:
    """Normalize numeric strings; keep '*', empty, and non-numeric text as-is."""
    if _is_missing_token(raw):
        return "*"
    t = raw.strip()
    if as_int:
        try:
            if "." in t or "e" in t.lower():
                return str(int(float(t)))
            return str(int(t))
        except ValueError:
            return t
    try:
        x = float(t)
    except ValueError:
        return t
    ax = abs(x)
    if ax == 0:
        return "0"
    # Match AEM-style mix: compact sci for large/small magnitudes, fixed decimals otherwise
    if ax >= 1e4 or (ax > 0 and ax < 1e-2):
        s = f"{x:.3E}"
        # Normalize exponent sign spacing to match example (1.234E+02)
        s = s.replace("e", "E")
        return s
    # Coordinates / common geophysics magnitudes: up to 6 dp, trim trailing zeros
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _detect_columns(
    headers: Sequence[str],
) -> Tuple[int | None, int | None, int | None, int | None]:
    """Return indices for Line, Sample, X, Y if present (case-insensitive keys)."""
    lower = [h.lower() for h in headers]

    def find_one(options: Iterable[str]) -> int | None:
        for opt in options:
            for i, h in enumerate(lower):
                if h == opt:
                    return i
        return None

    idx_line = find_one(["line"])
    idx_sample = find_one(["sample"])
    idx_x = find_one(["x"])
    idx_y = find_one(["y"])
    return idx_line, idx_sample, idx_x, idx_y


def _find_z_column(headers: Sequence[str], preferred: str | None) -> int | None:
    if preferred:
        p = preferred.strip().lower()
        for i, h in enumerate(headers):
            if h.lower() == p:
                return i
    candidates = [
        "gpsalt(m)",
        "gpsalt (m)",
        "gps_alt_m",
        "gpsalt_m",
        "alt_m",
        "altitude",
        "z",
        "elev",
        "elevation",
    ]
    lower = [h.lower() for h in headers]
    for cand in candidates:
        for i, h in enumerate(lower):
            if h == cand or cand in h.replace(" ", ""):
                return i
    return None


def _build_output_order(
    ncols: int,
    idx_line: int | None,
    idx_sample: int | None,
    idx_x: int | None,
    idx_y: int | None,
    idx_z: int | None,
) -> List[int]:
    preferred: List[int] = []
    for idx in (idx_line, idx_sample, idx_x, idx_y, idx_z):
        if idx is not None and idx not in preferred:
            preferred.append(idx)
    rest = [i for i in range(ncols) if i not in preferred]
    return preferred + rest


def _sensor_height_insert_pos(
    order: Sequence[int],
    idx_z: int | None,
    idx_y: int | None,
    idx_x: int | None,
) -> int:
    """Insert constant sensor height after Z, else after Y, else after X, else at end."""
    for idx in (idx_z, idx_y, idx_x):
        if idx is not None and idx in order:
            return order.index(idx) + 1
    return len(order)


def _scan_widths(
    data_rows: Sequence[Sequence[str]],
    order: Sequence[Optional[int]],
    out_names: Sequence[str],
    coord_csv_cols: Set[int],
    synthetic_fmt: Optional[Dict[int, str]] = None,
) -> List[int]:
    widths = [len(_slash_header_label(out_names[j])) for j in range(len(order))]
    for row in data_rows:
        if not row or all(not (c or "").strip() for c in row):
            continue
        for j, col_i in enumerate(order):
            if col_i is None:
                continue
            cell = row[col_i] if col_i < len(row) else ""
            if col_i in coord_csv_cols:
                formatted = _format_projected_coord(cell)
                widths[j] = max(widths[j], len(formatted))
            else:
                widths[j] = max(widths[j], len(cell.strip()))
    if synthetic_fmt:
        for j, s in synthetic_fmt.items():
            widths[j] = max(widths[j], len(s))
    return widths


def _write_xyz(
    path_in: Path,
    path_out: Path,
    delimiter: str,
    z_column: str | None,
    skip_bad_xy: bool,
    sensor_height: float | None,
    sensor_height_column: str,
    write_header: bool,
) -> None:
    with path_in.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header_row = _read_header_row_skip_comments(reader)
        data_rows = list(reader)

    headers = [_normalize_header(h) for h in header_row]
    ncols = len(headers)
    if ncols == 0:
        raise SystemExit("No columns in CSV header")

    idx_line, idx_sample, idx_x, idx_y = _detect_columns(headers)
    idx_z = _find_z_column(headers, z_column)
    order: List[Optional[int]] = _build_output_order(
        ncols, idx_line, idx_sample, idx_x, idx_y, idx_z
    )
    out_names = [headers[i] for i in order]

    sh_insert: int | None = None
    sh_fmt: str | None = None
    if sensor_height is not None:
        sh_insert = _sensor_height_insert_pos(order, idx_z, idx_y, idx_x)
        col_name = _normalize_header(sensor_height_column)
        order = order[:sh_insert] + [None] + order[sh_insert:]
        out_names = out_names[:sh_insert] + [col_name] + out_names[sh_insert:]
        sh_fmt = _format_metre_scalar(str(sensor_height))

    coord_csv_cols: Set[int] = {
        i
        for i in (idx_x, idx_y, idx_z)
        if i is not None
    }

    synthetic_fmt: Dict[int, str] | None = (
        {sh_insert: sh_fmt} if sh_insert is not None and sh_fmt is not None else None
    )
    widths = _scan_widths(data_rows, order, out_names, coord_csv_cols, synthetic_fmt)

    def pad_field(text: str, w: int) -> str:
        return text + " " * max(0, w - len(text))

    with path_out.open("w", encoding="utf-8", newline="") as fout:
        if write_header:
            fout.write(
                " ".join(
                    pad_field(_slash_header_label(out_names[j]), widths[j])
                    for j in range(len(order))
                )
                + "\n"
            )

        coord_output_cols: Set[int] = {
            j
            for j, col_i in enumerate(order)
            if col_i is not None and col_i in coord_csv_cols
        }

        int_cols = set()
        if idx_line is not None:
            try:
                int_cols.add(order.index(idx_line))
            except ValueError:
                pass
        if idx_sample is not None:
            try:
                int_cols.add(order.index(idx_sample))
            except ValueError:
                pass

        n_kept = 0
        n_skip = 0
        for row in data_rows:
            if not row or all(not (c or "").strip() for c in row):
                continue
            if skip_bad_xy and idx_x is not None and idx_y is not None:
                if idx_x >= len(row) or idx_y >= len(row):
                    n_skip += 1
                    continue
                if _is_missing_token(row[idx_x]) or _is_missing_token(row[idx_y]):
                    n_skip += 1
                    continue

            pieces: List[str] = []
            for j, col_i in enumerate(order):
                if col_i is None:
                    raw = str(sensor_height) if sensor_height is not None else ""
                    pieces.append(pad_field(_format_metre_scalar(raw), widths[j]))
                    continue
                raw = row[col_i] if col_i < len(row) else ""
                if j in int_cols:
                    pieces.append(pad_field(_format_cell(raw, as_int=True), widths[j]))
                elif j in coord_output_cols:
                    pieces.append(
                        pad_field(_format_projected_coord(raw), widths[j])
                    )
                else:
                    pieces.append(pad_field(_format_cell(raw, as_int=False), widths[j]))
            fout.write(" ".join(pieces) + "\n")
            n_kept += 1

    print(f"Wrote {n_kept} rows to {path_out}", file=sys.stderr)
    if skip_bad_xy:
        print(f"Skipped {n_skip} rows with missing X/Y", file=sys.stderr)


def convert_gem_csv_to_xyz(
    path_in: Path,
    path_out: Path | None = None,
    *,
    delimiter: str = ",",
    z_column: str | None = None,
    skip_bad_xy: bool = True,
    sensor_height: float | None = None,
    sensor_height_column: str = "SensorHeight(m)",
    write_header: bool = True,
) -> Path:
    """
    Convert a GEM-2 exporter CSV (with optional leading '#' comment lines) to
    Hedwige-style whitespace XYZ for `gem2_simpeg_invert.read_gem2_xyz`.

    Returns the path to the written ``.xyz`` file (``path_out`` or
    ``path_in.with_suffix('.xyz')``).
    """
    if path_out is None:
        path_out = path_in.with_suffix(".xyz")
    _write_xyz(
        path_in,
        path_out,
        delimiter=delimiter,
        z_column=z_column,
        skip_bad_xy=skip_bad_xy,
        sensor_height=sensor_height,
        sensor_height_column=sensor_height_column,
        write_header=write_header,
    )
    return path_out


def main() -> None:
    p = argparse.ArgumentParser(description="Convert GEM2 CSV to whitespace-column XYZ text.")
    p.add_argument("input_csv", type=Path, help="Path to GEM2 *_gem.csv export")
    p.add_argument(
        "output_xyz",
        type=Path,
        nargs="?",
        help="Output .xyz path (default: input basename + .xyz)",
    )
    p.add_argument(
        "--delimiter",
        default=",",
        help="CSV delimiter (default: comma)",
    )
    p.add_argument(
        "--z-column",
        default=None,
        help=(
            "Header name for Z/elevation column (default: auto-detect gpsalt/GPSalt/altitude/z)"
        ),
    )
    p.add_argument(
        "--keep-bad-xy",
        action="store_true",
        help="Keep rows where X or Y is '*' (default: skip those rows)",
    )
    p.add_argument(
        "--sensor-height",
        type=float,
        default=None,
        metavar="M",
        help="Add a constant sensor height column (m) after altitude/Y/X",
    )
    p.add_argument(
        "--sensor-height-column",
        default="SensorHeight(m)",
        help="Header name for --sensor-height (default: SensorHeight(m))",
    )
    p.add_argument(
        "--no-header",
        action="store_true",
        help="Omit column header line (e.g. Aarhus Workbench imports first row as data)",
    )
    args = p.parse_args()
    path_in: Path = args.input_csv
    if not path_in.is_file():
        raise SystemExit(f"Input not found: {path_in}")
    path_out = args.output_xyz
    if path_out is None:
        path_out = path_in.with_suffix(".xyz")
    convert_gem_csv_to_xyz(
        path_in,
        path_out,
        delimiter=args.delimiter,
        z_column=args.z_column,
        skip_bad_xy=not args.keep_bad_xy,
        sensor_height=args.sensor_height,
        sensor_height_column=args.sensor_height_column,
        write_header=not args.no_header,
    )


if __name__ == "__main__":
    main()
