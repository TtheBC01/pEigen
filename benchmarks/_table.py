"""Shared tabulate helpers for benchmark output."""

from __future__ import annotations

from tabulate import tabulate

TABLEFMT = "fancy_grid"

CG_HEADERS = (
    "op",
    "size",
    "scipy_p50(ms)",
    "peigen_p50(ms)",
    "speedup",
    "iters (s/p)",
    "resid (s/p)",
)


def _format_ms(value: float) -> str:
    if value != value:
        return "fail"
    return f"{value:.3f}"


def _format_speedup(ref_ms: float, peigen_ms: float, *, comparable: bool = True) -> str:
    if not comparable or peigen_ms != peigen_ms or peigen_ms <= 0:
        return "n/a"
    return f"{ref_ms / peigen_ms:.2f}x"


def compare_row(op: str, size: str, ref_ms: float, peigen_ms: float) -> list[str]:
    return [
        op,
        size,
        _format_ms(ref_ms),
        _format_ms(peigen_ms),
        _format_speedup(ref_ms, peigen_ms),
    ]


def cg_row(
    op: str,
    size: str,
    scipy_ms: float,
    peigen_ms: float,
    *,
    scipy_iters: int,
    peigen_iters: int,
    scipy_resid: float,
    peigen_resid: float,
    scipy_ok: bool,
    peigen_ok: bool,
) -> list[str]:
    comparable = scipy_ok and peigen_ok
    scipy_resid_str = f"{scipy_resid:.1e}" if scipy_ok else "fail"
    peigen_resid_str = f"{peigen_resid:.1e}" if peigen_ok else "fail"
    return [
        op,
        size,
        _format_ms(scipy_ms),
        _format_ms(peigen_ms),
        _format_speedup(scipy_ms, peigen_ms, comparable=comparable),
        f"{scipy_iters}/{peigen_iters}",
        f"{scipy_resid_str}/{peigen_resid_str}",
    ]


def print_compare_section(
    title: str,
    rows: list[list[str]],
    *,
    ref_label: str = "ref",
    notes: str | None = None,
) -> None:
    headers = (
        "op",
        "size",
        f"{ref_label}_p50(ms)",
        "peigen_p50(ms)",
        "speedup",
    )
    _print_section(title, rows, headers, notes=notes)


def print_cg_section(title: str, rows: list[list[str]], *, notes: str | None = None) -> None:
    _print_section(title, rows, CG_HEADERS, notes=notes)


def _print_section(
    title: str,
    rows: list[list[str]],
    headers: tuple[str, ...],
    *,
    notes: str | None = None,
) -> None:
    print()
    print(title)
    if notes:
        print(notes)
    print(tabulate(rows, headers=headers, tablefmt=TABLEFMT))
