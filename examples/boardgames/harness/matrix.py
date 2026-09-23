"""§C.20 CROSS-GAME REGRESSION MATRIX — rows = a change, columns = games, cells = a paired score delta.

WHY: a single-game A/B is confound-prone and easy to overfit a conclusion to (the whole §C.15-§C.19 struggle).
The matrix turns it into a GENERALISATION test — does a lever move EVERY game (a real transferable lever), just
ONE (game-specific or noise), or some one way and others the other (contested)? The measurement per cell already
exists (`paired_exploitability` where no solver, `paired_conversion` where one does); this is the analysis layer
that reads N per-game paired deltas and says what the CHANGE actually did.

The one discipline this adds over stacking N single-game reads: multiplicity is counted ACROSS the row. Testing a
change on N games is N comparisons, so a cell clears the bar only at alpha/N (Bonferroni) — the same correction
`measure_ab` applies to comparisons within one game, applied across games. A row's verdict is DERIVED from its
cells, never asserted, in the vocabulary the hypothesis register already uses (§C.30)."""
from __future__ import annotations


def corrected_alpha(n_games: int, alpha: float = 0.05) -> float:
    """Bonferroni across the row: N games is N comparisons on the change."""
    return alpha / max(1, n_games)


def _sign(x: float) -> str:
    return "+" if x > 0 else "-" if x < 0 else "0"


def _within_null_band(c: dict) -> bool:
    """Is the effect PROVABLY negligible — the whole CI inside the null band (a two-one-sided-test equivalence),
    not merely a small point estimate? Reading `null` off the point estimate alone is the absence-of-evidence
    fallacy the rest of the codebase is built to avoid (measurement.py gates on the interval, not the point).
    Falls back to the point estimate only when a CI is unavailable."""
    nb = c.get("null_below", 0.03)
    ci = c.get("ci")
    if ci is None:
        return abs(c["diff"]) < nb
    return ci[0] > -nb and ci[1] < nb


def row_verdict(cells: list[dict], alpha: float = 0.05) -> dict:
    """Classify what a change did across its games. Two DIFFERENT questions with two DIFFERENT bars:

      * "did ANY game move" is a union — N games is N comparisons, so it is tested at the Bonferroni bar
        alpha/N (this drives `local`, `contested`, and the significance that keeps a row out of null).
      * "did EVERY game move the same way" (`generalises`) is a CONJUNCTION — an intersection-union test, whose
        level is alpha with NO correction (Berger's min-test theorem, the same reason TOST uses alpha not
        alpha/2). Correcting the conjunction by alpha/N would systematically miss the very thing the matrix
        exists to find.

    Statuses: generalises (every game moves at alpha, one direction) · local (some — but not all — move at the
    corrected bar, consistent direction) · contested (games move at the corrected bar in OPPOSITE directions) ·
    null (every cell's CI lies inside its null band — provably negligible everywhere) · inconclusive (nothing
    clears the bar and at least one cell is not provably negligible — underpowered) · single_game (N==1, which
    cannot evidence generalisation) · untested (no cells)."""
    n = len(cells)
    if n == 0:
        return {"status": "untested", "direction": "0", "n_games": 0, "n_significant": 0,
                "cells": [], "caveated_games": []}
    acorr = corrected_alpha(n, alpha)
    graded = []
    for c in cells:
        union_sig = c["p"] <= acorr          # cleared the strict UNION bar (drives local/contested)
        iut_sig = c["p"] <= alpha            # cleared the per-cell CONJUNCTION bar (drives generalises)
        graded.append({**c, "significant": union_sig, "iut_significant": iut_sig,
                       "sign": _sign(c["diff"]) if union_sig else "0", "alpha_corrected": acorr})
    union_cells = [c for c in graded if c["significant"]]
    caveated = [c["game"] for c in graded if c.get("caveats")]
    union_signs = {c["sign"] for c in union_cells}
    if n == 1:
        # one game cannot evidence GENERALISATION — that is just a single-game A/B, the thing the matrix abolishes
        c = graded[0]
        status = ("effect" if c["significant"] else "null" if _within_null_band(c) else "inconclusive")
        direction = _sign(c["diff"]) if c["significant"] else "0"
        return {"status": "single_game", "sub_status": status, "direction": direction, "n_games": 1,
                "n_significant": len(union_cells), "cells": graded, "caveated_games": caveated}
    if len(union_signs) > 1:                 # opposite-direction effects at the strict bar
        status, direction = "contested", "0"
    elif all(c["iut_significant"] for c in graded) and len({_sign(c["diff"]) for c in graded}) == 1:
        status, direction = "generalises", _sign(graded[0]["diff"])   # every game moves (IUT at alpha)
    elif union_cells:                        # some game moves at the strict bar, one direction (>1 handled above)
        status, direction = "local", next(iter(union_signs))
    elif all(_within_null_band(c) for c in graded):
        status, direction = "null", "0"
    else:
        status, direction = "inconclusive", "0"
    return {"status": status, "direction": direction, "n_games": n, "n_significant": len(union_cells),
            "cells": graded, "caveated_games": caveated}


def build_matrix(rows: dict, alpha: float = 0.05) -> dict:
    """Assemble labelled rows (change -> list of per-game cells) into a matrix with a stable, sorted column set
    (the union of games across rows) and a derived verdict per row."""
    games = sorted({c["game"] for cells in rows.values() for c in cells})
    out_rows = {}
    for name, cells in rows.items():
        v = row_verdict(cells, alpha)
        v["by_game"] = {c["game"]: c for c in v["cells"]}
        out_rows[name] = v
    return {"games": games, "rows": out_rows, "alpha": alpha}


_MARK = {"generalises": "GENERALISES", "local": "local", "contested": "CONTESTED",
         "null": "null", "inconclusive": "inconcl.", "untested": "untested"}


def render_matrix(matrix: dict) -> str:
    """A text table: one row per change, one column per game, each cell the signed delta with a significance
    mark, then the row's verdict. A missing cell (a game not measured for that change) renders as '·'."""
    games = matrix["games"]
    w = max([len(g) for g in games] + [7])
    header = "change".ljust(22) + " | " + " | ".join(g.center(w) for g in games) + " | verdict"
    lines = [header, "-" * len(header)]
    for name, v in matrix["rows"].items():
        cells = []
        for g in games:
            c = v["by_game"].get(g)
            if c is None:
                cells.append("·".center(w))
            else:
                mark = "*" if c["significant"] else " "
                cells.append(f"{c['diff']:+.2f}{mark}".center(w))
        verdict = _MARK[v["status"]] + (v["direction"] if v["direction"] in "+-" else "")
        if v["caveated_games"]:
            verdict += f" ⚠{','.join(v['caveated_games'])}"
        lines.append(name.ljust(22) + " | " + " | ".join(cells) + " | " + verdict)
    return "\n".join(lines)


def cell_from_compare(res: dict, game: str, n: int, null_below: float = 0.03) -> dict:
    """Build a matrix cell from a `Ledger.compare()` result, so the matrix consumes REAL measurements. The
    paired difference is `rate_a - rate_b`; its CI is the exact-McNemar paired standard error; the ledger's
    warning strings (deployment/provenance/code/completeness/budget) travel through as caveats so a cell never
    outlives its reasons to doubt (the L6 rule, applied to the matrix)."""
    import math

    diff = res["rate_a"] - res["rate_b"]
    b, c = res.get("only_a", 0), res.get("only_b", 0)
    if n > 0 and (b or c):
        se = math.sqrt(max(0.0, b + c - (b - c) ** 2 / n)) / n
    else:
        se = 0.0
    caveats = {k: v for k, v in (
        ("deployment", res.get("deployment_warning")), ("provenance", res.get("provenance_warning")),
        ("code", res.get("code_warning")), ("completeness", res.get("completeness_warning")),
        ("budget", res.get("budget_note"))) if v}
    return {"game": game, "diff": diff, "p": res["p"], "n": n,
            "ci": (diff - 1.96 * se, diff + 1.96 * se), "null_below": null_below, "caveats": caveats}


def cell_from_seed_deltas(game: str, deltas: list, null_below: float = 0.03) -> dict:
    """Build a matrix cell whose unit of replication is the TRAINING RUN, not the opening. `deltas` is one paired
    (variant - baseline) score delta PER TRAINING SEED, measured on a fixed opening set. Testing their mean with
    a between-seed t-test answers "does the lever move NETS" rather than "did these two particular nets differ on
    the openings" — the single-net-vs-single-net confound §C.20 exists to kill (§C.19). One seed cannot cancel
    training-run variance, so it is returned with p=1.0 and a standing caveat, never as a significant result."""
    import math
    import statistics

    from harness.measurement import student_t_two_sided_p, t_critical

    r = len(deltas)
    if r == 0:
        raise ValueError("no seed deltas")
    effect = statistics.fmean(deltas)
    if r == 1:
        return {"game": game, "diff": effect, "p": 1.0, "n": 1, "ci": (effect - 1.0, effect + 1.0),
                "null_below": null_below, "n_seeds": 1,
                "caveats": {"provenance": "single training draw — training-seed variance is uncancelled; the "
                                          "p reflects opening variance only, not whether the lever moves nets"}}
    sd = statistics.stdev(deltas)
    se = sd / math.sqrt(r)
    if se == 0.0:
        # every seed gave the identical delta: no observed spread. A non-zero identical effect is perfectly
        # consistent (p->0); an all-zero set is a true no-op (p=1).
        p = 0.0 if effect != 0.0 else 1.0
        half = 0.0
    else:
        t = effect / se
        p = student_t_two_sided_p(t, r - 1)
        half = t_critical(r - 1, 0.05) * se
    return {"game": game, "diff": effect, "p": p, "n": r, "ci": (effect - half, effect + half),
            "null_below": null_below, "n_seeds": r, "caveats": {}}


def is_saturated(rates: list, floor: float = 0.05, ceiling: float = 0.95) -> bool:
    """Both arms pinned at the metric's floor or ceiling, so the cell CANNOT discriminate the change — every
    rate near 0 (both always lose) or near 1 (both never lose). Found on the first real sweep (§C.40): morris
    sat at 0.97-1.0 against a bounded refuter (a draw-heavy game the refuter can't force wins in), so its cell
    carried a tiny meaningless delta. A cell in this regime is uninformative, not a measured null."""
    if not rates:
        return False
    lo, hi = min(rates), max(rates)
    return hi <= floor or lo >= ceiling
