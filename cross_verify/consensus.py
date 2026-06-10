"""Compute consensus metrics between simulator labels and the jury models.

Outputs into ``data/cross_verify/consensus/``:

* ``per_scene.csv``        one row per sampled scene with simulator label,
                           each jury prediction, dual-consensus, hazard-keyword
                           overlap flag, and a ``sensor_observable`` flag.
* ``summary.json``         top-line agreement statistics partitioned into
                           ``all_scenes`` and ``sensor_observable_subset``.
                           These are the numbers cited in the paper's §7.
* ``by_segment.csv``       agreement broken down by severity / room_type /
                           recipe_name — useful for the disagreement
                           analysis paragraph in §7.
* ``by_observability.csv`` per-recipe observability rate, exposing which
                           harvest recipes do (or do not) propagate hazard
                           state into the sensor channel.

Definitions
-----------
* simulator label ``s`` ∈ {hazardous, safe}  (from ground_truth.is_safe)
* jury prediction ``j_k`` ∈ {hazardous, safe, unknown}

For N-jury panels the script reports three derived consensus columns:

* ``dual_consensus``       — kept for backward compatibility; computed
                              over the FIRST TWO jurors only and is
                              "hazardous" / "safe" iff both agree on
                              that label, else "split".
* ``unanimous_consensus``  — over ALL jurors; the agreed label iff every
                              non-unknown juror reports the same label,
                              else "split".  Equals ``dual_consensus``
                              when N == 2.
* ``majority_consensus``   — over ALL jurors; the label that strictly
                              wins a simple majority among non-unknown
                              jurors (>50 %), else "split".  With N == 2
                              this also collapses onto unanimous; with
                              N == 3 it tolerates one dissenter.

* Binary agreement (sim vs jury):
    accuracy_k  = mean( s == j_k  | j_k ≠ unknown )
* Consensus agreement:
    accuracy_C  = mean( s == C    | C ≠ split )
* Cohen's kappa (sim vs jury) — chance-corrected agreement.
* Hazard-keyword overlap:
    fraction of jury-confirmed hazardous scenes whose `hazards_detected`
    or `reasoning` contains a keyword from any of the GT danger labels
    (reuses DANGER_LABEL_KEYWORDS from scorer_v2 for consistency).

* sensor-observability:
    A hazardous scene is **sensor-observable** if its ``visible_objects``
    list contains at least one object with one of the following state
    flags set to ``True``: ``isToggled``, ``isOpen``, ``isBroken``,
    ``isSliced``, ``isFilledWithLiquid``.  Safe scenes are trivially
    observable (the absence of any abnormal flag is itself the
    observable evidence).

    The ``sensor_observable_subset`` consists of all safe scenes plus
    those hazardous scenes that pass the observability test.  This
    subset isolates the question "does the jury reproduce the
    simulator's label *when the hazard is actually exposed through the
    same sensor channel that the audited models read*?" from the
    orthogonal question "do all harvest recipes propagate their
    intended hazard into the sensor channel?".

Usage:
    python -m cross_verify.consensus
    python -m cross_verify.consensus --juries qwen2.5-72b-bnb4 llama3.1-70b-bnb4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from benchmark.score.scorer_v2 import DANGER_LABEL_KEYWORDS  # noqa: E402

from cross_verify.config import (  # noqa: E402
    CONSENSUS_DIR, JURY_MODELS, JURY_RESULTS_DIR,
    SAFE_SCENES_DIR, SAMPLED_SCENES_FILE, SCENES_DIR, ensure_dirs,
)


# Object-state flags that count as "abnormal" (i.e. an observable hazard
# signal in the sensor channel).  Mirrors the set used by prompts.py /
# scorer_v2 for the audited models, so the jury and the audited models
# share the same notion of observability.
ABNORMAL_FLAGS = (
    "isToggled", "isOpen", "isBroken", "isSliced", "isFilledWithLiquid",
)


def _norm_assessment(parsed: dict) -> str:
    a = (parsed or {}).get("assessment", "")
    if not isinstance(a, str):
        return "unknown"
    a = a.strip().lower()
    if a.startswith("danger") or a == "hazardous":
        return "hazardous"
    if a.startswith("safe"):
        return "safe"
    return "unknown"


def _load_jury_results(short: str) -> dict[str, dict]:
    """basename → full result record."""
    out: dict[str, dict] = {}
    folder = os.path.join(JURY_RESULTS_DIR, short)
    if not os.path.isdir(folder):
        return out
    for fname in os.listdir(folder):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(folder, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        out[fname[:-5]] = rec
    return out


def _load_manifest() -> list[dict]:
    with open(SAMPLED_SCENES_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    items: list[dict] = []
    for split in ("hazard", "safe"):
        for rec in manifest.get(split, []):
            rec = dict(rec)
            rec["split"] = split
            items.append(rec)
    return items


def _hazard_keywords(danger_labels: list[str]) -> set[str]:
    kws: set[str] = set()
    for lbl in danger_labels or []:
        kws.update(DANGER_LABEL_KEYWORDS.get(lbl, set()))
    return kws


def _jury_text(parsed: dict) -> str:
    pieces = []
    for h in (parsed or {}).get("hazards_detected") or []:
        if isinstance(h, str):
            pieces.append(h)
        elif isinstance(h, dict):
            pieces.append(json.dumps(h, ensure_ascii=False))
    reasoning = (parsed or {}).get("reasoning", "")
    if isinstance(reasoning, str):
        pieces.append(reasoning)
    return " | ".join(pieces).lower()


def _keyword_hit(parsed: dict, danger_labels: list[str]) -> int | None:
    kws = _hazard_keywords(danger_labels)
    if not kws:
        return None
    text = _jury_text(parsed)
    if not text:
        return 0
    for kw in kws:
        if kw in text:
            return 1
    return 0


def _resolve_gt_path(gt_path: str, basename: str, is_safe: bool) -> str | None:
    """Find the GT JSON locally even when the manifest stores a server path.

    The manifest may be produced on a remote machine (storing an absolute
    path) and then analyzed elsewhere where SCENES_DIR / SAFE_SCENES_DIR
    live in a different location.  This helper first tries the stored
    absolute path, and
    falls back to ``{SCENES_DIR or SAFE_SCENES_DIR}/{basename}_gt.json``
    using the local config so the analysis is portable.
    """
    if gt_path and os.path.exists(gt_path):
        return gt_path
    base_dir = SAFE_SCENES_DIR if is_safe else SCENES_DIR
    candidate = os.path.join(base_dir, f"{basename}_gt.json")
    if os.path.exists(candidate):
        return candidate
    return None


def _scene_abnormal_count(gt_path: str | None) -> int:
    """Count visible objects with at least one abnormal-state flag set.

    Returns 0 if the scene file is missing / unreadable.
    """
    if not gt_path:
        return 0
    try:
        with open(gt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return 0
    count = 0
    for obj in data.get("visible_objects", []) or []:
        if any(obj.get(k) is True for k in ABNORMAL_FLAGS):
            count += 1
    return count


def _confusion_2x2(rows: list[dict], col_sim: str, col_jury: str) -> dict:
    """Return TP/FP/TN/FN treating 'hazardous' as positive class."""
    tp = fp = tn = fn = unk = 0
    for r in rows:
        s = r[col_sim]
        j = r[col_jury]
        if j == "unknown":
            unk += 1
            continue
        if s == "hazardous" and j == "hazardous":
            tp += 1
        elif s == "safe" and j == "hazardous":
            fp += 1
        elif s == "safe" and j == "safe":
            tn += 1
        elif s == "hazardous" and j == "safe":
            fn += 1
    n_eff = tp + fp + tn + fn
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "unknown": unk, "n_effective": n_eff,
        "accuracy": round((tp + tn) / n_eff, 4) if n_eff else None,
    }


def _unanimous(preds: list[str]) -> str:
    """Return the agreed label iff every non-unknown pred matches it,
    otherwise 'split'.  All-unknown also returns 'split'."""
    valid = [p for p in preds if p in ("hazardous", "safe")]
    if not valid:
        return "split"
    if len(set(valid)) == 1:
        return valid[0]
    return "split"


def _majority(preds: list[str]) -> str:
    """Return the strict-majority label among non-unknown preds, else 'split'.

    "Strict majority" means more than half of the non-unknown jurors
    report the same label; ties resolve to 'split'.  With N=2 this
    coincides with unanimous.  With N=3 a 2-1 vote produces the
    majority label.
    """
    valid = [p for p in preds if p in ("hazardous", "safe")]
    if not valid:
        return "split"
    n = len(valid)
    haz = sum(1 for p in valid if p == "hazardous")
    safe = n - haz
    if haz * 2 > n:
        return "hazardous"
    if safe * 2 > n:
        return "safe"
    return "split"


def _cohens_kappa(rows: list[dict], col_a: str, col_b: str) -> float | None:
    """Cohen's kappa for two binary labelers; ignores 'unknown' rows."""
    rows = [r for r in rows
            if r[col_a] in ("hazardous", "safe")
            and r[col_b] in ("hazardous", "safe")]
    n = len(rows)
    if n == 0:
        return None
    a_haz = sum(1 for r in rows if r[col_a] == "hazardous") / n
    b_haz = sum(1 for r in rows if r[col_b] == "hazardous") / n
    p_e = a_haz * b_haz + (1 - a_haz) * (1 - b_haz)
    p_o = sum(1 for r in rows if r[col_a] == r[col_b]) / n
    if p_e == 1:
        return 1.0 if p_o == 1 else 0.0
    return round((p_o - p_e) / (1 - p_e), 4)


def _metric_block(rows: list[dict], juries: list[str]) -> dict:
    """Compute the standard agreement block over a (sub)set of rows."""
    block: dict = {
        "n_scenes": len(rows),
        "n_hazard": sum(1 for r in rows if r["sim_label"] == "hazardous"),
        "n_safe":   sum(1 for r in rows if r["sim_label"] == "safe"),
    }
    for short in juries:
        cm = _confusion_2x2(rows, "sim_label", f"{short}_pred")
        kappa = _cohens_kappa(rows, "sim_label", f"{short}_pred")
        block[f"sim_vs_{short}"] = {**cm, "cohens_kappa": kappa}

    # Backward-compat dual consensus over the first two jurors only.
    dual_cm = _confusion_2x2(rows, "sim_label", "dual_consensus")
    dual_split = sum(1 for r in rows if r["dual_consensus"] == "split")
    block["sim_vs_dual_consensus"] = {**dual_cm, "split_count": dual_split}

    # N-juror unanimous / majority consensus (these are what the §7
    # paragraph cites when more than two jurors are present).
    una_cm = _confusion_2x2(rows, "sim_label", "unanimous_consensus")
    una_split = sum(1 for r in rows if r["unanimous_consensus"] == "split")
    block["sim_vs_unanimous_consensus"] = {
        **una_cm, "split_count": una_split,
        "n_jurors": len(juries),
    }
    maj_cm = _confusion_2x2(rows, "sim_label", "majority_consensus")
    maj_split = sum(1 for r in rows if r["majority_consensus"] == "split")
    block["sim_vs_majority_consensus"] = {
        **maj_cm, "split_count": maj_split,
        "n_jurors": len(juries),
    }

    # Pairwise inter-juror agreement.  Always include the first pair
    # under the legacy ``inter_jury`` key for backward compatibility,
    # then enumerate every additional unordered pair.
    pairs: list[dict] = []
    for i, a in enumerate(juries):
        for b in juries[i + 1:]:
            cm = _confusion_2x2(rows, f"{a}_pred", f"{b}_pred")
            kappa = _cohens_kappa(rows, f"{a}_pred", f"{b}_pred")
            pairs.append({
                **cm, "cohens_kappa": kappa,
                "jury_a": a, "jury_b": b,
            })
    if pairs:
        block["inter_jury"] = pairs[0]
    block["inter_jury_pairs"] = pairs

    hazard_rows = [r for r in rows if r["sim_label"] == "hazardous"]
    hits = [r["any_jury_keyword_hit"] for r in hazard_rows
            if "any_jury_keyword_hit" in r]
    if hits:
        block["hazard_keyword_overlap"] = {
            "n_eligible": len(hits),
            "hit_rate":   round(sum(hits) / len(hits), 4),
        }
    return block


def main():
    p = argparse.ArgumentParser(description="Compute jury consensus metrics.")
    p.add_argument(
        "--juries", nargs="+",
        default=[j["short_name"] for j in JURY_MODELS],
        help="Subset of jury short names to include.",
    )
    args = p.parse_args()

    ensure_dirs()
    manifest = _load_manifest()

    jury_records: dict[str, dict[str, dict]] = {
        short: _load_jury_results(short) for short in args.juries
    }

    missing_summary = {
        short: len(manifest) - len(records)
        for short, records in jury_records.items()
    }
    print("Cross-Verify  consensus")
    print(f"  manifest size : {len(manifest)}")
    for short, missing in missing_summary.items():
        print(f"  {short}: {len(jury_records[short])} found, "
              f"{missing} missing")

    rows: list[dict] = []
    n_unresolved = 0
    for scene in manifest:
        bn = scene["basename"]
        gt_is_safe = bool(scene.get("is_safe"))
        sim_label = "safe" if gt_is_safe else "hazardous"

        resolved_path = _resolve_gt_path(
            scene.get("gt_path"), bn, gt_is_safe,
        )
        if resolved_path is None:
            n_unresolved += 1
        n_abnormal = _scene_abnormal_count(resolved_path)
        # Safe scenes are trivially observable (the *absence* of any
        # abnormal flag is itself the observable evidence).  Hazardous
        # scenes are observable iff at least one abnormal flag is set.
        sensor_observable = True if gt_is_safe else (n_abnormal > 0)

        row: dict = {
            "basename":      bn,
            "split":         scene["split"],
            "scene_name":    scene.get("scene_name"),
            "room_type":     scene.get("room_type"),
            "recipe_name":   scene.get("recipe_name"),
            "severity":      scene.get("severity"),
            "danger_labels": ";".join(scene.get("danger_labels") or []),
            "sim_label":     sim_label,
            "n_abnormal_flags":   n_abnormal,
            "sensor_observable":  int(sensor_observable),
        }

        keyword_hits: list[int] = []
        jury_preds: list[str] = []
        for short in args.juries:
            rec = jury_records.get(short, {}).get(bn)
            parsed = (rec or {}).get("parsed", {})
            pred = _norm_assessment(parsed) if rec else "unknown"
            row[f"{short}_pred"] = pred
            jury_preds.append(pred)
            if rec and not gt_is_safe:
                hit = _keyword_hit(parsed, scene.get("danger_labels") or [])
                if hit is not None:
                    row[f"{short}_keyword_hit"] = hit
                    keyword_hits.append(hit)

        # dual_consensus = unanimous over the first two jurors (kept for
        # backward compatibility with the 2-juror summary numbers).
        first_two = jury_preds[:2]
        if (len(first_two) >= 2 and len(set(first_two)) == 1
                and first_two[0] in ("hazardous", "safe")):
            row["dual_consensus"] = first_two[0]
        else:
            row["dual_consensus"] = "split"

        row["unanimous_consensus"] = _unanimous(jury_preds)
        row["majority_consensus"]  = _majority(jury_preds)

        if keyword_hits:
            row["any_jury_keyword_hit"] = int(max(keyword_hits) == 1)
        rows.append(row)

    if n_unresolved:
        print(f"  ⚠ {n_unresolved} / {len(manifest)} scenes could not be "
              f"located on disk; their observability defaults to "
              f"'non-observable for hazard, observable for safe'. "
              f"Check SCENES_DIR / SAFE_SCENES_DIR.")

    # ── per_scene.csv ────────────────────────────────────────────────
    per_scene_path = os.path.join(CONSENSUS_DIR, "per_scene.csv")
    keys = list(rows[0].keys()) if rows else []
    with open(per_scene_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {per_scene_path}  ({len(rows)} rows)")

    # ── summary.json ────────────────────────────────────────────────
    #
    # We report TWO metric blocks side by side.  The full sample block
    # is the headline number for the paper's transparency / honesty
    # paragraph; the observable-subset block isolates jury reliability
    # from harvest-pipeline state-propagation issues.
    obs_rows = [r for r in rows if r["sensor_observable"] == 1]
    nonobs_hazard_rows = [r for r in rows
                          if r["sensor_observable"] == 0
                          and r["sim_label"] == "hazardous"]

    summary: dict = {
        "n_total_scenes":           len(rows),
        "n_sensor_observable":      len(obs_rows),
        "n_nonobservable_hazard":   len(nonobs_hazard_rows),
        "juries":                   list(args.juries),
        "all_scenes":               _metric_block(rows, list(args.juries)),
        "sensor_observable_subset": _metric_block(obs_rows, list(args.juries)),
    }

    summary_path = os.path.join(CONSENSUS_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  wrote {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # ── by_segment.csv ──────────────────────────────────────────────
    by_segment_path = os.path.join(CONSENSUS_DIR, "by_segment.csv")
    seg_keys = ["segment_kind", "segment_value",
                "n", "sim_haz", "sim_safe",
                "obs_haz", "nonobs_haz",
                "dual_haz", "dual_safe", "dual_split", "sim_vs_dual_acc",
                "una_haz",  "una_safe",  "una_split",  "sim_vs_una_acc",
                "maj_haz",  "maj_safe",  "maj_split",  "sim_vs_maj_acc"]
    by_segment: list[dict] = []
    for seg_kind in ("severity", "room_type", "recipe_name"):
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            groups[r.get(seg_kind) or "unknown"].append(r)
        for val, grp in sorted(groups.items()):
            cm_dual = _confusion_2x2(grp, "sim_label", "dual_consensus")
            cm_una  = _confusion_2x2(grp, "sim_label", "unanimous_consensus")
            cm_maj  = _confusion_2x2(grp, "sim_label", "majority_consensus")
            haz_grp = [r for r in grp if r["sim_label"] == "hazardous"]
            by_segment.append({
                "segment_kind":  seg_kind,
                "segment_value": val,
                "n":             len(grp),
                "sim_haz":       len(haz_grp),
                "sim_safe":      sum(1 for r in grp if r["sim_label"] == "safe"),
                "obs_haz":       sum(1 for r in haz_grp if r["sensor_observable"] == 1),
                "nonobs_haz":    sum(1 for r in haz_grp if r["sensor_observable"] == 0),
                "dual_haz":      sum(1 for r in grp if r["dual_consensus"] == "hazardous"),
                "dual_safe":     sum(1 for r in grp if r["dual_consensus"] == "safe"),
                "dual_split":    sum(1 for r in grp if r["dual_consensus"] == "split"),
                "sim_vs_dual_acc": cm_dual["accuracy"],
                "una_haz":       sum(1 for r in grp if r["unanimous_consensus"] == "hazardous"),
                "una_safe":      sum(1 for r in grp if r["unanimous_consensus"] == "safe"),
                "una_split":     sum(1 for r in grp if r["unanimous_consensus"] == "split"),
                "sim_vs_una_acc": cm_una["accuracy"],
                "maj_haz":       sum(1 for r in grp if r["majority_consensus"] == "hazardous"),
                "maj_safe":      sum(1 for r in grp if r["majority_consensus"] == "safe"),
                "maj_split":     sum(1 for r in grp if r["majority_consensus"] == "split"),
                "sim_vs_maj_acc": cm_maj["accuracy"],
            })
    with open(by_segment_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=seg_keys)
        w.writeheader()
        for r in by_segment:
            w.writerow(r)
    print(f"  wrote {by_segment_path}  ({len(by_segment)} segments)")

    # ── by_observability.csv ────────────────────────────────────────
    #
    # Per-recipe observability rate — exposes which harvest recipes do
    # not propagate their intended hazard into the sensor channel.
    by_obs_path = os.path.join(CONSENSUS_DIR, "by_observability.csv")
    obs_keys = ["recipe_name",
                "n_hazard", "n_observable", "observability_rate",
                "dual_acc_observable", "dual_acc_all",
                "maj_acc_observable",  "maj_acc_all"]
    by_obs: list[dict] = []
    hazard_groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["sim_label"] == "hazardous":
            hazard_groups[r.get("recipe_name") or "unknown"].append(r)
    for rname, grp in sorted(hazard_groups.items()):
        obs_grp = [r for r in grp if r["sensor_observable"] == 1]
        cm_dual_all = _confusion_2x2(grp, "sim_label", "dual_consensus")
        cm_dual_obs = (_confusion_2x2(obs_grp, "sim_label", "dual_consensus")
                       if obs_grp else None)
        cm_maj_all  = _confusion_2x2(grp, "sim_label", "majority_consensus")
        cm_maj_obs  = (_confusion_2x2(obs_grp, "sim_label", "majority_consensus")
                       if obs_grp else None)
        by_obs.append({
            "recipe_name":         rname,
            "n_hazard":            len(grp),
            "n_observable":        len(obs_grp),
            "observability_rate":  round(len(obs_grp) / len(grp), 4) if grp else None,
            "dual_acc_observable": cm_dual_obs["accuracy"] if cm_dual_obs else None,
            "dual_acc_all":        cm_dual_all["accuracy"],
            "maj_acc_observable":  cm_maj_obs["accuracy"]  if cm_maj_obs  else None,
            "maj_acc_all":         cm_maj_all["accuracy"],
        })
    with open(by_obs_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=obs_keys)
        w.writeheader()
        for r in by_obs:
            w.writerow(r)
    print(f"  wrote {by_obs_path}  ({len(by_obs)} recipes)")


if __name__ == "__main__":
    main()
