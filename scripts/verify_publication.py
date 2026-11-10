"""Verify the distributed report without loading data or models."""

import ast
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/three-seed-2026-10-14"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_file(path, expected_sha256):
    require(
        hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256,
        f"checksum mismatch: {path.relative_to(ROOT)}",
    )


def verify_manifests():
    publication = read_json(REPORT / "publication_manifest.json")
    payloads = set()
    for entry in publication["files"]:
        path = (ROOT / entry["repository_path"]).resolve()
        require(path.is_relative_to(REPORT), "publication path outside report")
        require(path not in payloads, f"duplicate publication path: {path}")
        verify_file(path, entry["sha256"])
        require(path.stat().st_size == entry["bytes"], f"size mismatch: {path}")
        payloads.add(path)
    expected = {REPORT / name for name in ("README.md", "report.md", "SANITIZATION.md")}
    for directory in ("evidence", "figures", "plot_data", "tables"):
        expected.update(
            path for path in (REPORT / directory).rglob("*") if path.is_file()
        )
    require(payloads == expected, "publication manifest does not cover the payload")

    evidence = REPORT / "evidence"
    completion = read_json(evidence / "final_headline_completion.json")
    manifest = evidence / "figure_manifest.json"
    verify_file(manifest, completion["figure_manifest"]["publication_sha256"])
    verify_file(
        evidence / "collection_report.json", completion["collection_report"]["sha256"]
    )
    verify_file(REPORT / "report.md", completion["report_sha256"])
    verify_file(REPORT / "report.md", publication["original_report_sha256"])
    for name, digest in completion["headline_tables"].items():
        verify_file(REPORT / "tables" / name, digest)

    figures = read_json(manifest)
    require(
        figures["omitted_figures"] == ["training_dynamics"],
        "unexpected figure omission",
    )
    require(
        completion["training_dynamics"].startswith("OMITTED"),
        "unexpected dynamics status",
    )
    require(len(figures["figures"]) == 11, "expected eleven retained figures")
    listed = set()
    for figure in figures["figures"]:
        name = figure["id"]
        paths = {f"figures/{name}.{extension}" for extension in ("pdf", "png", "svg")}
        paths.add(f"plot_data/{name}.json")
        require(figure["files"].keys() == paths, f"unexpected figure files: {name}")
        for relative, digest in figure["files"].items():
            verify_file(REPORT / relative, digest)
            listed.add(REPORT / relative)
    require(
        listed == {p for p in payloads if p.parent.name in ("figures", "plot_data")},
        "figure manifest does not cover all figures and plot data",
    )
    return len(payloads), len(figures["figures"])


def verify_results():
    tables = REPORT / "tables"
    headline = read_json(tables / "headline_results.json")
    aggregate = read_json(tables / "aggregate_results.json")
    require(
        headline["omitted"] == ["training_dynamics"], "unexpected headline omission"
    )
    with (tables / "headline_results.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    require(
        len(csv_rows) == len(headline["rows"]) == 48, "unexpected headline row count"
    )
    for row, csv_row in zip(headline["rows"], csv_rows, strict=True):
        label = row["paper_id"]
        fields = ("paper_id", "condition", "metric", "sample_unit", "n_training_seeds")
        require(
            all(csv_row[key] == str(row[key]) for key in fields),
            f"CSV row mismatch: {label}",
        )
        values = {key: row[key] for key in ("mean", "sample_sd", "paper_value")}
        values.update(
            {f"seed_{seed}": row["per_seed"].get(seed) for seed in ("0", "1", "2")}
        )
        values["n_eval_by_seed"] = row["n_eval_by_seed"]
        for key, value in values.items():
            actual = csv_row[key]
            if value is None:
                actual = None if actual == "" else actual
            elif isinstance(value, (int, float)):
                actual = float(actual)
            elif isinstance(value, dict):
                actual = ast.literal_eval(actual)
            require(actual == value, f"CSV value mismatch: {label}/{key}")

    for row in headline["rows"] + aggregate["metrics"]:
        label = f"{row['condition']}/{row['metric']}"
        if row["n_training_seeds"] == 0:
            require(row["sample_sd"] is None, f"unexpected baseline SD: {label}")
            continue
        per_seed = row["per_seed"]
        require(
            row["n_training_seeds"] == 3 and per_seed.keys() == {"0", "1", "2"},
            f"incorrect seed coverage: {label}",
        )
        values = per_seed.values()
        for key, expected in (
            ("mean", statistics.mean(values)),
            ("sample_sd", statistics.stdev(values)),
        ):
            require(
                math.isclose(row[key], expected, rel_tol=1e-12, abs_tol=1e-14),
                f"incorrect {key}: {label}",
            )
        require(row.get("ddof", 1) == 1, f"incorrect ddof: {label}")
    return len(headline["rows"]), len(aggregate["metrics"])


if __name__ == "__main__":
    files, figures = verify_manifests()
    headline, aggregate = verify_results()
    print(
        f"Verified {files} payloads, {figures} figures, {headline} headline rows "
        f"and {aggregate} aggregate metrics."
    )
