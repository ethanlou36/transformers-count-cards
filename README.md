# TRANSFORMERS CARD COUNTING

Research artifacts for **Do Transformers Learn to Count Cards?**, accepted at the [NeurIPS 2026 Interpretability for Discovery workshop](https://interpretability4discovery.github.io/). The project studies whether transformers trained on simulated blackjack learn to track running count and deck composition, and whether those representations affect play.

- [Final three-seed report](reports/three-seed-2026-10-14/report.md) for findings, protocol corrections, and limitations
- [Headline results (CSV)](reports/three-seed-2026-10-14/tables/headline_results.csv) and [aggregate results (JSON)](reports/three-seed-2026-10-14/tables/aggregate_results.json)
- [Figures, plot data, and verification evidence](reports/three-seed-2026-10-14/README.md#figures)

The retained report covers nine independently trained models across three conditions, 17 EV cohorts, and six finite-deck models scored on the same 75 matched-history pairs. Model-dependent headline results give all three seeds and mean ± sample SD (`ddof=1`). EV shoe-cluster sampling uncertainty is reported separately. The corrected protocol differs from the historical comparison, particularly for the IID count probe. Training dynamics were intentionally omitted, and the report makes no claim about when counting emerges.

## Match paper topics to the corrected results

The [paper comparison inventory](reports/three-seed-2026-10-14/evidence/paper_comparison_36_rows.json) records the historical comparisons and protocol differences. Use the corrected artifacts below when citing this release.

| Paper topic | Corrected artifact |
| --- | --- |
| Running-count probes and finite-deck versus IID controls | [Probe figure](reports/three-seed-2026-10-14/figures/main_probes_finite_iid.pdf) |
| Next-card rank sensitivity | [Signed matrices](reports/three-seed-2026-10-14/figures/sensitivity_signed_matrices.pdf) and [direction and amplitude](reports/three-seed-2026-10-14/figures/sensitivity_direction_and_amplitude.pdf) |
| Gameplay EV and ablations | [Final EV table](reports/three-seed-2026-10-14/report.md#ev-table) and [per-seed EV](reports/three-seed-2026-10-14/tables/ev_per_seed.csv) |
| Matched-history evaluation | [Matched-history figure](reports/three-seed-2026-10-14/figures/matched_history.pdf) |
| Training dynamics | Omitted from the corrected experiments |

## Check from a fresh clone

From the repository root, use Python 3.11 or later. The report check uses only the standard library and verifies the publication and figure manifests, the headline CSV and JSON, and the published seed summaries.

```sh
python3 scripts/verify_publication.py
```

For a small CPU smoke check, install the package and generate and replay three synthetic shoes. The commands need no dataset download, credentials, or model.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
blackjackgpt-data generate-shard --start-shoe-index 0 --shoe-count 3 \
  --seed 0 --split smoke --output-dir data/camera-ready-smoke
blackjackgpt-data verify-shard --shard data/camera-ready-smoke --replay
```

The clone includes the [100-shoe synthetic fixture](data/fixtures/blackjack-v1-100-shoes/), [source code](src/blackjackgpt/), and [example training configs](configs/). Install `.[ml]` in the same environment for the training and analysis modules.

## Reproduce the final experiments with the separate archive

Obtain the original hash-pinned experiment workspace from the authors. It contains the frozen datasets and split manifests, resolved configs and source, nine checkpoints, matched histories, saved predictions and gameplay traces, and the reviewed runners. Use the saved outputs to regenerate the report or the archived configs and data to rerun training and evaluation. The configs in this clone are examples for the public code. Follow the [archived protocol and regeneration instructions](reports/three-seed-2026-10-14/report.md#deliverables-and-regeneration) from the restored workspace.

```sh
# Run from the root of the restored experiment workspace.
python3 -B scripts/regenerate_final_headline.py --output results/NEW_DIRECTORY
```

This archive-only command reads saved outputs and does no training or fitting. The publication check above verifies the distributed evidence and cannot verify raw data or repeat model inference. See the [sanitization record](reports/three-seed-2026-10-14/SANITIZATION.md) for the metadata edits and which checksums refer to the original archive.

### Optional archive tooling

The local checks require no environment variables. Set the following only for the corresponding optional workflow.

| Variable | When to set it | Safe example or source |
| --- | --- | --- |
| `BLACKJACKGPT_HUB_REPO_ID` | Private Hugging Face dataset transfer with `.[hub]`. You can also use `--repo-id` on transfer commands. | `example-org/example-dataset` |
| `HF_TOKEN` | Private Hub authentication if you are using an injected token. Interactive users can use `hf auth login`. | Inject from your secret store. |
| `BLACKJACKGPT_HUB_REVISION` | Optional provenance label when merging downloaded shards. Pass `--revision` separately to the transfer command. | `example-release-tag` |
| `BLACKJACKGPT_PRODUCTION_SOURCE_REPOSITORY` | Required by the fixed production dataset generator. Must match the clean archive checkout’s Git origin with the pinned runtime and fetched source ref. | `https://example.org/org/repo.git` |
| `PYTHON` | Override the interpreter for `scripts/run_full_pipeline.sh`. This general pipeline requires a sharded release and is separate from the final report runners. | `.venv/bin/python` |
