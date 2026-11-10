# Publication sanitization

The original publication is commit `6dfea6ecb12c63d7036b43fe14fa3e4b989c8734`. The report, tables, and all 33 rendered figure files retain their original bytes.

- In `plot_data/bet_patch_strength_curves.json`, `plot_data/nextcard_patch_strength_curves.json`, and `plot_data/sensitivity_signed_matrices.json`, source paths now start at the experiment workspace root. Relative suffixes, source artifact hashes, and every result value are unchanged.
- In `evidence/figure_manifest.json`, the interpreter is `python3`, module paths are relative to `site-packages`, and the rendering command uses the runner path relative to the archive root. Interpreter and package versions are preserved. The three sanitized plot-data files have updated hashes.
- In `evidence/final_headline_completion.json`, `figure_manifest.sha256` identifies the archived original. The added `figure_manifest.publication_sha256` verifies the copy in this repository.
- In `evidence/scope.json`, the verbatim internal instruction was removed. The required workloads, budgets, and training-dynamics decision are retained.
- The publication manifest records updated byte counts and SHA-256 digests for the seven changed payloads and the new sanitization record. Archive input, script, and historical report digests continue to identify the original evidence they refer to.

For the public copy, 2 cancelled jobs in `evidence/compute_usage.csv` retain the `CANCELLED` status with the scheduler account identifiers removed. Job IDs, usage values, and all other columns are unchanged. The publication manifest records the new hashes and byte counts for the CSV and this note.
