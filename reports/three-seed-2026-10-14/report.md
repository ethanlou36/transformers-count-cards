# JackGPT: three independent training seeds

Finalized 2026-10-14 01:09 UTC. **Complete for the retained headline scope:** nine independently trained models, all seventeen full EV cohorts, and matched-history scoring for all six finite-deck models. All 35 retained paper-comparison entries are complete under the corrected protocol. Training dynamics is explicitly omitted; this report makes no claim about when counting emerges or how its development tracks training loss.

## Main findings

Finite-deck running-count R² is 0.98359 ± 0.00021 for λ=1 and 0.96032 ± 0.00337 for λ=0. The IID comparison gives 0.75780 ± 0.02785; this differs substantially from the historical paper reference and should not be described as an exact reproduction of its count-null comparison. Composition variance-weighted R² is 0.91798 ± 0.00842 and 0.93443 ± 0.01067.

The intact λ=1 model earns 1.44933 ± 0.01771% profit per unit wagered; λ=0 earns -0.58937 ± 0.00499%. Ablation outcomes vary considerably across training seeds. The composition clamp is not a reliable composition-only intervention because it also changes ordinary hand play. See the per-seed table and damage diagnostics before drawing causal conclusions.

Matched-history directional accuracy is 0.89333 ± 0.01333 for λ=1 and 0.45778 ± 0.04286 for λ=0 on the same 75 frozen pairs. This measures the sign of the expert-oriented logit shift, not the fraction of argmax action flips or both members answered correctly.

## EV table

EV is 100 × total profit / total wager. Each trained model is evaluated on 150,000 identical frozen shoe seeds (1,000,000,000–1,000,149,999), giving 2.55 million shoe-condition evaluations across the seventeen cohorts. Means and SD below are in percentage points. Scripted policies have no training-seed SD.

| Policy / intervention | Seed 0 | Seed 1 | Seed 2 | Mean ± training SD (%) | Paper (%) |
|---|---:|---:|---:|---:|---:|
| Scripted Hi-Lo counter | — | — | — | 1.50233 (scripted) | 1.502 |
| Scripted count-blind | — | — | — | -0.57976 (scripted) | -0.58 |
| λ=1 intact | 1.43494 | 1.44392 | 1.46911 | 1.44933 ± 0.01771 | 1.46 |
| λ=1 count clamp | -0.72413 | -0.01190 | 0.98584 | 0.08327 ± 0.85895 | 0.398 |
| λ=1 composition-variance k=1 clamp | 0.96200 | -0.74008 | 1.43229 | 0.55140 ± 1.14291 | -1.701 |
| λ=1 random k=1 clamp | 1.42849 | 1.42428 | 1.47951 | 1.44409 ± 0.03074 | 1.478 |
| λ=0 intact | -0.58529 | -0.58789 | -0.59494 | -0.58937 ± 0.00499 | -0.597 |

The training SD describes variation across three separately trained checkpoints. Gameplay sampling error is estimated separately by clustering entire shoes, including paired comparisons on the shared shoe seeds. It is not the training SD and hands within a shoe are not treated as independent. Per-checkpoint shoe-cluster SEs and 95% intervals are in the EV tables; paired differences and SEs are also saved.

## All scalar headline results

Every retained scalar mapping is reported below. Full sample counts per seed, frozen evaluation hashes, exact metrics and all auxiliary results are in the machine-readable aggregate tables. R² uses held-out positions from frozen shoe groups; bet correlations use hands; matched-history metrics use 75 pairs (150 unique source shoes).

| Audited result | Condition | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD | Evaluation unit | N by seed |
|---|---|---:|---:|---:|---:|---|---|
| rc_l1 / probe.card.trained.running_count.r2 | finite_lambda1 | 0.98347 | 0.98347 | 0.98383 | 0.98359 ± 0.00021 | positions | 30286, 30286, 30286 |
| rc_random_finite_lambda1 / probe.card.random_init.running_count.r2 | finite_lambda1 | 0.49319 | 0.49392 | 0.48821 | 0.49177 ± 0.00311 | positions | 30286, 30286, 30286 |
| rc_l0 / probe.card.trained.running_count.r2 | finite_lambda0 | 0.95722 | 0.95984 | 0.96391 | 0.96032 ± 0.00337 | positions | 30205, 30205, 30205 |
| rc_random_finite_lambda0 / probe.card.random_init.running_count.r2 | finite_lambda0 | 0.35274 | 0.35056 | 0.40262 | 0.36864 ± 0.02945 | positions | 30205, 30205, 30205 |
| rc_inf / probe.card.trained.running_count.r2 | infinite_lambda0 | 0.78463 | 0.72902 | 0.75975 | 0.75780 ± 0.02785 | positions | 30201, 30201, 30201 |
| rc_random_infinite_lambda0 / probe.card.random_init.running_count.r2 | infinite_lambda0 | 0.44491 | 0.43638 | 0.46640 | 0.44923 ± 0.01547 | positions | 30201, 30201, 30201 |
| rc_shuffled_labels / probe.card.trained.shuffled_running_count.r2 | finite_lambda1 | -0.00493 | -0.00501 | -0.00431 | -0.00475 ± 0.00038 | positions | 30286, 30286, 30286 |
| rc_shuffled_labels / probe.card.trained.shuffled_running_count.r2 | finite_lambda0 | -0.00381 | -0.00398 | -0.00379 | -0.00386 ± 0.00011 | positions | 30205, 30205, 30205 |
| rc_embedding_only / probe.card.trained_embedding.running_count.r2 | finite_lambda1 | 0.05535 | 0.05525 | 0.05532 | 0.05531 ± 0.00005 | positions | 30286, 30286, 30286 |
| rc_embedding_only / probe.card.trained_embedding.running_count.r2 | finite_lambda0 | 0.05479 | 0.05480 | 0.05488 | 0.05483 ± 0.00005 | positions | 30205, 30205, 30205 |
| changed_history_new / changed_history_trained_permuted_permuted_running_count_r2 | finite_lambda0 | 0.83987 | 0.92961 | 0.86936 | 0.87961 ± 0.04574 | positions | 18898, 18898, 18898 |
| changed_history_original / changed_history_trained_permuted_original_running_count_r2 | finite_lambda0 | 0.37470 | 0.39479 | 0.38533 | 0.38494 ± 0.01005 | positions | 18898, 18898, 18898 |
| composition_joint / probe.card.trained.remaining_composition.r2 | finite_lambda1 | 0.90858 | 0.92049 | 0.92486 | 0.91798 ± 0.00842 | positions | 30286, 30286, 30286 |
| composition_joint / probe.card.trained.remaining_composition.r2 | finite_lambda0 | 0.92800 | 0.92854 | 0.94675 | 0.93443 ± 0.01067 | positions | 30205, 30205, 30205 |
| composition_pr / representation.projected_covariance_participation_ratio | finite_lambda1 | 1.80078 | 1.95577 | 1.97328 | 1.90994 ± 0.09494 | positions | 30286, 30286, 30286 |
| composition_pr / representation.projected_covariance_participation_ratio | finite_lambda0 | 1.69567 | 1.86912 | 2.08578 | 1.88352 ± 0.19546 | positions | 30205, 30205, 30205 |
| next_card_matrix / sensitivity.fd1_800_vs_analytic_derivative.cosine | finite_lambda0 | 0.89519 | 0.90668 | 0.93932 | 0.91373 ± 0.02289 | positions | 25616, 25616, 25616 |
| ablation_rc_intact / ablation_damage.intact.refit_clamped.running_count.r2 | finite_lambda1 | 0.99523 | 0.99525 | 0.99588 | 0.99545 ± 0.00037 | positions | 5823, 5823, 5823 |
| ablation_rc_count / ablation_damage.count.refit_clamped.running_count.r2 | finite_lambda1 | 0.32932 | 0.32715 | 0.35140 | 0.33596 ± 0.01342 | positions | 5823, 5823, 5823 |
| ablation_bet_intact / gameplay.full.bet_count_correlation | l1_intact | 0.76829 | 0.76824 | 0.76826 | 0.76827 ± 0.00003 | hands | 1089315, 1088628, 1088709 |
| ablation_bet_count / gameplay.full.bet_count_correlation | l1_count_ablated | 0.14429 | 0.37567 | 0.65327 | 0.39108 ± 0.25484 | hands | 1094201, 1091340, 1094182 |
| ordinary_play_intact / gameplay.first15000.ordinary_basic_strategy_agreement | l1_intact | 0.99584 | 0.99708 | 0.99715 | 0.99669 ± 0.00074 | decisions | 131123, 131412, 131334 |
| ordinary_play_comp / gameplay.first15000.ordinary_basic_strategy_agreement | l1_compvar_ablated_k1 | 0.97017 | 0.90459 | 0.98356 | 0.95277 ± 0.04226 | decisions | 131322, 127720, 131897 |
| twins_finite_lambda1 / synthetic_pairs.directional_accuracy | finite_lambda1 | 0.89333 | 0.90667 | 0.88000 | 0.89333 ± 0.01333 | pairs | 75, 75, 75 |
| twins_finite_lambda0 / synthetic_pairs.directional_accuracy | finite_lambda0 | 0.50667 | 0.42667 | 0.44000 | 0.45778 ± 0.04286 | pairs | 75, 75, 75 |
| count_direction_std / ablation_damage.count_direction_std | finite_lambda1 | 2.60033 | 2.22611 | 3.37923 | 2.73522 ± 0.58828 | positions | 5823, 5823, 5823 |
| probe_direction_std / ablation_damage.probe_direction_std_min | finite_lambda1 | 0.00889 | 0.01055 | 0.01531 | 0.01158 ± 0.00333 | positions | 5823, 5823, 5823 |
| probe_direction_std / ablation_damage.probe_direction_std_max | finite_lambda1 | 0.19934 | 0.25474 | 0.25431 | 0.23613 ± 0.03186 | positions | 5823, 5823, 5823 |
| count_in_top10 / ablation_damage.count_in_top10 | finite_lambda1 | 0.03711 | 0.04461 | 0.03989 | 0.04053 ± 0.00379 | direction_pairs | 1, 1, 1 |
| true_count_probes / probe.card.trained.raw_true_count.r2 | finite_lambda1 | 0.94866 | 0.95058 | 0.94711 | 0.94878 ± 0.00174 | positions | 30286, 30286, 30286 |
| true_count_probes / probe.card.trained.raw_true_count.r2 | finite_lambda0 | 0.89254 | 0.87312 | 0.88463 | 0.88343 ± 0.00977 | positions | 30205, 30205, 30205 |
| depth_residualized_probes / probe.card.depth_partial.running_count.r2 | finite_lambda1 | 0.98357 | 0.98351 | 0.98388 | 0.98365 ± 0.00020 | positions | 30286, 30286, 30286 |
| depth_residualized_probes / probe.card.depth_partial.running_count.r2 | finite_lambda0 | 0.95729 | 0.95992 | 0.96404 | 0.96042 ± 0.00340 | positions | 30205, 30205, 30205 |
| depth_band_probes / probe.card.depth_band_0_10.running_count.r2 | finite_lambda1 | 0.99226 | 0.99127 | 0.99395 | 0.99249 ± 0.00135 | positions | 6841, 6841, 6841 |
| depth_band_probes / probe.card.depth_band_10_20.running_count.r2 | finite_lambda1 | 0.99155 | 0.99140 | 0.99232 | 0.99175 ± 0.00049 | positions | 7768, 7768, 7768 |
| depth_band_probes / probe.card.depth_band_20_30.running_count.r2 | finite_lambda1 | 0.99030 | 0.99050 | 0.98974 | 0.99018 ± 0.00039 | positions | 7713, 7713, 7713 |
| depth_band_probes / probe.card.depth_band_30_40.running_count.r2 | finite_lambda1 | 0.98820 | 0.98896 | 0.98668 | 0.98795 ± 0.00116 | positions | 7243, 7243, 7243 |
| depth_band_probes / probe.card.depth_band_0_10.running_count.r2 | finite_lambda0 | 0.98699 | 0.98634 | 0.98632 | 0.98655 ± 0.00038 | positions | 6837, 6837, 6837 |
| depth_band_probes / probe.card.depth_band_10_20.running_count.r2 | finite_lambda0 | 0.98084 | 0.98259 | 0.98438 | 0.98261 ± 0.00177 | positions | 7750, 7750, 7750 |
| depth_band_probes / probe.card.depth_band_20_30.running_count.r2 | finite_lambda0 | 0.97691 | 0.98178 | 0.98002 | 0.97957 ± 0.00246 | positions | 7686, 7686, 7686 |
| depth_band_probes / probe.card.depth_band_30_40.running_count.r2 | finite_lambda0 | 0.97195 | 0.97439 | 0.97432 | 0.97355 ± 0.00139 | positions | 7211, 7211, 7211 |

Activation patching is represented by dose-response curves with each seed and across-seed mean ± SD, rather than one arbitrary scalar. The verified patch cohorts and axes are preserved in the collection and rendered figures. Finite-λ=1 bet patching, finite-λ=0 next-card patching and next-card sensitivity each cover all three training seeds.

## Protocol, provenance and interpretation

These are corrected experiments, not exact historical reruns. Source is the archived paper-v1.3-results commit `6309918eea9384e195db794007a6de39554b5d03`; the separate matched-history generator originated at `72e422940c01da36d91451041cb44e51acc59c20`. The timestamped local and Bouchet workspaces are isolated; the original checkout, paper and GitHub were not modified. No historical trained checkpoint was reused.

Each condition uses independent initialization and training order for seeds 0, 1 and 2, with fixed data, architecture, optimization duration and checkpoint-selection rule. Frozen release identities preserve original evaluation shoes outside training. Probe fitting and layer/regularization selection use fitting shoes with grouped folds; final test shoes are excluded from tuning. The core probe pool has 1,200 fit shoes and 800 test shoes per model. Exact configurations, training/checkpoint hashes, split manifests and validation histories remain in the request and completed-training reviews.

Historical source discrepancies include the infinite-deck dataset/configuration, release split reshuffling, evaluation-based probe selection and position-level cross-validation. The controlled replacements are documented in the audit; comparison values are historical references and do not prove identical historical data or checkpoints.

Changed histories are rank permutations with fixed action/outcome tokens, without simulator replay. They can violate blackjack semantics and are reported as token perturbations. Matched histories are a separate engine-replayed construction from the original 5,000-shoe evaluation pool: 75 pairs, five for each of fifteen expert deviations, 150 globally distinct source shoes and public prefixes. Paired members share target hand/upcard, depth, token length, decision position and common legal actions and straddle the deviation threshold. Composition differs beyond the scalar count, so these pairs do not isolate only Hi-Lo count.

A versioned generator fix rejects reused individual public-prefix hashes before accepting a pair; the old generator checked only duplicate ordered pairs. The deterministic seed, source pool, candidate/replay limit and model-independent selection remain fixed. Local regression checks and real bounded Slurm validation precede production. Prior failed artifacts are preserved and excluded. Scoring compares the exact parsed vocabulary schema and ordered token IDs, with each vocabulary file independently hash-bound to its own dataset; differing JSON formatting is allowed. The preceding 17-second byte-equality guard failure is preserved and counted in usage. Completed V2 generation and corrected V3 scoring have separate exact implementation reviews. EV seed 0 uses a versioned launcher with a 7,800-second ceiling matching its measured smoke recommendation; the gameplay producer, full cohort and fitted directions are unchanged.

Intervention directions are fitted separately for each checkpoint using fitting shoes. EV applies the historical all-context clamp-to-fit-mean hook at block 1 (zero based). Direction normalization alone does not establish matched intervention energy; removed-component magnitude, legality and ordinary-hand damage are reported separately. Offline readback and gameplay propagation are different measurements.

## Completion and compute

Actual allocated use, including failed jobs: **3.1128 GPU-hours** of 11.5 and **13.3789 CPU-hours** of 48. Latest measured combined new storage: **77.89 GiB** of 96. Standard-tier compute fees: **$0**. At most one experiment job and one GPU were used at a time; no arrays, paid priority or automatic retries were used. All job IDs, allocations, wall times and failed attempts are in the ledger and compute-usage CSV. Storage is a measured snapshot; outputs were checked before each stage.

The strict broad-inventory collector may still label unsupported housekeeping adapters and omitted dynamics incomplete. That status is retained, not relabeled as evidence of a run. The retained numerical completion checklist independently requires all nine training/core results, seventeen full EV cohorts, six matched-history scores, three changed-history results and all 35 retained comparison entries. Dataset freeze evidence and actual payload verification are recorded separately. Figure 3 learning trajectories are not rerun or claimed.

## Deliverables and regeneration

- Strict collection: `results/collection_final_headline_20261013/collection_report.json` and its raw-artifact inventory.
- Headline per-seed / mean-SD tables: `tables/headline_results.csv` and `.json`.
- Full scalar and EV tables: `results/collection_final_headline_20261013/aggregate/` (including shoe-cluster and paired uncertainty).
- Paper comparison: `results/collection_final_headline_20261013/paper_comparison_36_rows.json`.
- Rendered figures and provenance: `figures/final_headline_20261013/figure_manifest.json`.
- Regenerate from saved outputs: `python3 -B scripts/regenerate_final_headline.py --output results/NEW_DIRECTORY` (no training, fitting or network).
- Compute: `tables/compute_usage.csv`, `manifests/compute_ledger.json`, immutable `configs/budgets/job_*.json`.
- Scope and corrections: `audit/final_headline_scope_20261013.json`, `audit/paper/paper_inventory.md`, `audit/source_audit.md`, `audit/analysis_design.md`, isolated fix diffs and actual validation reviews.

Use the hash-pinned collection manifest with `scripts/collect_production_results_v5.py collect`, then `scripts/render_replication_figures_headline_v2.py render`; exact arguments/hashes are saved in the final regeneration script. `scripts/finalize_headline_report.py` verifies completeness and ddof=1 before writing this report. Raw runs are never overwritten or substituted with historical values.
