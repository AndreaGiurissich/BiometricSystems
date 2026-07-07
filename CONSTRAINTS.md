# Project constraints (frozen)

University Biometric Systems project: robustness comparison of fingerprint
recognition models on **SOCOFing** under contrast-mask preprocessing. These
constraints are **frozen** -- do not change them without explicit sign-off.

## Hard rules

- **No training / fine-tuning. Ever.** All models run frozen, inference-only. No
  weights are updated on SOCOFing.
- **Kaggle-only execution.** Free tier: T4 GPU, 4 vCPU, 30 GB RAM, 12 h sessions.
  Dataset at `/kaggle/input/.../SOCOFing` (confirm the slug with
  `scripts/verify_dataset.py` before trusting it). Results in `/kaggle/working/`
  are ephemeral -> checkpoint incrementally; persist via `scripts/save_results.py`.
- **Do not invent results** or fill placeholders with fake numbers.
- **Do not change committed preprocessing params silently** -- surface and ask.
- Distinguish **verified-on-disk** from **written-but-not-run** when reporting
  (there is no Python locally; execution happens on Kaggle).

## Protocol (frozen)

- Closed-set 1:N identification.
- **Identity = each finger** -> 6000 identities (600 subjects x 10 fingers).
- **Gallery** = all 6000 Real images (one template per identity).
- **Probes** = all Altered images, evaluated per level {Easy, Medium, Hard} and
  per alteration type {Obl, CR, Zcut}.
- **Conditions**: `baseline` (raw) vs `preprocessed` (contrast-mask pipeline on
  gallery AND probes, same operator).

## Models (frozen set; unified extract/score interface)

Three models. The **minutiae** model (NBIS, the SourceAFIS stand-in) was
**dropped 2026-06-27**: SOCOFing's ~96x103 px is below the operating resolution
of both NBIS/MINDTCT and deep MinutiaeNet, so extraction is degenerate. See
`docs/minutiae_investigation.md`. Spike scripts are retained for reproducibility.

1. **Gabor** texture descriptor (8 orient x 4 scale, 8x8 grid, cosine).
2. **SIFT + RANSAC** (inlier count). O(N x gallery) -> Tier A on a nested 500 probes.
3. **DINOv2** ViT-B/14 frozen, CLS embedding (768-D), cosine, 224x224.

## Evaluation tiers

- **Tier A (fair cross-model):** all 3 models on a shared, seeded, stratified
  **2000-probe** manifest (SIFT on a nested **500**). Report the SIFT caveat.
- **Tier B (best estimate):** Gabor + DINOv2 on the **full** probe set.

## Metrics

- **Identification (rank-based, cross-model comparable):** Rank-1/5/10, full CMC,
  Rank-N (trivially 1.0 in closed set), MRR.
- **Verification (per-model native scale, never cross-normalized):** ROC, EER,
  AUC, FAR@FRR=1%, FRR@FAR=1%. Genuine = 1/probe; impostors = 100 seeded
  non-self templates/probe.

## Preprocessing

Single contrast-mask pipeline adapted from the Moresca/Suozzi/Santini face paper
(NO facial alignment). **Paper defaults kept verbatim** for the first pass;
montages reviewed before the full matrix; a "fingerprint-scaled" variant is only
added after explicit sign-off.

## Reproducibility

All randomness seeded from `configs/default.yaml`; the exact config is snapshotted
to `run_config.yaml` per run, with lib versions + git commit.

## Implementation order (incremental, smoke-test each)

scaffold + parser + on-Kaggle verification -> ~~NBIS spike~~ (**done -> minutiae
dropped**, see docs/minutiae_investigation.md) -> preprocessing + montage
(**surface montages**) -> evaluation utils -> Gabor end-to-end -> checkpointing +
subsampling -> DINOv2 -> SIFT -> full matrix.
Gate: surface preprocessing montages before moving past them.
