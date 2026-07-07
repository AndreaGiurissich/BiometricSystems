# SOCOFing Robustness Study

Closed-set 1:N fingerprint identification on **SOCOFing**, comparing three frozen
recognition models and measuring how a contrast-mask preprocessing pipeline
(adapted from a face-detection robustness paper) affects performance as
alteration difficulty increases.

**No training.** Every model runs in frozen / deterministic inference mode; no
weights are updated on SOCOFing. This is a pure evaluation study.

## Protocol (frozen)

- **Task:** closed-set 1:N identification.
- **Identity = each finger** -> 6000 identities (600 subjects x 10 fingers).
- **Gallery:** all 6000 Real images (one template per identity).
- **Probes:** all Altered images, evaluated separately per difficulty level
  (Easy / Medium / Hard) and per alteration type (Obl / CR / Zcut).
- **Conditions:** `baseline` (raw) and `preprocessed` (contrast-mask pipeline
  applied to gallery *and* probes with the same operator).

## Models (all frozen)

| Model | Representation | Score |
|-------|----------------|-------|
| Gabor texture | filter-bank feature vector | cosine |
| SIFT + RANSAC | keypoints/descriptors | RANSAC inlier count |
| DINOv2 ViT-B/14 | CLS embedding (768-D) | cosine |

> **Minutiae model dropped (2026-06-27).** A minutiae baseline was investigated
> (NBIS/MINDTCT+BOZORTH3 as the SourceAFIS stand-in, then deep MinutiaeNet via
> `fingerflow`) but excluded: SOCOFing's ~96x103 px images are below the
> operating resolution of both, giving degenerate templates (native extraction
> yields 0–6 minutiae). The full analysis, the upsampling experiment, and a
> report-ready write-up are in **`docs/minutiae_investigation.md`**. The spike
> scripts (`scripts/spike_nbis.py`, `scripts/spike_minutiaenet.py`) are kept for
> reproducibility.

## Evaluation tiers

- **Tier A (fair cross-model):** all 3 models on a shared, stratified, seeded
  **2000-probe** manifest. SIFT runs on a **nested 500-probe** subset (it is
  O(N x gallery)); its Tier A numbers carry that caveat in the report.
- **Tier B (best estimate):** Gabor + DINOv2 on the **full** probe set.

## Metrics

- **Identification (rank-based, cross-model comparable):** Rank-1/5/10,
  full CMC, Rank-N (trivially 1.0 in closed set), MRR.
- **Verification (per-model native score scale, never cross-normalized):**
  ROC, EER, AUC, FAR@FRR=1%, FRR@FAR=1%. Genuine = 1 pair/probe; impostors =
  100 seeded random non-self templates/probe.

## Running on Kaggle (the only supported environment)

Kaggle free tier: T4 GPU, 4 vCPU, 30 GB RAM, 12 h sessions. In a notebook:

```bash
!git clone https://github.com/<user>/socofing-robustness.git
%cd socofing-robustness
!pip install -q -r requirements.txt
```

1. **Enable "Internet"** in the notebook settings (DINOv2 weights are pulled via
   `torch.hub`).
2. **Attach the SOCOFing dataset.** Confirm the real path first:
   ```bash
   !python scripts/verify_dataset.py --input-root /kaggle/input
   ```
   This walks `/kaggle/input` (2 levels), confirms the 6000-identity gallery,
   reports per-level / per-alteration counts, orphans, and a suffix casing
   histogram. Override `dataset_root` in `configs/default.yaml` if the slug
   nests differently.
3. Run experiments from the CLI in `src/` (added incrementally).

`/kaggle/working/` is ephemeral: after a run, save results with
`scripts/save_results.py` (zips + timestamps `results/`) and download them, or
commit them back to GitHub manually (a stored PAT via Kaggle Secrets). No
automatic git push is performed.

## Reproducibility

All randomness (impostor sampling, probe subsampling, RANSAC) is seeded from the
config. The exact config used is snapshotted to `run_config.yaml` next to each
run's results, alongside library versions and the git commit.

## Repository layout

```
configs/      default.yaml  (single source of truth for all parameters)
src/          dataset, preprocessing, models/, evaluation, pipeline
scripts/      verify_dataset.py, spike_nbis.py, spike_minutiaenet.py, save_results.py, ...
docs/         minutiae_investigation.md  (why the minutiae model was dropped)
tests/        synthetic-fixture unit tests (no real data needed)
results/      raw/  (per-experiment CSVs)  figures/  summary.csv   [gitignored]
notebooks/    run_experiments.ipynb  (thin launcher; heavy logic stays in src/)
```

## Local development

Unit tests need no data: `python -m pytest tests/ -v`.

To run the **full pipeline locally**, download SOCOFing to `data/socofing/SOCOFing/`
(so `data/socofing/SOCOFing/Real/` exists) and use the `local` profile:

```bash
# one command: all models x levels x conditions -> tables + figures + significance
SOCOFING_PROFILE=local python scripts/run_all.py \
    --dataset-root data/socofing/SOCOFing --workers 4
# no GPU? skip DINOv2:  --models gabor,sift
```

`scripts/run_all.py` reuses each model across levels, skips empty levels and
unbuildable models (e.g. DINOv2 without torch), and then runs `synthesize.py`
(`results/summary.csv` + figures) and `significance.py` (`results/significance.csv`).
The same steps, as a notebook, are in `notebooks/run_experiments.ipynb` (works on
Kaggle and locally). Results land under `results/` (gitignored); bundle them with
`scripts/save_results.py`.
