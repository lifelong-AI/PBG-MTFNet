# Reproducibility and snapshot notes

## What is included

The `experiments` directory contains byte-for-byte copies of the 97 Python files recovered from the original experiment archive. Folder names and source behavior are preserved. README text is adapted from the author's PBG-MTFNet research project PDF and checked against the main `F5T35` implementation. Figures are extracted originals, not reconstructed diagrams.

Only project source code, documentation, and architecture figures are included. Third-party datasets, samples, labels, model weights, caches, and historical result files are excluded. Data-loading modules are source code; they do not contain or distribute the datasets.

## Items to review before a new training run

These observations apply to the inspected main `F5T35` scripts. They are preserved here rather than silently changing the historical experiments.

1. **Classifier initialization:** the MLP head is created lazily in the first forward pass. The training script creates its optimizer before that pass, so the newly created head parameters are absent from the optimizer. Materialize the head before optimizer construction in a corrected training version.
2. **Tensor ordering:** the per-band embedding is shaped `(B, T, N, C)`, then permuted to `(B, T, C, N)` before reshaping to `(B*T, N, C)`. This changes the node/channel ordering. Verify the intended layout before interpreting graph features or retraining.
3. **Graph normalization:** the loader adds self-loops and normalizes adjacency, and the main model applies its own normalization by default. Verify whether this second normalization is intended.
4. **Evaluation protocol:** main mixed-subject scripts randomly split pooled samples, not subjects. Their loaders compute per-subject normalization using all loaded samples before the split. This uses validation/test feature statistics; it should not be described as train-only normalization or unseen-subject evaluation. Inspect the separate LOSO scripts and preprocessing protocol before making cross-subject claims.
5. **Portability:** `ROOT` values are historical paths and need configuration. Each experiment is a standalone script collection with local imports; it is not an installable Python package.

Correcting these issues would change experiment behavior and requires retraining and reevaluation before reporting updated results. Existing weights should not be assumed equivalent to a corrected implementation.

## Validation scope

Repository preparation uses static Python parsing, local module/symbol checks, checks of Markdown links and images, and source-copy integrity checks. No training, inference, or numerical reproduction has been performed in the preparation environment, which does not include PyTorch or the datasets.

All 97 files passed syntax parsing and byte-for-byte source checks. Static import inspection found one unresolved historical import: `experiments/seed_model/LOSO/train.py` imports `compute_stats_crossband`, which its adjacent `data_loader.py` does not define. That alternative entry point requires repair before use. The `LOSO.py` entry point does not have this missing-symbol import; neither entry point has been runtime-validated here. All README image and document links resolve locally.

The dependency list records imports required by the archived scripts. Only the PyTorch version is pinned from the report; it is not an independently verified lockfile.
