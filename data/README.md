# Shared Dataset Contract

Dataset: [RealVsFake_162k_by_Wish](https://www.kaggle.com/datasets/wish096/realvsfake-81k-by-wish).
Use the same Kaggle dataset version as Member A. The owner reports 81,000 real,
70,000 StyleGAN, 9,971 Stable Diffusion and 1,001 other AI images. These are
publisher counts, not verified local counts. Record original FFHQ/CelebA provenance
and applicable source terms; the aggregator's CC0 label alone is not a complete
license audit. Cite source datasets and pretrained weights in the report.

Images are described as already-cropped 299x299 faces. Do not independently
recrop ViT data. All models use the shared crops resized to 224x224 with their
checkpoint's normalization. Source is encoded in basenames, NOT generator folders:

| Prefix | Source column | Label | Allowed split |
|---|---|---:|---|
| RFF | FFHQ | 0 | train, val, test, cross_gen |
| RCA | CelebA | 0 | train, val, test, cross_gen |
| FSG | StyleGAN | 1 | train, val, test |
| FSD | StableDiffusion | 1 | cross_gen only |
| AI | AiGenImage | 1 | excluded only |

Member A supplies a UTF-8 CSV containing `filepath,label,source,split` and optional
`identity`. Keep original names such as `Fake/FSG (123).jpg`. Paths must be
relative to the image root, not machine-specific Kaggle paths. If A has four CSVs,
combine rows with their existing assignments; never run a new random split.

Every active split must contain real and fake examples. Use distinct real controls
for the two tests. Do not drain the primary real test set to balance the cross test.
Inspect balance/source composition in audit.json; balance is reported, not silently
repaired. SD in training, missing real controls, duplicates or wrong labels require
Member A to correct one manifest for the entire team BEFORE any final experiment.

See [the runnable audit/training workflow](../models/vit/README.md). It adds content
and pixel hashes and dHash review candidates without changing split assignments.
Exact duplicates are rejected even within a split to avoid inflated sample counts.
Identity IDs are source-qualified; when metadata is unavailable, document that
identity-disjointness across FFHQ/CelebA cannot be established by filename/hash checks.

Raw images, extracted archives, audited manifests, review files and checkpoints
stay out of Git. Share them through the team's approved storage. Never commit
Kaggle API credentials. The manifests are portable across Mac and GPU mounts.
