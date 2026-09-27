# Member C: Post-Training Evidence

This folder preserves the user-supplied `vit_wish_s42_ddp2_v1_model 2`
training export and the reports generated from it. Exported Python modules are
an archived runtime snapshot, not replacements for the team's current modules.
The original Downloads folder was left unchanged. macOS metadata and Python
caches are not included.

## Contents

- [Machine-readable results](reports/vit_wish_s42_ddp2_v1/result.json)
- [Report with graphs](reports/vit_wish_s42_ddp2_v1/training_report.html)
- [Markdown report](reports/vit_wish_s42_ddp2_v1/training_report.md)
- [Generated figures](reports/vit_wish_s42_ddp2_v1/figures/)
- [Original training export](training-export/): runtime code, history, config,
  completion record, original learning curves, best weights and Gradio files.
- [Report generator](scripts/create_vit_training_report.py)

## Quality Assessment

This run completed seven epochs. Epoch 1 was selected by minimum validation
loss; six subsequent epochs did not improve it, consistent with patience 6.
Recorded selected-checkpoint validation accuracy is 99.8845%, fake precision
99.8980%, fake recall 99.8471%, F1 0.998725 and ROC-AUC 0.99999348.
Recorded training plus validation time is approximately 59 minutes 58 seconds.

These are strong within-dataset validation results, not proof of generalization.
The Gradio self-test reports 36/36 correct selected validation examples; it is
not an independent final test. Neither primary-test nor cross-generator results
were supplied, and their JSON fields remain null. The manifest/audit is also
absent, so split membership and leakage cannot be independently verified here.
The report uses a best-practice model-card structure, not a proprietary
OpenCodeGen schema.

## Weights and Reproduction

The checkpoint is stored with Git LFS. After cloning:

```sh
git lfs install
git lfs pull
```

The exported checkpoint has the same content as
`artifacts/vit_wish_s42_ddp2_v1/best_model.pt`, retained there for the existing
local Gradio app. Both paths reference the same LFS object:

```text
be78ec4969aaf73dc82918f6c9aa794e6d196287920d1957c511106f458a452c
```

From the repository root, in a Python environment with PyTorch, Matplotlib
and markdown-it-py installed:

```sh
python post-training/scripts/create_vit_training_report.py \
  --source post-training/training-export \
  --output post-training/reports/vit_wish_s42_ddp2_v1
```

This reads the saved evidence and regenerates reports and graphs. It does not
train a model or evaluate test images. All 14 evidence consistency checks must
pass. Metric values are recorded evidence, not independently reproduced
predictions. The checkpoint is loaded with `weights_only=True`.

Before claiming robustness, recover the exact manifest/audit, freeze the
checkpoint and threshold, and evaluate the untouched primary and cross-generator
sets. A fair four-model comparison also requires matching split assignments.
