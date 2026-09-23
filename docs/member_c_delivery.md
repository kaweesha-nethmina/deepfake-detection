# Member C Delivery and Viva

## Completion gates

- Code: offline tests and architecture forward checks pass.
- Data: Member A's complete manifest passes the audit and perceptual-candidate review.
- Training: all four real-data runs have histories, environment metadata and validation-selected weights.
- Evaluation: frozen protocol, complete status, all eight model/domain rows and prediction CSVs.
- Analysis: figures regenerated from predictions, actual error examples, explicit limitations.
- Submission: ten-section report, rendered PDF checked, team details, Turnitin, video, links and code package.

The first gate is software verification. It must never be represented as completing
the other gates. The dev merge/GUI integration remain pending approval; no teammate
work was merged, reset or reauthored by the Member C changes.

## Ten-minute demo outline

1. 0:00-1:00: problem, four architectures and generator-shift hypothesis.
2. 1:00-2:30: dataset provenance, source counts, manifest and leakage checks.
3. 2:30-4:00: architectures, loss, pretraining and validation-only selection.
4. 4:00-6:00: measured comparison, confusion matrices and generator gap.
5. 6:00-7:30: run cropped-face inference and regenerate plots from predictions.
6. 7:30-9:00: actual failures, cost trade-offs and limitations.
7. 9:00-10:00: individual contributions, reproducibility and conclusions.

## Viva questions to rehearse

- Why is Stable Diffusion excluded from tuning, and what invalidates that claim?
- Why use a pretrained ViT rather than train one from scratch?
- What do patch size, attention, one logit and BCEWithLogitsLoss mean?
- Why can a high accuracy hide failure on one class?
- Why is a smaller generalization gap not sufficient to choose the best model?
- How can source/identity/compression differences confound generator generalization?
- How are checkpoint normalization and real=0/fake=1 enforced across architectures?
- Which three actual errors can you explain without claiming proof of causality?
- What did you personally change and verify, and what did A/B contribute?

## Required team inputs

Member A: image access/version, corrected shared assignments if needed, crop details,
identity metadata if available, Custom CNN run artifacts and dataset EDA.
Member B: both CNN checkpoints/configs, histories, preprocessing and training provenance.
All members: names, registration numbers, emails, verified contribution statements,
report review and viva practice. Leader: Turnitin, YouTube link, Gradescope/CourseWeb uploads.

Target: data/integration by 25 September; core training/model freeze by 27 September;
final evaluation on 28 September; report/demo on 29 September; submit by 30 September.
These dates depend on prompt data and GPU access. No fabricated/backdated commits.
