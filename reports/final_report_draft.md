# Cross-Generator Generalization in Synthetic Face Detection

**SE4050 report draft, not a completed experimental report.** Real-data training,
test results and team identity fields are pending. Do not submit with pending items.
ViT superiority is a hypothesis, not an established result.

## 1. Introduction and Problem Definition

We investigate binary real-versus-generated face classification and generalization
from StyleGAN to a held-out Stable Diffusion source. The question is whether the
four architectures preserve useful discrimination under generator shift without
excessively misclassifying real controls. This study concerns synthetic face
images, not all video face swaps, audio deepfakes or identity verification.

## 2. Background and Related Work

Explain CNN locality, pretrained transfer learning, patch embeddings/self-attention,
and generator-specific artifacts. Cite original architecture and generator papers.
Do not reproduce unverified numerical/challenge claims from planning guides.
ViT's weaker inductive bias makes pretraining and regularization relevant [2,3].
The optional FFT branch requires a matched ablation; it is not assumed beneficial
and does not replace one of the four core architectures.

## 3. Dataset Description and Exploratory Data Analysis

Use RealVsFake_162k_by_Wish [1]. Add the exact downloaded version, access date,
source provenance and license review. Report actual audit counts by split, label,
FFHQ/CelebA and generator, dimensions, duplicate findings and identity availability.
Include representative training images and image-quality/source distributions.
Never present publisher counts as measured retained training counts.

**Pending evidence:** Member A's approved manifest/audit, source-quality EDA,
license references and preprocessing handoff.

## 4. Data Preprocessing and Feature Engineering

All models use Member A's unchanged approved split membership and common face
crops at 224x224 RGB. Checkpoint-specific normalization is recorded in checkpoint
metadata. Training-only flip, rotation, mild color jitter, JPEG recompression and
blur are shared across models. Validation/test transforms are deterministic.
SHA-256 and decoded-pixel checks reject exact duplicates; dHash candidates require
review. These checks do not guarantee complete identity or semantic deduplication.
Stable Diffusion never appears in training, validation or the primary test.

## 5. Experimental Design

Four core models: Custom CNN, ResNet50, EfficientNetV2 and pretrained ViT-B/16.
Use seed 42, effective batch 32, a 30-epoch cap, validation-loss early stopping
with patience 6, and threshold 0.5. CNN head warmup and partial fine-tuning follow
the saved configs. Optimization differs by architecture and must be disclosed.
All model selection precedes a checkpoint/manifest hash freeze; final primary
and cross-generator evaluations use identical ordered examples across models.
Both tests contain real controls. Source/quality differences can confound a claim
that the generator alone caused a performance change.

**Pending evidence:** Actual run configs, seeds, hardware, package versions,
early-stop epochs, training budgets and frozen protocol hash. Single-seed results
cannot establish training stability; state this unless matched repeat runs finish.

## 6. Model Architectures

- Custom CNN: member-owned convolution/BatchNorm/ReLU blocks, pooling and binary head.
- ResNet50: pretrained residual backbone, member-owned MLP head and partial fine-tuning.
- EfficientNetV2: member-owned efficient convolutional backbone/head and staged tuning.
- ViT-B/16: 16x16 patches at 224x224 (196 patches), pretrained transformer features,
  dropout and one binary logit. BCEWithLogitsLoss avoids a sigmoid inside the model;
  sigmoid is applied for probabilities. AdamW starts at 3e-5 with warmup/decay.

Document the actual architecture code revision, layers, activations, loss, optimizer,
hyperparameters and initialization for every checkpoint. Different ImageNet
pretraining sources prevent attributing every difference solely to architecture.

## 7. Results and Model Comparison

**No real-data results yet.** Populate this section only from completed evaluation
artifacts, never synthetic-test numbers. Include accuracy, precision, recall, F1,
ROC-AUC and confusion matrices for both tests; training/validation learning curves;
total/trainable parameters; training cost; synchronized single-image latency.
Report accuracy drop in percentage points and F1 drop as an absolute difference.
Assess absolute cross-generator performance, not just the smallest gap.

Evidence: `training_history.csv`, `environment.json`, frozen protocol,
`predictions.csv`, `model_comparison.csv`, per-model `results.json` and figures.

## 8. Critical Analysis and Discussion

This section carries 30% of marks. For each conclusion connect a measured result,
a concrete failure example and an appropriately qualified explanation:

- Which model performs best on the unseen source, and with what real-image false positives?
- Do learning curves suggest overfitting, underfitting or unstable optimization?
- Does an apparent gain justify parameter count, GPU time and latency?
- Which high-confidence false positives/negatives involve blur, compression,
  unusual lighting or plausible synthesis? Treat visual explanations as hypotheses.
- Could source dataset, face crop, identity overlap, compression or pretraining
  explain differences? What remains uncontrolled?
- What are the limitations of one held-out generator, one dataset and one seed?
- Which feasible future experiment addresses each limitation without retroactively
  tuning on the final test set?

Avoid universal-detection, fairness or calibrated-confidence claims unsupported
by these experiments. Document AI assistance and independently verify the code
and interpretations used in the submission.

## 9. Conclusion

Pending actual findings. State which model best meets the measured accuracy,
generalization and efficiency objectives. Report a negative ViT/FFT result honestly
if observed. The detector is an experimental aid, not forensic proof of authenticity.

## 10. References

1. Wish096. RealVsFake_162k_by_Wish. https://www.kaggle.com/datasets/wish096/realvsfake-81k-by-wish
2. Dosovitskiy et al. An Image is Worth 16x16 Words. https://arxiv.org/abs/2010.11929
3. Steiner et al. How to train your ViT? https://arxiv.org/abs/2106.10270
4. timm ViT AugReg checkpoint. https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k

Add primary ResNet, EfficientNetV2, FFHQ, CelebA, StyleGAN and diffusion references;
exact weight-card URLs; all borrowed code; and an AI-assistance acknowledgment.

### Contribution and Submission Appendix

Use the genuine commit history and contribution log. The official assignment
requires Members.txt, Report.pdf, Turnitin report, Submission.txt (GitHub and
10-minute YouTube video links), Gradescope code and a leader-ID-named ZIP.
The team must supply identity details, obtain Turnitin and perform final uploads.
