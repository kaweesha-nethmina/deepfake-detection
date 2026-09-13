Model 4 — ViT-B/16 (+ Frequency-Hybrid) — Owner: Member C

What's in this folder







File



Purpose





model.py



ViTDeepfakeDetector — pretrained ViT-B/16 backbone (via timm) with a binary classification head. Optionally fuses a FrequencyBranch (2D FFT → small CNN) before the final layer.





train.py



Trains the model, evaluates on the primary (in-distribution) test set once, then runs the cross-generator evaluation and writes the shared comparison row.





README.md



This file.

Config for this model lives in configs/vit.yaml — never hardcode paths, LR,
batch size, or seed here; change the config instead.

Architecture summary





Backbone: ViT-B/16, patch size 16×16, pretrained on ImageNet-21k, loaded via timm.create_model(..., num_classes=0) so it returns pooled features.



Head: Dropout -> Linear(1) producing a single logit (sigmoid → real/fake probability).



Frequency-hybrid branch (novelty add-on, toggle via model.use_frequency_hybrid):
log-magnitude 2D FFT of the input image → 3 conv blocks → global average pool →
linear projection to fusion_dim. This vector is concatenated with the ViT's
pooled features before the classifier. Rationale: GAN/diffusion upsampling
artifacts often show up more clearly in the frequency domain than in raw pixels.



Fine-tuning: small LR (2e-5–5e-5), linear warmup + linear decay, since ViTs
are more LR-sensitive than CNNs (see configs/vit.yaml → train:).



How to run

# from the repo root
python models/vit/train.py -c configs/vit.yaml

To run the frequency-hybrid ablation, duplicate the config with a new
run_name and flip the flag:

cp configs/vit.yaml configs/vit_freq_hybrid.yaml
# edit vit_freq_hybrid.yaml: run_name -> "vit_b16_freq_hybrid_v1",
#                             model.use_frequency_hybrid -> true
python models/vit/train.py -c configs/vit_freq_hybrid.yaml



Outputs

Everything is written under results/vit/<run_name>/:





checkpoints/best_model.pt — best checkpoint by validation loss



training_history.csv — per-epoch train/val loss and val metrics



metrics.csv — one row: val/test/cross-gen accuracy, precision, recall, F1,
ROC-AUC, accuracy drop, train time, inference time/image, parameter count
(same schema the other 3 models write, so notebook 04 can merge them)



roc_primary_test.csv, roc_cross_gen.csv — ROC curve points



confusion_matrix_primary_test.csv, confusion_matrix_cross_gen.csv



run_config_used.json — exact config snapshot for reproducibility

Shared file (this folder is the only writer):
results/comparison/crossgen_<run_name>.csv — the row this run contributes
to the final 4-model comparison. Each of my runs writes its own file here;
I never hand-edit another model's row.

Cross-generator evaluation harness

train.py evaluates on two held-out sets, in this order, each touched exactly once:





Primary test set (data/test/) — same generator family as training.



Cross-generator test set (data/cross_gen_test/) — a different generator

family, never seen during training or hyperparameter tuning.

The headline number is accuracy_drop = test_acc - cross_gen_acc per model —
this is what notebook 04 charts across all 4 architectures.

Notebook

notebooks/04_vit_crossgen_evaluation.ipynb builds the final all-model
comparison: it reads my metrics.csv plus the other members' metrics.csv
files read-only (I don't edit their folders) and produces the report-ready
tables, ROC overlays, confusion matrix grids, and the accuracy-drop /
efficiency comparison charts.