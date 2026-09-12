# Kamuplahe - Camouflaged Object Segmentation

Binary segmentation of camouflaged objects in natural images: given an RGB photo, predict a
pixel-level mask of the camouflaged object (animal or otherwise) within it.

![predictions](docs/images/res.png)


## Applications

Binary camouflage segmentation has applications across fields where objects blend into their surroundings, including ecological monitoring, search and rescue, medical imaging, industrial inspection, defense and surveillance research, and agricultural pest detection. These applications involve detecting targets that are difficult to distinguish from complex backgrounds due to similar color, texture, or patterns.

## Dataset

Kaggle "Camouflaged-Object-Detection" (COD10K + CAMO training pool, evaluated on the CAMO,
CHAMELEON, COD10K, and NC4K benchmarks). 11,000 matched image/mask training pairs (9,900
train / 1,100 validation), 6,473 test pairs across the four benchmarks (CAMO 250, CHAMELEON 76,
COD10K 2,026, NC4K 4,121). Camouflaged objects cover a mean of 5.7% of image area, motivating
the loss weighting described below.

## Method

Two architectures trained and compared under identical conditions, with the winner selected by
validation performance rather than assumed in advance:

- **Baseline** - U-Net, ResNet18 encoder (ImageNet-pretrained).
- **Main model** - SegFormer, MiT-B1 encoder (ImageNet-pretrained).

Both trained at 288×288 with AdamW, a cosine LR schedule, mixed precision, and early stopping
(patience 5 on validation Dice). Loss is BCE + Dice, with BCE's `pos_weight` set from the
measured foreground/background ratio (5.7% → weight ≈16.4), so a missed object pixel costs
proportionally more than a missed background pixel:

```python
class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return self.bce(logits, targets) + self.dice(logits, targets)
```

Dataset pairing is done by filename stem rather than assumed folder-to-folder correspondence,
since Kaggle repackagings of this benchmark vary in structure and occasionally include masks
with no matching image (250 such orphans were found and dropped in this run):

```python
def pair_images_and_masks(base_dir, label=""):
    img_dir = max(find_dir_by_name(base_dir, IMG_DIR_NAMES), key=lambda d: len(os.listdir(d)))
    mask_dir = max(find_dir_by_name(base_dir, MASK_DIR_NAMES), key=lambda d: len(os.listdir(d)))
    img_files = {Path(f).stem: os.path.join(img_dir, f) for f in os.listdir(img_dir)
                 if f.lower().endswith(IMAGE_EXTS)}
    mask_files = {Path(f).stem: os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                  if f.lower().endswith(IMAGE_EXTS)}
    common = sorted(set(img_files) & set(mask_files))
    return [(img_files[k], mask_files[k]) for k in common]
```

At inference, predictions are averaged with their horizontal-flip counterpart (test-time
augmentation), and each model's decision threshold is tuned on validation Dice independently
rather than fixed at 0.5 for both:

```python
@torch.no_grad()
def predict_logits(model, images, tta=True):
    logits = model(images)
    if tta:
        flipped = model(torch.flip(images, dims=[3]))
        logits = (logits + torch.flip(flipped, dims=[3])) / 2
    return logits
```

## Results

Validation, both models at their own tuned threshold:

| Model | Threshold | IoU | Dice | Precision | Recall | MAE |
|---|---:|---:|---:|---:|---:|---:|
| U-Net (ResNet18) | 0.70 | 0.580 | 0.648 | 0.668 | 0.833 | 0.076 |
| SegFormer (MiT-B1) | 0.70 | 0.567 | 0.626 | 0.734 | 0.744 | 0.088 |

U-Net was selected for final evaluation. SegFormer's validation Dice peaked at epoch 7 and
triggered early stopping at epoch 12/18; U-Net trained the full 18 epochs and was still
improving at the end.

Test set, by benchmark (U-Net, TTA, threshold 0.70):

| Benchmark | n | IoU | Dice | Precision | Recall | MAE |
|---|---:|---:|---:|---:|---:|---:|
| CAMO | 250 | 0.469 | 0.584 | 0.680 | 0.682 | 0.161 |
| CHAMELEON | 76 | 0.602 | 0.726 | 0.640 | 0.888 | 0.109 |
| COD10K | 2,026 | 0.509 | 0.630 | 0.583 | 0.825 | 0.097 |
| NC4K | 4,121 | 0.522 | 0.638 | 0.699 | 0.735 | 0.121 |
| **Pooled** | 6,473 | **0.517** | **0.634** | 0.661 | 0.763 | 0.115 |

CAMO is consistently the hardest of the four benchmarks; CHAMELEON the easiest. Correlation
between object-coverage fraction and per-image IoU is weak (r=0.30) object size is not the
dominant factor in segmentation difficulty. The hardest failures at test time are concentrated
in a small set of natural camouflage specialists (cephalopods, leaf-mimicking insects) rather
than being spread evenly across the dataset.

## Known limitation

MAE (0.115) is higher here than in an earlier, unweighted-loss configuration (0.095), despite
higher IoU/Dice. The `pos_weight` correction trades calibration for detection sensitivity
worth a follow-up sweep over weight values before treating the current setting as final.

