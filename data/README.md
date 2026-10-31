# Data

No data is stored in this repository. Download the dataset yourself and check its license and terms before use.

## Dataset

**CT KIDNEY DATASET: Normal-Cyst-Tumor and Stone**: 12,446 CT images in four classes. The recorded notebook run read it from Kaggle.

Expected layout (the folder names are the ones used by the notebook):

```
CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/
├── Cyst/      (label 0)
├── Normal/    (label 1)
├── Stone/     (label 2)
└── Tumor/     (label 3)
```

## Preprocessing (as in the notebook)

- Read with OpenCV (BGR channel order), resized to 128 x 128, scaled to [0, 1].
- Shuffled with seed 2, then split 80% / 20% (train / test) at random, without stratification. In the notebook this gave 9,956 training and 2,490 test images.

## Source and citation

Please verify the exact source and license of the copy you use.

- Islam, M. N. et al. *Vision transformer and explainable transfer learning models for auto detection of kidney cyst, stone and tumor from CT-radiography.* Scientific Reports 12, 11440 (2022). This is the publication usually cited for this dataset.
