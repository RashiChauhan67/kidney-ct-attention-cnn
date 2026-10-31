# Data

No data is stored in this repository. Download the dataset yourself and check its license and terms before use.

## Dataset

**CT KIDNEY DATASET: Normal-Cyst-Tumor and Stone**: 12,446 CT images in four classes, as described in the paper: 3,709 cysts, 5,077 normal kidneys, 1,377 stones and 2,283 tumors. The notebook read it from Kaggle (dataset slug `ct-kidney-dataset-normal-cyst-tumor-and-stone`).

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

The paper cites this dataset as:

- Islam, M. N., Hasan, M., Hossain, M. K., Alam, M. G. R., Uddin, M. Z., Soylu, A. *Vision transformer and explainable transfer learning models for auto detection of kidney cyst, stone and tumor from CT-radiography.* Scientific Reports (2022).

Check the license shown on the Kaggle dataset page for your copy before reuse, and cite the publication above when you publish results.
