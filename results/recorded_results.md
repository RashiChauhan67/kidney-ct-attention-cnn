# Recorded results

Everything on this page is copied from the saved outputs of the original notebook, [`../notebooks/kidney-final.ipynb`](../notebooks/kidney-final.ipynb). **None of it was re-run** for this repository.

Protocol of the recorded run: random 80% / 20% split (seed 2): 9,956 training and 2,490 test images. The 20% split was also used as `validation_data`, with early stopping (patience 10) and best-checkpoint saving on validation accuracy. The notebook then loaded `best_weights-15.keras` and evaluated on the same split. The test accuracy therefore equals the best validation accuracy.

Training ran 25 epochs. Validation accuracy in the last epoch was 0.6137; the best was 0.9896 (epoch 15).

## Test set (2,490 images)

Test loss 0.0289, test accuracy 0.9896.

| Class | Label | Precision | Recall | F1 | Support | AUC |
|---|---|---|---|---|---|---|
| Cyst | 0 | 0.98 | 0.99 | 0.98 | 740 | 0.9996 |
| Normal | 1 | 1.00 | 1.00 | 1.00 | 1036 | 1.0000 |
| Stone | 2 | 1.00 | 0.94 | 0.97 | 272 | 0.9996 |
| Tumor | 3 | 0.98 | 1.00 | 0.99 | 442 | 1.0000 |
| accuracy | | | | 0.99 | 2490 | |
| macro avg | | 0.99 | 0.98 | 0.99 | 2490 | |
| weighted avg | | 0.99 | 0.99 | 0.99 | 2490 | |

Confusion matrix (rows = actual, columns = predicted; Cyst, Normal, Stone, Tumor):

| | Cyst | Normal | Stone | Tumor |
|---|---|---|---|---|
| **Cyst** | 731 | 0 | 0 | 9 |
| **Normal** | 0 | 1036 | 0 | 0 |
| **Stone** | 14 | 3 | 255 | 0 |
| **Tumor** | 0 | 0 | 0 | 442 |

## Model

| Quantity | Value | Source |
|---|---|---|
| Trainable parameters | 414,389 | notebook `model.summary()` |
| Non-trainable parameters | 448 | notebook `model.summary()` |
| Total model parameters | 414,837 | sum of the two |
| "Total params" printed | 1,243,617 | includes Adam optimizer state |
| Model size printed | 1.58 MB | trainable weights only |
| FLOPs printed | 198 M | hand-written estimator that reuses the input shape for every conv; **unreliable** |

## What the paper reports

| Quantity | Paper | Saved notebook |
|---|---|---|
| Accuracy | 99.48% | 98.96% |
| Parameters | 0.22 M | 0.415 M (0.226 M without the Dense 512 layer) |
| Images | 12,446 | 12,446 |

See the README for the other differences between the notebook and the paper.
