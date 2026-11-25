# Breast Cancer Tumor Classification

A convolutional neural network (CNN) that classifies microscopic breast biopsy images as **benign** or **malignant**, compared against transfer learning models from TensorFlow Hub. Built as a graduate research project (December 2022).

> For research and educational purposes only. Not for clinical use.

## Dataset

- Kaggle sample of the BreakHis breast cancer histopathology dataset (400X magnification), 700 x 460 px RGB PNG images
- 1,148 training images (371 benign, 777 malignant) and 545 test images (176 benign, 369 malignant)
- 15% of the training images were held out for validation
- Training images were augmented with horizontal flips
- Class imbalance (roughly 2:1 malignant to benign) is a key limitation of the data

The images are not included in this repository. See the Kaggle "Breakhis 400X" dataset and the original BreakHis paper (Spanhol et al., IEEE Transactions on Biomedical Engineering, 2016).

## Approach

**Custom CNN (Keras):** four convolution blocks (32, 64, 128 and 256 filters, 3x3 kernels) with max pooling, dropout (25%), a 64-unit dense layer, and a sigmoid output. Trained with binary cross-entropy and the Adam optimizer, with early stopping and model checkpoints. Batch sizes of 8, 16 and 32 were compared.

**Transfer learning (TensorFlow Hub):** Inception V3, Inception ResNet V2 and EfficientNet V2, all trained with a batch size of 16.

## Results

| Model | Validation accuracy | Test accuracy | Test precision | Test recall |
|---|---|---|---|---|
| Custom CNN (batch size 16) | 89.60% | 86% | 0.88 | 0.91 |
| Inception ResNet V2 | 91.91% | 86% | 0.88 | 0.92 |
| EfficientNet V2 | 86.13% | 84% | 0.85 | 0.93 |
| **Inception V3** | **92.49%** | **88%** | **0.88** | **0.96** |

Inception V3 performed best overall. The custom CNN's validation accuracy fluctuated more between epochs than the pretrained models, and test accuracy was lower than validation accuracy across all models, which suggests some overfitting. Because this is a healthcare problem, recall on malignant cases matters more than raw accuracy. The full discussion is in the report.

## Files

- `breast_cancer.ipynb`: data preparation, model training and evaluation
- `FinalReport.pdf`: full write-up of the methods, results and discussion

## Running the notebook

The notebook was written for Google Colab, with the data stored in Google Drive. To run it elsewhere, download the dataset and update the file paths in the notebook. It uses TensorFlow, TensorFlow Hub, scikit-learn, pandas, NumPy, Matplotlib, Pillow and split-folders.
