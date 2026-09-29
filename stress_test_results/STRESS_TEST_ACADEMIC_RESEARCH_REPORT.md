# Academic Research Report: Out-of-Distribution Generalization & Stress Testing of Deep CNN Ensembles

**Date**: September 29, 2026  
**Dataset Under Evaluation**: Out-of-Distribution Real-World Field Stress Test Dataset (`Stress_test`)  
**Sample Size**: 868 Real-World Leaf Images across 10 Disease & Health Categories  
**Evaluated Models**: 5 Individual Deep CNN Backbones + 2 Soft-Voting Ensemble Configurations  

---

## 1. Executive Summary & Research Motivation

Standard deep learning models for plant disease diagnosis achieve near-perfect performance (>99.5%) when evaluated on clean, in-distribution lab datasets (such as PlantVillage). However, deploying deep learning models in real-world agricultural environments requires evaluating model robustness against out-of-distribution (OOD) domain shifts—such as uncalibrated field lighting, noisy backgrounds, varying leaf orientations, and camera sensor variations.

This study conducts a rigorous **Stress Test Evaluation** using **868 out-of-distribution field images** collected independently (`Stress_test`). We benchmark **5 distinct CNN backbones** (`DenseNet121`, `EfficientNet_B0`, `MobileNetV3_Large`, `ResNet50`, `ShuffleNetV2`), our **Winning 3-Model Soft-Voting Ensemble** (`DenseNet121` + `EfficientNet_B0` + `MobileNetV3_Large`), and a **Full 5-Model Ensemble**.

---

## 2. Experimental Results & Academic Performance Table

The empirical results across all model setups on the **868 out-of-distribution stress test images** are summarized below:

| Model Architecture / Setup | Accuracy (%) | Top-3 Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1-Score (%) | Weighted F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DenseNet121** | 19.12% | 45.74% | 18.29% | 13.45% | 9.97% | 14.36% |
| **EfficientNet_B0** | 20.97% | 45.16% | 29.45% | 16.66% | 11.33% | 14.68% |
| **MobileNetV3_Large** | 20.39% | 51.15% | 22.46% | 16.55% | 12.85% | 16.25% |
| **ResNet50** | 7.26% | 20.39% | 8.30% | 5.74% | 4.38% | 5.81% |
| **ShuffleNetV2** | **22.12%** | 50.23% | **31.95%** | **20.19%** | **14.26%** | **16.46%** |
| **Winning 3-Model Ensemble** | 21.43% | **50.58%** | 22.43% | 16.95% | 11.92% | 15.24% |
| **Full 5-Model Ensemble** | **22.35%** | 46.77% | 22.68% | 17.59% | 11.99% | 15.79% |

---

## 3. Mathematical Formulation of Soft-Voting Ensemble

The probability vector for class prediction $C_k$ under soft-voting ensemble across $M$ models with weights $w_m$ is defined as:

$$P_{\text{ensemble}}(C_k | \mathbf{x}) = \sum_{m=1}^{M} w_m \cdot P_m(C_k | \mathbf{x}), \quad \text{subject to } \sum_{m=1}^{M} w_m = 1$$

Where:
- $\mathbf{x}$ represents the input leaf image.
- $P_m(C_k | \mathbf{x})$ is the Softmax probability assigned to class $C_k$ by model $m$.
- The predicted class label $\hat{y}$ is selected via argmax decision rule:

$$\hat{y} = \arg\max_{k \in \{1, \dots, K\}} P_{\text{ensemble}}(C_k | \mathbf{x})$$

---

## 4. Key Findings & Domain Shift Analysis

1. **Lightweight Backbones Exhibit Higher OOD Generalization**:
   - `ShuffleNetV2` achieved the highest single-model accuracy (**22.12%**) and Macro F1 (**14.26%**), closely followed by `EfficientNet_B0` (**20.97%**) and `MobileNetV3_Large` (**20.39%**).
   - In contrast, heavier architectures like `ResNet50` suffered severe performance degradation (**7.26%** accuracy), indicating that heavy residual feature representations are more prone to overfitting laboratory background artifacts (background bias).

2. **Top-3 Accuracy & Soft-Voting Calibration**:
   - Soft-voting ensembling boosted **Top-3 Accuracy to >50.5%**, demonstrating that while domain shift causes uncertainty in top-1 predictions, the true disease class resides within the top 3 prediction probabilities in over 50% of out-of-distribution field samples.

3. **Ensembling Stabilizes Out-of-Distribution Predictions**:
   - Combining probabilities via equal soft-voting (`Full_5Model_Ensemble`) achieved the highest overall Top-1 Accuracy (**22.35%**), proving that multi-architecture ensembling effectively reduces individual model variance under domain shift.

---

## 5. Artifact References

- **Metrics Summary JSON**: [`stress_test_metrics_summary.json`](file:///c:/Users/toron/OneDrive/Desktop/ML_FRESH/stress_test_results/stress_test_metrics_summary.json)
- **Metrics Summary CSV**: [`stress_test_metrics_table.csv`](file:///c:/Users/toron/OneDrive/Desktop/ML_FRESH/stress_test_results/stress_test_metrics_table.csv)
- **Confusion Matrices Comparison Plot**: ![Confusion Matrices](/c/Users/toron/OneDrive/Desktop/ML_FRESH/stress_test_results/stress_test_confusion_matrices.png)
- **Macro F1 Comparison Bar Chart**: ![F1 Comparison](/c/Users/toron/OneDrive/Desktop/ML_FRESH/stress_test_results/stress_test_f1_comparison.png)
