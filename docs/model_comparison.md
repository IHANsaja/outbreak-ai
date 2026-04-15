# Flood Forecasting Model Comparison Report

This report documents the comparative analysis of three machine learning models for flood forecasting: **Temporal Fusion Transformer (TFT)**, **Long Short-Term Memory (LSTM)**, and **XGBoost**.

## 1. Executive Summary

The evaluation was performed on the **90-rivers dataset**, using an 80/20 chronological split per station. The goal was to determine which architecture provides the most accurate and reliable water level predictions.

| Model | MAE | RMSE | R² Score |
| :--- | :--- | :--- | :--- |
| **TFT** | 0.1245 | 0.1872 | 0.9412 |
| **LSTM** | 0.3049 | 0.6542 | 0.6811 |
| **XGBoost** | 0.7293 | 1.4015 | 0.2714 |

> [!IMPORTANT]
> **TFT** is the clear winner, achieving an R² score of **0.94**, significantly outperforming both LSTM and XGBoost. This confirms that the attention-based architecture effectively captures long-range temporal dependencies and seasonal patterns in river dynamics.

---

## 2. Performance Visualization

### Comparative Metrics
The following charts show a side-by-side comparison of error metrics across all models.

![Leaderboard Metrics](file:///d:/Projects/outbreakAI/outbreak-ai/reports/comparisons/leaderboard_metrics.png)

### The Battle Plot: Models vs. Reality
This chart shows a sample of predictions against actual water levels for a specific station.

![Battle Plot](file:///d:/Projects/outbreakAI/outbreak-ai/reports/comparisons/battle_plot_sample.png)

---

## 3. Analysis of Findings

### Temporal Fusion Transformer (TFT)
- **Strengths**: Highest accuracy across all metrics. Excellent at handling multiple time-series simultaneously and identifying global patterns while respecting local station characteristics.
- **Complexity**: Requires a specialized environment (`pytorch-forecasting`) and significant compute for training.

### Long Short-Term Memory (LSTM)
- **Strengths**: Solid performance with an R² of 0.68. Better than XGBoost at capturing the sequential nature of water level rises.
- **Weaknesses**: Struggled with some high-peak events compared to TFT.

### XGBoost
- **Strengths**: Extremely fast training and inference. Good baseline.
- **Weaknesses**: Lowest accuracy. As a non-sequential model (in this baseline configuration), it relies heavily on lag features and lacks the "memory" inherent in LSTM or TFT architectures.

---

## 4. Next Steps & Recommendations

1. **Deployment**: Pursue TFT as the production model for forecasting.
2. **Optimization**: For edge deployments where compute is limited, a pruned LSTM might serve as a reliable lightweight alternative.
3. **Data Augmentation**: Incorporate more exogenous variables (e.g., dam releases, soil moisture) to further improve the LSTM and XGBoost baselines.
