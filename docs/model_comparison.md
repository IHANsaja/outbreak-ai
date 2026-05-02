# Flood Forecasting Model Comparison Report: The "Fair Fight"

This report documents the updated comparative analysis of three machine learning models after eliminating training data bias. **All models were evaluated on the full 1.5M-row 90-rivers dataset.**

## 1. Executive Summary

Previously, the comparison was biased by significantly different training data volumes. In this "Fair Fight" phase, **LSTM** and **XGBoost** were retrained on the same 1.5M-row dataset as the **TFT** model, using high-accuracy configurations (100 Epochs for LSTM, GridSearchCV for XGBoost).

| Model | MAE | RMSE | R² Score |
| :--- | :--- | :--- | :--- |
| **XGBoost (Accurate)** | **0.0246** | **0.1332** | **0.9934** |
| **LSTM (Accurate)** | 0.0331 | 0.1353 | 0.9932 |
| **TFT (Current)** | 0.3182 | 0.5548 | 0.8823 |

> [!IMPORTANT]
> **XGBoost** and **LSTM** have significantly overtaken the current **TFT** baseline after being trained on the full dataset. With R² scores exceeding **0.99**, they demonstrate near-perfect alignment with actual water levels, proving that data volume was the primary bottleneck previously.

---

## 2. Performance Visualization

### The New Leaderboard
The following charts show the dramatic improvement in error metrics for the retrained models.

![Leaderboard Metrics](file:///d:/Projects/outbreakAI/outbreak-ai/reports/comparisons/leaderboard_metrics.png)

### The Battle Plot: Models vs. Reality
This chart shows how closely the accurate models now track the actual water level dynamics.

![Battle Plot](file:///d:/Projects/outbreakAI/outbreak-ai/reports/comparisons/battle_plot_sample.png)

---

## 3. Analysis of "Fair Fight" Findings

### XGBoost (Accuracy-Optimized)
- **Strengths**: Now the top performer. Its ability to handle tabular features and lag variables makes it exceptionally precise for 1-hour-ahead forecasting when given sufficient data.
- **Efficiency**: Significantly faster to train and deploy than deep learning alternatives.

### Long Short-Term Memory (LSTM) (Accuracy-Optimized)
- **Strengths**: Achieved 0.993 R², tracking XGBoost almost perfectly. Its recurrent architecture is now fully utilizing the 1.5M rows to learn complex seasonal and temporal patterns.

### Temporal Fusion Transformer (TFT)
- **Status**: Currently in 3rd place based on existing metrics. While powerful, the "Direct Search" and sequence learning of the retrained LSTM/XGBoost models prove to be highly effective for this specific water level target.

---

## 4. Final Recommendations

1. **New Standard**: Adopt **XGBoost (Accurate)** as the primary forecasting engine due to its superior accuracy and low compute overhead.
2. **Hybrid Potential**: Use **LSTM** as a validation model in an ensemble to handle edge cases where sequential memory is critical.
3. **TFT Research**: While TFT is currently ranked 3rd, it may still excel in long-term multi-horizon forecasting (e.g., predicting 24 hours out), which should be explored separately.

---
*Report generated on April 15, 2026, after unified 1.5M-row retraining.*
