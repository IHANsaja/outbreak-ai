# Research Summary: Flood Prediction Optimization

## 📅 Project Phase: Training & Evaluation (Current)

The primary objective for this phase is to train high-precision predictive models (TFT, LSTM, XGBoost) using the nationwide hydrological dataset.

### 📋 Key Updates & Documents

- **[Dataset Coverage Report](file:///d:/Projects/outbreakAI/outbreak-ai/docs/dataset_coverage_report.md)**: Documentation of the decision to proceed with **88 stations and 36 rivers**, excluding 2 stations with insufficient data count.
- **[TFT Training Script](file:///d:/Projects/outbreakAI/outbreak-ai/scripts/train_tft.py)**: Revised training pipeline featuring robust normalization and zero-variance protection.
- **[Model Comparison](file:///d:/Projects/outbreakAI/outbreak-ai/docs/model_comparison.md)**: Comparative evaluation of different architectural approaches.

### 📊 Dataset Status (Final Selection)

| Metric | Count |
| :--- | :--- |
| **Total Rows** | ~1,479,819 |
| **Active Rivers** | 36 |
| **Active Stations** | 88 |

---

*Decision: Proceeding with available real-world data (88 stations / 36 rivers).*  
*Last Updated: 2026-04-11*
