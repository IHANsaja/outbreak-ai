# Dataset Coverage & Station Exclusion Report

## 📋 Overview
This report documents the final decision regarding the spatial coverage of the nationwide flood prediction model using the `water_levels_90_rivers_ready.csv` dataset.

During the preprocessing for time-series modeling (TFT and LSTM), which requires sequence data (typically >12 entries), certain monitoring points were identified as having insufficient historical depth.

## 🚨 Final Exclusion List
The following **1 river** and **2 stations** were excluded from the final training pipeline because they contain only **exactly 1 observation** each in the source dataset.

### Missing River
- **Mee Oya (RB 95)** [River ID: 44]
  - *Reason*: This river contains only one gauging station (ID: 54), which did not have enough historical observations to form a valid time-series sequence.

### Missing Stations
- **Mee Oya** [Station ID: 54] (Basin: Mee Oya)
  - *Data Count*: 1 row.
- **Panadugama** [Station ID: 66] (Basin: Nilwala Ganga)
  - *Data Count*: 1 row.
  - *Note*: Other stations on the Nilwala Ganga (IDs 63, 68, 70, 81, 82) remain active in the dataset.

---

## 🏛️ Strategic Decision: Continue with Available Data
After evaluating synthetic augmentation (Temporal Padding), the project decision is to **proceed with the currently available high-fidelity data**. 

- **Policy**: No synthetic data or "padded" rows will be used to artificially inflate the station count.
- **Objective**: Maintain the highest degree of empirical accuracy for the research paper and model performance.

### Final Model Scope
| Metric | Original Count | Final Model Scope | Coverage % |
| :--- | :--- | :--- | :--- |
| **Rivers** | 37 | **36** | 97.3% |
| **Stations** | 90 | **88** | 97.8% |

---
*Last Updated: 2026-04-11*  
*Status: Approved for Training (88 Stations / 36 Rivers)*
