# Backup: Optimizing TFT Flood Prediction Model
*Originally generated on: 2026-04-03*
*ID: 881886c4-7831-4ba7-a105-f06c92b3e502*

## Overview
This document outlines the hydrological structures and mathematical strategies used to prepare the "Research-Grade" 90-station flood forecasting dataset for the Transformer (TFT) model.

## 1. River vs. Station Scenarios
In hydrology, a **River** is the entire body of water (e.g., Mahaweli Ganga), whereas a **Station** is a specific sensor location along that river (e.g., Kandy, Nawalapitiya, Peradeniya).

*   **The Structure**: The dataset contains **37 unique River IDs** representing **90 individual Stations**.
*   **Significance**: Recovers information for all 90 stations, enabling the AI to see the "flow" of water move downstream and learn travel times.

## 2. Data Augmentation (Active Data Repair)
To increase usable data from 30 to 90 stations, the following was implemented:

*   **Hourly Gridding**: Filled sensor gaps using **linear interpolation**.
*   **Atmospheric Rainfall Imputation**: Borrowed average rainfall from other stations when specific sensors failed.
*   **Geological Threshold Imputation**: Assigned missing flood levels based on **85% of historical maximums**.

## 3. Why the Temporal Fusion Transformer (TFT)?
*   **Temporal Continuity**: Uses Position Encodings to understand temporal sequences.
*   **Multi-Station Attention**: Can "attend" to all 90 stations simultaneously to detect relationships across the geography.
*   **Static Covariates**: Learns the unique "personalities" of different rivers.

## 4. Advanced Capabilities
1.  **Explainable AI**: Variable importance analysis for flood science.
2.  **Quantile Forecasting**: Predicting risk probabilities (e.g., "90% confidence water stays below 3m").
3.  **Multi-Horizon Prediction**: Forecasts for 1h, 6h, and 24h horizons simultaneously.

## 5. Stations Inventory (Selected Rivers)
- **Mahaweli Ganga**: Agra Oya, Ginigathena, Holombuwa, Nawalapitiya, Peradeniya, Weragantota
- **Kelani Ganga**: Deraniyagala, Glencourse, Hanwella, Holombuwa, Kitulgala, Nagalagam Street
- **Kalu Ganga**: Ellagawa, Millakanda, Putupaula, Ratnapura
- **Gin Ganga**: Baddegama, Tawalama
- **Nilwala Ganga**: Panadugama, Pitabeddara, Urawa, Urubokka Ganga

---
*End of Backup.*
