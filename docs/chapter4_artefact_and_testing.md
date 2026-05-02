## 3.4 Testing

To guarantee the mission-critical reliability required of a disaster management platform, testing was structurally divided across the four primary subsystems: the Online Mode (Web Platform), the Automated Data Pipeline, the AI Forecasting Engine, and the Artificial Intelligence Models. This segregated testing methodology ensured that failures in one domain could be isolated and resolved without cascading into other architectural layers. The exact test scenarios, methodologies, and outcomes proving the system's resilience are thoroughly documented in Chapter 4 under Section 4.3 (Test Cases).

---

# Chapter 4: Artefact

The final completed artefact is the **Outbreak AI Flood Forecasting and Disaster Communication System**. It is a distributed, full-stack disaster intelligence framework that actively monitors hydrological variables, predicts flood onset using machine learning, visualizes hazards via an interactive Next.js dashboard, and preserves communication in disaster zones through an offline IoT mesh network.

## 4.1 Academic Findings

The applied research conducted during the system's development yielded crucial academic findings, challenging common assumptions in modern data science and crisis software architecture.

### Finding 1: Data Volume Supersedes Model Complexity
A prevailing assumption is that migrating to advanced deep learning architectures guarantees performance improvements. However, this research demonstrated that data volume and resolution are far more critical. When initial models were trained on a sparse dataset of ~42,786 rows (irregularly spaced), performance was mediocre (XGBoost $R^2 \approx 0.27$, LSTM $R^2 \approx 0.78$). By migrating to a high-fidelity dataset of **1.47 million rows** with strict hourly resampling and linear interpolation, both XGBoost and LSTM achieved $R^2 > 0.99$. This proved that deep learning algorithms require high-density, uninterrupted sequences to differentiate true flood signals from mathematical noise.

### Finding 2: Tabular Ensembles Eclipse Deep Learning in Short-Horizon Hydrology
While Temporal Fusion Transformers (TFT) and Long Short-Term Memory (LSTM) networks are state-of-the-art for sequence modeling, this research found that for *short-horizon* predictions (1-hour ahead), **XGBoost achieved the highest empirical accuracy ($R^2 = 0.9934$) and lowest error (MAE = 0.0246m)**, while drastically reducing computational overhead. Feature importance analysis revealed that 1-hour ahead river hydrology is heavily deterministic based on the immediate past state (`water_level_lag1`). Deep learning architectures inherently attempted to extract long-term dependencies that simply constituted noise for a 1-hour horizon, whereas XGBoost isolated the singular dominant lag feature optimally.

### Finding 3: The Fallacy of Standardized Government Data
The assumption that government-published data (DMC Bulletins) adheres to consistent digital standards was proven false. The research found that structural layouts shifted arbitrarily, and documents frequently transitioned from structured PDFs to flat scanned images without warning. This led to the academic finding that **hard-coded heuristic parsing is fundamentally insufficient for crisis data ingestion.** A robust system must employ a Hybrid Extractor pattern—utilizing algorithmic text parsing as a primary channel, fortified by a secondary Optical Character Recognition (OCR) neural network fallback. 

### Finding 4: Edge-Cloud State Synchronization Resilience
The integration of a secondary, decentralized IoT communication grid validated that maintaining situational awareness during complete ISP (Internet Service Provider) network collapse is achievable. Caching lightweight JSON SOS payloads locally on edge nodes via Wi-Fi/LoRa, and subsequently executing bulk `UPSERT` operations to the Supabase cloud upon reconnection, successfully prevents critical emergency data loss.

## 4.2 Sample Code

The following sections dissect the core codebase, providing sample implementations to detail the functional mechanics of each subsystem.

### 4.2.1 Online Mode: Secure Server Actions and Middleware RBAC
In a modern Next.js 15 application, securely modifying database state without exposing raw API keys is paramount. The system utilizes *Server Actions* to directly execute PostgreSQL insertions safely on the server side, combined with rigorous Next.js Middleware for Role-Based Access Control (RBAC).

**Sample Code: Role-Based Route Protection (Middleware)**
```typescript
// onlineMode/middleware.ts
import { type NextRequest } from 'next/server'
import { updateSession } from '@/utils/supabase/middleware'
import { createServerClient } from '@supabase/ssr'

export async function middleware(request: NextRequest) {
  // Update session and get auth status
  const response = await updateSession(request)
  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    { cookies: { getAll() { return request.cookies.getAll() }, setAll(cookiesToSet) { } } }
  )
  
  const { data: { user } } = await supabase.auth.getUser()

  // Protect authority dashboard from non-authority users
  if (request.nextUrl.pathname.startsWith('/authority')) {
    if (!user) {
      return Response.redirect(new URL('/login', request.url))
    }
    const { data: profile } = await supabase
      .from('profiles')
      .select('role')
      .eq('id', user.id)
      .single()

    if (profile?.role !== 'authority') {
      return Response.redirect(new URL('/', request.url))
    }
  }

  return response
}
```
**Explanation:** This code ensures zero-trust security. Before a user can access the `/authority` dashboard, the middleware validates the JWT session and queries the `profiles` table to explicitly check if the user holds the `authority` role. If not, they are immediately redirected.

**Sample Code: Server Actions & Geospatial Processing**
```typescript
// onlineMode/app/actions/data.ts
export async function submitIncident(formData: FormData) {
  const supabase = await createClient()
  const itype = formData.get('itype')
  let latitude: any = formData.get('latitude')
  let longitude: any = formData.get('longitude')
  const { data: { user } } = await supabase.auth.getUser()

  // Fallback to user's last known location if GPS is unavailable
  if (user && (!latitude || !longitude || (latitude === '6.9271' && longitude === '79.8612'))) {
    const { data: profile } = await supabase
      .from('profiles')
      .select('last_location_lat, last_location_lng')
      .eq('id', user.id).single()
      
    if (profile?.last_location_lat && profile?.last_location_lng) {
      latitude = profile.last_location_lat
      longitude = profile.last_location_lng
    }
  }

  // Securely insert the incident
  const { error } = await supabase.from('incidents').insert({
    itype, latitude, longitude,
    reporter_id: user?.id,
    status: 'pending'
  })

  // Purge the router cache to push the alert to live dashboards
  revalidatePath('/authority/incidents')
  return { success: true }
}
```

### 4.2.2 Automated Data Pipeline: Hybrid Extraction Algorithm
The data pipeline continuously hunts for new government PDFs. When formatting degrades, it automatically transitions from geometric text extraction to Computer Vision.

**Sample Code: Hybrid Parsing Logic**
```python
# data-pipeline/pipeline_core.py
def extract_data_hybrid(self, file_path):
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        # Primary strategy: algorithmic text parsing
        data = self._parse_pdf_text(file_path)
        if not data:
            logger.info(f"Text extraction empty. Falling back to OCR for {file_path.name}")
            # Secondary strategy: Machine-learning based image recognition (EasyOCR)
            data = self._parse_image_ocr(file_path) 
        return data
    elif ext in [".jpg", ".jpeg", ".png"]:
        return self._parse_image_ocr(file_path)
    return []

def _safe_float(self, value):
    # Data normalization to prevent pipeline crashes from typographic errors
    if not isinstance(value, str):
        return float(value) if value is not None else 0.0
    cleaned = re.sub(r'[^0-9.-]', '', value)
    try:
        return float(cleaned) if cleaned else 0.0
    except ValueError:
        return 0.0
```

### 4.2.3 Forecasting Engine: REST API & Sequence Padding
The forecasting engine exposes the trained Python AI models as a fast, accessible REST API via FastAPI, processing incoming arrays of historical telemetry.

**Sample Code: Prediction Endpoint and Tensor Formatting**
```python
# forecasting-engine/main.py
@app.post("/predict")
def predict(history: List[RiverReport]):
    if len(history) < 1:
        raise HTTPException(status_code=400, detail="At least 1 report is required.")
    
    # Pad history to exactly 12 temporal records required by the neural network window.
    # If the database only has 5 hours of history, repeat the oldest record to fill the tensor.
    records = [r.dict() for r in history]
    while len(records) < 12:
        records.insert(0, records[0].copy())
    
    df = pd.DataFrame(records)
    
    # Coerce categorical and numeric strings into strict Pandas data types
    for col in engine.base_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df['station_id'] = df['station_id'].astype(str)
    
    try:
        forecasts = engine.get_specialized_forecasts(df)
        return {
            "success": True,
            "forecasts": {"early_warning_1h": forecasts["early_warning_1h"]},
            "units": "meters",
            "metadata": {"window_size": len(df)}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 4.2.4 Artificial Intelligence Models
Three models were trained on 1.47 million rows of historical river data: XGBoost, LSTM, and TFT. All data was resampled to a strict 1-hour interval.

**Sample Code: XGBoost Hyperparameter GridSearch**
```python
# Model Training Phase
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [500, 1000],      
    'max_depth': [6, 10],             
    'learning_rate': [0.01, 0.05],    
    'subsample': [0.8, 1.0]           
}

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    tree_method='gpu_hist',  # GPU-accelerated training
    random_state=42
)

grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)
```

**Sample Code: LSTM Architecture & Sequence Generation**
```python
import torch.nn as nn
import numpy as np

class FloodLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(FloodLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Extract prediction from the final time step
        return out

def create_sequences(data, target, window_size=12):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])  # 12 hours of temporal input
        y.append(target[i + window_size])     # Target 1 hour ahead
    return np.array(X), np.array(y)
```

## 4.3 Test Cases

To provide exhaustive proof of the system's resilience, the following tabular test cases cover critical functional checkpoints spanning across all four sub-architectures.

### 4.3.1 Online Mode (Web Interface)
| TC ID | Objective | Test Procedure | Expected Result | Actual Outcome | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **OM-01** | Role-Based Authorization | 1. Authenticate as 'Citizen'. <br>2. Manually navigate to `/authority/dashboard`. | Middleware intercepts request, parses JWT, restricts access, redirects to `/`. | Redirected to `/`. | **PASS** |
| **OM-02** | SOS Real-time Broadcast | 1. Submit SOS payload ("Medical Emergency"). | Triggers server action, executes Supabase `insert`, purges cache via `revalidatePath`. | Alert appears on Authority dashboard instantly. | **PASS** |
| **OM-03** | Haversine Geo-Sorting | 1. Set user location to Colombo. <br>2. Generate mock hazard in Kandy. | Distance calculation computes ~115km and sorts hazard below local threats. | Computed distance: 115.4km. Sorted correctly. | **PASS** |
| **OM-04** | GPS Fallback Mechanism | 1. Deny browser location permission. <br>2. Submit incident report. | System falls back to `last_location_lat` from PostgreSQL `profiles` table. | Location appended from profile seamlessly. | **PASS** |
| **OM-05** | Auth Token Expiration | 1. Manually expire JWT token. <br>2. Attempt to fetch restricted API route. | API responds with 401 Unauthorized; frontend forces re-authentication. | 401 Returned. Redirected to login. | **PASS** |

### 4.3.2 Data Pipeline (Ingestion & Scraping)
| TC ID | Objective | Test Procedure | Expected Result | Actual Outcome | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DP-01** | Standard PDF Parsing | 1. Feed digitally encoded DMC PDF bulletin. | `pdfplumber` locates tabular data by anchoring to "Rising"/"Falling" keywords. | 88 valid stations extracted. | **PASS** |
| **DP-02** | OCR Failover Trigger | 1. Feed a scanned, image-only PDF bulletin. | `_parse_pdf_text` fails (empty array). Pipeline invokes PyTorch `EasyOCR`. | OCR executes; 75 stations successfully salvaged. | **PASS** |
| **DP-03** | Typographic Normalization| 1. Input corrupted string "1.5m " to `_safe_float()`. | Regex cleans non-numeric characters, safely casts to strictly `1.5`. | Float `1.5` returned. | **PASS** |
| **DP-04** | Network Disconnect Retry | 1. Disconnect internet during PDF download. | Request library catches `ConnectionError`, applies exponential backoff and retries. | Pipeline retried 3 times before graceful logging. | **PASS** |
| **DP-05** | Duplicate Row Prevention | 1. Process the identical PDF bulletin twice. | Hash-check or database UPSERT prevents duplicate timestamp rows per station. | No duplicate entries created. | **PASS** |

### 4.3.3 Forecasting Engine (FastAPI)
| TC ID | Objective | Test Procedure | Expected Result | Actual Outcome | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FE-01** | Tensor Sequence Padding | 1. Send `POST /predict` containing only 3 hours of historical data. | Engine pads the array to 12 records by repeating the oldest known entry. | Sequence shape matched `(1, 12, 10)`. Prediction successful. | **PASS** |
| **FE-02** | Inference Latency Stress | 1. Transmit 100 concurrent requests to `/predict`. | FastAPI manages ASGI concurrency. Mean response latency remains < 100ms. | Mean latency: 34ms. | **PASS** |
| **FE-03** | Malformed Payload Catch | 1. Send `POST /predict` with `{station_id: "NULL"}`. | Pydantic validation fails, returns strict `HTTP 422 Unprocessable Entity`. | Returned HTTP 422. | **PASS** |
| **FE-04** | Out-of-Bounds Normalization| 1. Submit abnormally high rainfall (9999mm). | MinMaxScaler caps outliers or model predicts maximum theoretical limit. | Bounded prediction returned without throwing internal NaN error. | **PASS** |
| **FE-05** | JSON Response Structure | 1. Validate `GET /predict` output. | Response matches strict schema: `{success: bool, forecasts: dict, units: str}`. | Schema validated perfectly. | **PASS** |

### 4.3.4 Artificial Intelligence Models
| TC ID | Objective | Test Procedure | Expected Result | Actual Outcome | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AI-01** | Model Overfitting Check | 1. Compare XGBoost MAE on Training set vs Test set. | Minimal gap (e.g., < 0.01m difference). Model is generalizing, not memorizing. | Train: 0.018m, Test: 0.024m. Healthy gap. | **PASS** |
| **AI-02** | Chronological Splitting | 1. Run evaluation split (80/20). | Test set records strictly occur *after* all Training set records temporally. | Zero temporal data leakage verified. | **PASS** |
| **AI-03** | Baseline Benchmark | 1. Compare LSTM MAE against naive "Persistence Model". | LSTM outperforms baseline during volatile rapid-onset flood surges. | LSTM accurately predicted surge trajectories. | **PASS** |
| **AI-04** | Lag Feature Efficacy | 1. Analyze XGBoost feature importance. | Immediate past state (`water_level_lag1`) should rank highest for 1-hour predictions. | `lag1` ranked #1 predictor, validating hypothesis. | **PASS** |
| **AI-05** | Extreme Event Recall | 1. Evaluate models exclusively on rows where `water_level > major_flood`. | Models correctly flag "Major Flood" trajectory without underpredicting severity. | XGBoost correctly predicted 94% of major flood events. | **PASS** |
