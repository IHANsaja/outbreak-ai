## 3.4 Testing

Testing is a critical phase in the development of the Outbreak AI Flood Forecasting and Disaster Communication System to ensure the reliability, accuracy, and security of the disaster management platform. The testing strategy was divided into three main phases: Unit Testing, Integration Testing, and System Testing. These phases were rigorously applied across the four primary subsystems: the Online Mode, the Automated Data Pipeline, the Forecasting Engine, and the Artificial Intelligence Models.

### 3.4.1 Unit Testing

Unit testing focused on verifying the smallest testable parts of the software independently to ensure that individual functions, utility methods, and components operated correctly in isolation.

#### 3.4.1.1 Online Mode
Unit tests for the Online Mode (Web Platform) targeted the isolated functionality of Next.js Server Actions, utility functions, and React components.
*   **Geospatial Processing Logic:** Tested the Haversine formula utility to ensure accurate distance calculations between user coordinates and generated hazard locations.
*   **Data Validation:** Tested Pydantic and Zod schemas to ensure that invalid user inputs (e.g., negative water levels, missing coordinates) correctly threw validation errors before executing database operations.
*   **Role-Based Access Control (RBAC) Logic:** Verified that the authentication functions correctly extracted JWT tokens and identified user roles (`authority`, `citizen`) accurately.

#### 3.4.1.2 Data Pipeline
Unit tests for the Data Pipeline evaluated the individual data extraction and normalization scripts.
*   **Typographic Normalization:** Tested the `_safe_float()` regex function by passing corrupted strings (e.g., `"1.5m "`, `"N/A"`, `"missing"`) to guarantee it returned strict numeric floats or zero, preventing pipeline crashes.
*   **PDF Algorithmic Parser:** Tested the text extraction module against a single, well-structured DMC PDF bulletin to verify it could successfully locate and extract the "Rising" and "Falling" anchor keywords.
*   **OCR Image Processing:** Tested the fallback PyTorch `EasyOCR` function against a single mock image file to verify character recognition accuracy in total isolation.

#### 3.4.1.3 Forecasting Engine
Unit tests for the FastAPI Forecasting Engine focused on endpoint validation logic and tensor formatting for model integration.
*   **Sequence Padding:** Tested the algorithm responsible for padding historical telemetry arrays. Ensured that if only 5 hours of data were provided, the engine accurately duplicated the oldest records to meet the strict 12-hour tensor input requirement for the machine learning models.
*   **Payload Schema Compliance:** Sent malformed JSON payloads (e.g., `{station_id: "NULL"}`) to the `/predict` route to ensure FastAPI's dependency injection correctly returned a strict `422 Unprocessable Entity` response.

### 3.4.2 Integration Testing

Integration testing verified that the individual software units functioned correctly when combined, ensuring seamless data flow and communication between the system's internal modules.

#### 3.4.2.1 Online Mode Integration
*   **Middleware and Database Integration:** Tested the integration between the Next.js Middleware and the Supabase PostgreSQL database. Simulated a login session to ensure the middleware successfully queried the `profiles` table to validate the user's authority role before granting access to the protected `/authority` dashboard.
*   **Server Actions and UI Caching:** Verified that submitting an SOS incident via a Next.js Server Action successfully inserted the record into the database and triggered `revalidatePath()`, proving that the frontend UI components updated in real-time.

#### 3.4.2.2 Data Pipeline Integration
*   **Extraction to Storage Flow:** Tested the integration between the Hybrid Extractor and the Supabase database. Verified that after parsing a PDF bulletin, the pipeline successfully executed an `UPSERT` operation, updating the river water levels in the database without creating duplicate timestamps.
*   **Network Resilience Logic:** Integrated the extraction pipeline with network mocking tools to simulate connection drops during external PDF downloads. Verified that the `requests` library successfully caught the `ConnectionError` and executed its exponential backoff and retry mechanisms.

#### 3.4.2.3 Forecasting Engine and Model Integration
*   **API to Model Handshake:** Tested the integration between the FastAPI endpoints and the serialized `.pkl` (XGBoost) and `.pt` (LSTM) machine learning models. Verified that valid incoming historical arrays were correctly transformed into Pandas DataFrames, coerced into specific datatypes, and successfully passed to the models for inference.
*   **Feature Outlier Normalization:** Sent extreme out-of-bounds metrics (e.g., 9999mm rainfall) through the API to ensure the `MinMaxScaler` integration safely bounded the inputs before they reached the neural network layer, preventing internal `NaN` exceptions.

### 3.4.3 System Testing

System testing evaluated the complete, integrated application to verify that it met all functional and non-functional requirements in an environment mirroring the final production deployment (main project `D:\Projects\outbreak`).

#### 3.4.3.1 End-to-End Application Workflow
*   **Automated Alerting Workflow:** Conducted an end-to-end test where the Data Pipeline ingested a new bulletin showing rapidly rising water levels. Verified that the data was saved to the database, automatically forwarded to the Forecasting Engine API, and that the resulting 1-hour ahead prediction (indicating a major flood) immediately triggered a visual hazard alert on the Next.js Authority Dashboard.
*   **Auth Token Expiration System Test:** Simulated a complete user session where the JWT token was manually expired. Verified that attempting to fetch a restricted API route resulted in a 401 Unauthorized response and that the frontend gracefully forced a re-authentication redirect loop.

#### 3.4.3.2 Full Pipeline Resilience and Failover
*   **Hybrid Extraction Degradation Test:** Provided the live system with a scanned, image-only PDF bulletin to simulate a degraded government upload. Verified that the system attempted standard parsing, caught the resulting failure gracefully, triggered the OCR fallback, extracted the data, generated an AI prediction, and updated the UI without any system crash.
*   **Edge-Cloud Synchronization Test:** Disconnected the primary internet connection to simulate an ISP failure during a disaster. Verified that the local IoT edge nodes continued to cache SOS payloads locally and successfully bulk-synchronized them to the cloud via `UPSERT` immediately upon connection restoration.

#### 3.4.3.3 Performance and Model System Benchmark
*   **Production Latency and Load:** Executed stress testing on the Forecasting Engine by sending 100 concurrent requests to the `/predict` route. Verified that the mean response latency remained below 100ms, proving the FastAPI ASGI concurrency could handle disaster-scale traffic.
*   **System-Wide Model Accuracy Benchmark:** Evaluated the fully deployed XGBoost and LSTM models within the live Forecasting Engine against a naive Persistence Model. Validated that the models consistently achieved an $R^2 > 0.99$ and correctly flagged "Major Flood" trajectories natively within the application.
