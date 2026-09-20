# AgroAI — FastAPI REST API Reference

Base URL: `http://localhost:8000/api`

## 1. System & Health

### `GET /api/health`
Returns backend connection status.

**Response**:
```json
{
  "status": "ok",
  "project": "AgroAI",
  "fastapi_version": "0.110",
  "ai_modules_ready": true
}
```

---

## 2. Farm Management (CRUD)

* `GET /api/farms` — List all farms
* `POST /api/farms` — Create new farm
* `PUT /api/farms/{id}` — Update farm details
* `DELETE /api/farms/{id}` — Delete farm

---

## 3. Field Management (CRUD)

* `GET /api/fields` — List all field telemetry records
* `POST /api/fields` — Add new field sector
* `PUT /api/fields/{id}` — Update field sensor values
* `DELETE /api/fields/{id}` — Remove field record

---

## 4. Telemetry & Logs

* `GET /api/weather` — Get current microclimate metrics & 7-day forecast
* `GET /api/resources` — Get water reservoir and IoT probe status
* `GET /api/logs` — List immutable system audit logs
* `POST /api/logs` — Add activity record to audit log

---

## 5. AI Engine Endpoints

### `POST /api/ai/kmeans`
Run K-Means soil management zone clustering.

### `POST /api/ai/decision-tree`
Run Decision Tree irrigation recommendation.

### `POST /api/ai/cnn`
Upload leaf image file (`multipart/form-data`) for CNN OpenCV preprocessing & disease classification.

### `POST /api/ai/csp`
Run AC-3 Arc Consistency domain reduction & Backtracking CSP irrigation solver.

### `POST /api/ai/search`
Run graph/grid pathfinding (`bfs`, `dfs`, `astar`).

### `POST /api/ai/minimax`
Run Minimax decision simulation for pest & climate risk management.

### `POST /api/ai/genetic`
Run Genetic Algorithm irrigation schedule optimization.
