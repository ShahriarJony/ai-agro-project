# AgroAI — Firestore Database Schema Documentation

This document specifies the collection structure for Google Cloud Firestore.

## Implementation Scope
- **Week 1 Active Persistence**: Firebase Modular SDK initialization, Auth foundation helpers (`registerUser`, `loginUser`, `logoutUser`), and `fields` collection CRUD persistence (`saveField`, `getFields`).
- **Future Production Collections**: Documented schema blueprints below for full system rollout in Week 2.

## Collections Breakdown

### 1. `users`
```json
{
  "uid": "usr_99812",
  "name": "Demo Farmer",
  "email": "farmer@agroai.edu",
  "role": "farmer",
  "createdAt": "2026-09-16T12:00:00Z"
}
```

### 2. `farms`
```json
{
  "farmId": "farm_01",
  "name": "University Demo Farm",
  "location": "Sector 4",
  "totalAreaHectares": 25.5
}
```

### 3. `fields`
```json
{
  "fieldId": "field_a",
  "name": "Field A",
  "crop": "Rice",
  "soilMoisture": 75,
  "soilPH": 6.5,
  "temperature": 28,
  "status": "Healthy"
}
```

### 4. `soil_data`
```json
{
  "sensorId": "sensor_s01",
  "fieldId": "field_a",
  "moisturePercent": 75.2,
  "phValue": 6.5,
  "nitrogen": 40,
  "phosphorus": 25,
  "potassium": 30,
  "timestamp": "2026-09-16T12:30:00Z"
}
```

### 5. `weather_data`
```json
{
  "stationId": "weather_main",
  "temperatureC": 28.5,
  "humidityPercent": 70,
  "rainfallMm": 15.0,
  "solarRadiation": 450,
  "timestamp": "2026-09-16T12:30:00Z"
}
```

### 6. `disease_results`
```json
{
  "scanId": "scan_104",
  "fieldId": "field_b",
  "crop": "Tomato",
  "predictedDisease": "Early Blight",
  "confidenceScore": 0.94,
  "algorithm": "CNN MobileNetV2",
  "timestamp": "2026-09-16T10:15:00Z"
}
```

### 7. `resources`
```json
{
  "resourceId": "res_water_tank",
  "name": "Central Reservoir Tank",
  "type": "Water",
  "capacityLiters": 10000,
  "currentLevelLiters": 8200
}
```

### 8. `vehicles`
```json
{
  "vehicleId": "tractor_01",
  "name": "Autonomous Farm Tractor",
  "status": "Idle",
  "currentCoordinates": [1, 1],
  "fuelPercent": 88
}
```

### 9. `irrigation_schedules`
```json
{
  "scheduleId": "sched_week1",
  "fieldId": "field_d",
  "assignedSlot": "Morning",
  "pumpId": "Pump P1",
  "waterAllocatedLiters": 1500,
  "solver": "AC-3 CSP"
}
```

### 10. `ai_decisions`
```json
{
  "decisionId": "dec_5501",
  "fieldId": "field_c",
  "algorithm": "Decision Tree",
  "recommendation": "Irrigation Urgent",
  "triggeredBy": "Soil moisture < 25%",
  "createdAt": "2026-09-16T09:00:00Z"
}
```

### 11. `simulation_runs`
```json
{
  "simulationId": "sim_001",
  "module": "A* Pathfinding",
  "stepsTaken": 5,
  "startNode": "[0, 0]",
  "goalNode": "[2, 3]",
  "executionTimeMs": 1.4
}
```
