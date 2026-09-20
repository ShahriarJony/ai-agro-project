# AgroAI System Architecture Documentation

## 1. High-Level System Architecture Flow

```text
┌─────────────────────────────────────────────────────────────┐
│                 React Frontend (Stitch UI)                  │
│             (TypeScript + Vite + Tailwind CSS)              │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP REST API / JSON
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                        │
│                   (Python 3.14 Server)                      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
┌─────────────────────────────┐ ┌─────────────────────────────┐
│          AI Engine          │ │    Firebase / Firestore     │
│  (Algorithmic Solvers +     │ │   (Cloud Auth & Data)       │
│   Prediction Interfaces)    │ └─────────────────────────────┘
└─────────────────────────────┘
```

## 2. ML Training vs Algorithmic Engine Status

> **Notice on Machine Learning Status:**
> Full application architecture, API contracts, OpenCV preprocessing pipelines, and model-loading abstractions are implemented. Machine learning model training (K-Means, Decision Tree, CNN) is intentionally postponed to a future phase. Fallback/demo predictions are clearly labeled (`status: "Demo / Model Not Trained"`).

```text
AI Engine Modules:
├── Algorithmic Solvers (Fully Implemented & Active)
│   ├── CSP & AC-3 Solver (Arc Consistency domain reduction)
│   ├── Backtracking Search (Timetable allocation)
│   ├── Genetic Algorithm Optimizer (Multi-objective schedule evolution)
│   ├── Search Engine (BFS & DFS diagnostic graph search)
│   ├── A* Pathfinding (Grid tractor navigation with f(n) = g(n) + h(n))
│   └── Minimax Simulation (Adversarial pest & climate risk evaluation)
│
└── Machine Learning Modules (Prediction & Ingestion Interfaces Ready; Training Pending)
    ├── K-Means Clustering (Feature extraction & zone interpretation)
    ├── Decision Tree Classifier (Rule-based recommendation interface)
    └── CNN Disease Classifier (OpenCV 224x224 RGB preprocessing & PyTorch model loader)
```

## 3. Data Pipeline & Storage Flow

1. **Frontend UI (Stitch Design System)**: Sends user inputs or leaf image binary streams to FastAPI endpoints via `apiService`.
2. **FastAPI Middleware**: Validates CORS headers and inspects incoming Pydantic request models or multipart image uploads.
3. **OpenCV & Feature Preprocessing**: Image files are converted to RGB, resized to `224x224`, and normalized to `[0.0, 1.0]`. Numeric field data is normalized for clustering.
4. **Model Loading Abstraction**: Checks for pre-trained weights (`.pkl`, `.pt`, `.pth`). If missing, invokes deterministic demo/heuristic handlers returning structured JSON responses marked with `is_trained: false`.
5. **Algorithmic Search & Solvers**: Executes pure Python solvers (AC-3, Backtracking, GA, A*, BFS, DFS, Minimax) returning optimal execution schedules or paths.
6. **Persistence Layer**: Cloud Firestore persists farm dossiers, field telemetries, activity audit logs, and irrigation schedules.
