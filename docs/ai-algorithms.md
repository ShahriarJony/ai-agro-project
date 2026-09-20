# AgroAI — AI & Algorithmic Foundations Documentation

## Overview

AgroAI combines deterministic graph search, constraint satisfaction, genetic optimization, and machine learning interfaces to deliver explainable decision support for modern precision agriculture.

---

## 1. Algorithmic Engines (Fully Implemented)

### 1.1 Arc Consistency 3 (AC-3) & CSP Backtracking
* **Purpose**: Generates conflict-free 24-hour irrigation schedules across fields, pumps, and water reservoir limits.
* **Input**: Variables (Fields), Domains (Time Slots), Constraints (Pump capacities, thermal lockout windows).
* **Process**: AC-3 reduces variable domains by removing inconsistent time slots. Backtracking search then assigns feasible slots.
* **Output**: Valid daily timetable with 0 constraint violations.

### 1.2 Genetic Algorithm (GA) Schedule Optimizer
* **Purpose**: Evolves long-term irrigation schedules to maximize water efficiency and minimize electricity tariff costs.
* **Input**: Field moisture deficits, time slots, population size (20), generations (15).
* **Process**: Chromosome evaluation via fitness function, tournament selection, single-point crossover, and random mutation.
* **Output**: Optimized schedule with efficiency gain metrics (+19.4% average savings vs fixed timers).

### 1.3 A* Pathfinding Search
* **Purpose**: Autonomous tractor navigation on farm grid maps avoiding obstacles (reservoir, compaction zones).
* **Formula**: $f(n) = g(n) + h(n)$ where $h(n)$ is Manhattan distance $|x_1 - x_2| + |y_1 - y_2|$.
* **Input**: Grid map, start coordinates `[r1, c1]`, goal coordinates `[r2, c2]`.
* **Output**: List of grid steps representing optimal path and path cost.

### 1.4 Breadth-First Search (BFS) & Depth-First Search (DFS)
* **Purpose**: Uninformed search for shortest path decision trees (BFS) and deep fault tree diagnosis (DFS).
* **Input**: Graph adjacency matrix, start node, goal node.
* **Output**: Traversal path list and visited node order.

### 1.5 Minimax Decision Simulation
* **Purpose**: Evaluates adversarial risk scenarios (farmer decisions MAX vs pest outbreak severity MIN).
* **Input**: Field crop type, candidate intervention strategies.
* **Output**: Recommended strategy maximizing minimum payoff score under worst-case pest vector outcomes.

---

## 2. Machine Learning Interfaces (Training Pending)

### 2.1 K-Means Clustering
* **Purpose**: Groups soil moisture, pH, temperature, humidity, and rainfall into homogeneous management zones.
* **ML Status**: **Not Trained** (Prediction interface and scikit-learn model loader implemented; demo heuristics active).
* **Future Training Pipeline**: `field_data.csv` -> StandardScaler -> `KMeans(n_clusters=4)` -> `kmeans_model.pkl`.

### 2.2 Decision Tree Classifier
* **Purpose**: Recommends irrigation intensity based on environmental threshold rules.
* **ML Status**: **Not Trained** (Rule-based prediction interface and joblib loader implemented).
* **Future Training Pipeline**: Soil dataset -> `DecisionTreeClassifier(max_depth=4)` -> `decision_tree_model.pkl`.

### 2.3 CNN Leaf Disease Classifier
* **Purpose**: Computer vision identification of foliar crop diseases (Early Blight, Late Blight, Rust).
* **ML Status**: **Not Trained** (OpenCV 224x224 RGB image preprocessing and PyTorch `.pth` loader abstraction implemented; fallback demo response active).
* **Future Training Pipeline**: PlantVillage dataset -> OpenCV resize & normalize -> PyTorch MobileNetV2 -> `plant_disease_cnn.pth`.
