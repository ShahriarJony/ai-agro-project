# AgroAI — Full Project Implementation Skill

## 1. Project Identity

Project Name:
AgroAI — An Intelligent AI-Based Agricultural Decision Support System

Project Type:
University AI Lab project / academic prototype

Primary Goal:
Build a complete, functional, explainable agricultural decision-support web application.

IMPORTANT:
This phase implements the COMPLETE APPLICATION.
Machine-learning model training is intentionally postponed.

The application must be architected so trained models can be integrated later without redesigning the frontend or backend.

---

# 2. CORE DEVELOPMENT RULE

Follow this rule throughout the entire project:

> IMPLEMENT THE COMPLETE PROJECT FIRST. TRAIN ML MODELS LATER.

Do NOT stop implementation because ML datasets or trained models are unavailable.

Do NOT train K-Means, Decision Tree, or CNN models during this implementation phase.

Instead:

* build their complete API contracts
* build frontend interfaces
* build input validation
* build preprocessing interfaces
* build prediction service interfaces
* build model-loading architecture
* provide clearly labeled demo/fallback behavior where necessary
* make integration possible later without changing the UI architecture

Never claim demo output is a trained-model prediction.

---

# 3. SOURCE OF TRUTH

There are three sources of truth:

### Existing project

Existing functionality, business logic, API behavior, Firebase structure, algorithms, and useful components must be preserved.

### design_folder/

`design_folder/` is the UI/design source of truth.

### SKILL.md

This file defines implementation rules and engineering constraints.

Follow all three.

If there is a conflict:

1. Preserve working functionality.
2. Follow the requested Stitch design.
3. Follow this SKILL.md.
4. Make the smallest safe change.

---

# 4. DESIGN FOLDER — LOCKED

The directory:

```text
design_folder/
```

is a READ-ONLY design reference.

NEVER:

* modify files
* delete files
* rename files
* move files
* overwrite files
* add generated files inside it

Use the files only as design references.

Do NOT embed the HTML files using iframe.

Convert their visual structure into proper React + TypeScript components.

---

# 5. DESIGN FILES

Expected files include:

```text
design_folder/
├── agroai_ai_analysis_workspace.html
├── agroai_dashboard_farm_overview.html
├── agroai_decision_engine.md
├── agroai_disease_detection_studio.html
├── agroai_irrigation_planner.html
├── agroai_decision_engine1.md
├── agroai_fields.html
├── agroai_history_logs.html
├── agroai_my_farm.html
├── agroai_settings.html
└── agroai_weather.html
```

Map designs approximately as follows:

```text
Dashboard
→ agroai_dashboard_farm_overview.html

AI Analysis
→ agroai_ai_analysis_workspace.html

Disease Detection
→ agroai_disease_detection_studio.html

Irrigation Planner
→ agroai_irrigation_planner.html

Fields
→ agroai_fields.html

My Farm
→ agroai_my_farm.html

History & Logs
→ agroai_history_logs.html

Weather
→ agroai_weather.html

Settings
→ agroai_settings.html
```

Inspect the two Decision Engine markdown files and integrate useful content into the application where appropriate.

There is no dedicated Algorithms HTML design.
Create Algorithms using the same Stitch visual language.

---

# 6. VISUAL DESIGN RULES

Use:

* light theme
* professional agricultural technology aesthetic
* deep forest green
* medium agricultural green
* warm off-white
* white
* muted olive
* soft beige
* slate/dark gray
* light gray borders

Typography:

* Inter
* Manrope
* Plus Jakarta Sans

Preferred UI:

* compact sidebar
* clean top header
* organized cards
* tables
* forms
* tabs
* filters
* clear status indicators
* practical agricultural visuals
* responsive layouts

Avoid:

* dark theme
* cyberpunk style
* neon colors
* excessive gradients
* glassmorphism
* glowing effects
* robot illustrations
* oversized headings
* excessive shadows
* generic AI dashboard appearance

Overall feeling:

> Quiet, useful, trustworthy, professional agricultural software.

---

# 7. EXISTING FUNCTIONALITY MUST BE PRESERVED

Before modifying anything:

1. inspect the project
2. inspect package.json
3. inspect frontend source
4. inspect backend source
5. inspect Firebase implementation
6. inspect existing API endpoints
7. inspect AI modules
8. inspect routing
9. inspect reusable components
10. inspect documentation
11. inspect progress.md
12. inspect Git status

Never blindly overwrite the project.

Do not remove working functionality just because the UI is being redesigned.

---

# 8. TECHNOLOGY STACK

Preferred architecture:

Frontend:

```text
React
TypeScript
Vite
Tailwind CSS
React Router
```

Backend:

```text
Python
FastAPI
```

Database/Auth:

```text
Firebase Authentication
Cloud Firestore
```

AI/Scientific:

```text
Python
Pandas
NumPy
Scikit-learn
OpenCV
PyTorch
```

Do not replace the stack unnecessarily.

---

# 9. REQUIRED ROUTES

The application should support at least:

```text
/dashboard
/my-farm
/fields
/weather
/ai-analysis
/disease-detection
/irrigation-planner
/algorithms
/resources
/history
/settings
```

All sidebar links must work.

No dead navigation links.

---

# 10. DASHBOARD

Dashboard must be functional.

Display meaningful live/application data such as:

* Total Fields
* Healthy Fields
* Needs Attention
* Critical Fields
* Water Available
* farm overview
* field monitoring
* recent activity
* AI status
* farm map / virtual farm overview

Data should come from application state/API/Firebase where available.

Do not hard-code everything if the underlying data already exists.

Provide:

* loading states
* empty states
* error states
* refresh behavior

---

# 11. MY FARM

Implement complete farm management.

Users should be able to:

* create farm
* view farm
* edit farm
* delete farm where appropriate
* add fields
* view field information
* update field information

Suggested data:

```text
Farm Name
Location
Area
Soil Type
Main Crop
Description
```

Connect the data to Firestore if Firebase is configured.

---

# 12. FIELDS

Implement functional field management.

Each field should support:

```text
Field ID
Field Name
Crop
Area
Soil Moisture
Soil pH
Temperature
Humidity
Rainfall
Water Availability
Status
```

Support:

* create
* read
* update
* delete
* search
* filtering
* field details
* AI analysis entry point

Do not break existing field functionality.

---

# 13. WEATHER

Implement the weather page.

Support:

* current weather
* temperature
* humidity
* rainfall
* wind
* forecast where API supports it
* location
* weather status

Use environment variables for API configuration.

If weather API credentials are unavailable:

* show a proper configuration state
* do not fabricate live weather
* optionally provide clearly labeled sample/demo data

---

# 14. RESOURCES & SENSORS

Implement agricultural resource monitoring.

Support conceptual/simulated prototype data for:

```text
Water Tank
Pump 1
Pump 2
Soil Moisture Sensor
Temperature Sensor
Humidity Sensor
Water Level Sensor
Rain Sensor
```

Simulated sensor values are acceptable for this academic prototype.

Clearly label simulated data.

Do not claim actual IoT connectivity unless implemented.

---

# 15. FIREBASE

If Firebase is already configured, complete the real integration.

Use the official Firebase JavaScript SDK.

Implement:

```text
initializeApp()
getAuth()
getFirestore()
```

Authentication:

```text
register
login
logout
auth state
```

Firestore operations where appropriate:

```text
create
read
update
delete
```

Suggested collections:

```text
users
farms
fields
soil_data
weather_data
resources
sensors
disease_results
ai_predictions
irrigation_schedules
activity_logs
```

Respect the existing schema if already implemented.

Do not expose real secrets.

Use:

```text
.env
.env.example
```

Never commit real credentials.

If Firebase is not configured, the application must still start gracefully and explain the configuration state.

---

# 16. AI ARCHITECTURE

The AI system has three ML components:

```text
K-Means
Decision Tree
CNN
```

and several algorithmic components:

```text
CSP
AC-3
Backtracking
BFS
DFS
A*
Minimax
Genetic Algorithm
```

IMPORTANT:

The ML models are NOT trained during this implementation phase.

---

# 17. K-MEANS

Build the complete K-Means architecture.

Expected features:

```text
soil_moisture
soil_ph
temperature
humidity
rainfall
```

Architecture:

```text
Input
↓
Validation
↓
Preprocessing
↓
Feature preparation
↓
K-Means prediction service
↓
Cluster
↓
Agricultural interpretation
↓
Frontend
```

Create a clean interface so a trained model can later be loaded.

Before training:

* use a clearly labeled demo/simulation service if necessary
* never claim it is trained
* expose the same response format that the future trained model will use

Possible status labels:

```text
Healthy / Moist
Moderate
Dry
Critical
```

These labels are domain interpretations, not native K-Means labels.

---

# 18. DECISION TREE

Build the complete Decision Tree architecture.

Potential inputs:

```text
crop
soil_moisture
soil_ph
temperature
humidity
rainfall
water_availability
```

Potential output:

```text
irrigation recommendation
```

Architecture:

```text
Input
↓
Validation
↓
Feature preprocessing
↓
Decision Tree inference interface
↓
Recommendation
↓
Explanation
```

Do NOT train the model now.

Use a clearly labeled temporary/demo prediction implementation until a trained model exists.

The API contract must remain stable after training.

---

# 19. CNN DISEASE DETECTION

Implement the complete disease-detection workflow WITHOUT training.

Frontend must support:

```text
Upload leaf image
↓
Preview
↓
Validate file
↓
Send to FastAPI
↓
Preprocess
↓
CNN inference interface
↓
Disease result
↓
Confidence
↓
Severity
↓
Recommendation
```

Recommended future architecture:

```text
OpenCV
↓
Resize 224x224
↓
Normalize
↓
MobileNetV2 / CNN
↓
Disease classification
```

Current phase:

* implement upload
* implement validation
* implement preprocessing interface
* implement API endpoint
* implement prediction service interface
* implement result display
* implement error handling

Do NOT train the CNN.

Do NOT fabricate model accuracy.

Do NOT display fake confidence as if it came from a trained model.

If demo results are required for UI testing, explicitly label them:

```text
Demo Prediction
Simulation Result
Model Not Trained
```

---

# 20. CSP

Implement real CSP irrigation scheduling logic.

Variables may include:

```text
Field
Pump
Time Slot
Water Allocation
```

Constraints:

```text
A pump cannot serve two fields at the same time.

Water usage must not exceed available water.

A field must receive sufficient irrigation.

Pump availability must be respected.

Time constraints must be respected.
```

Implement:

```text
CSP
AC-3
Backtracking
```

Show the generated schedule in the UI.

---

# 21. AC-3

Implement actual Arc Consistency.

The implementation should include:

```text
ac3()
revise()
is_consistent()
```

For an arc:

```text
X → Y
```

every remaining value in X must have compatible support in Y.

Use AC-3 to reduce irrigation scheduling domains.

---

# 22. BACKTRACKING

Implement a real backtracking search for CSP scheduling.

Expected workflow:

```text
Domains
↓
Choose variable
↓
Choose value
↓
Check constraints
↓
Continue
or
Backtrack
↓
Valid schedule
```

Provide explainable output where practical.

---

# 23. BFS / DFS

Implement real:

```text
BFS
DFS
```

Use them for agricultural decision/diagnostic search.

BFS:

```text
Queue / FIFO
```

DFS:

```text
Depth-first exploration
Visited tracking
```

The Algorithms page should explain these in academic language.

---

# 24. A*

Implement actual A*.

Use:

```text
f(n) = g(n) + h(n)
```

Use Manhattan distance for grid-based farm routing where appropriate.

Support:

* start
* goal
* obstacles
* path
* path cost

Use the result for virtual farm/tractor route planning.

---

# 25. MINIMAX

Implement Minimax as an agricultural decision simulation.

Example concept:

```text
Pest-risk scenario
        ↓
Possible farmer actions
        ↓
Possible adversarial/environment outcomes
        ↓
Minimax
        ↓
Recommended simulated action
```

Clearly explain that Minimax is a decision simulation and not a trained pest-detection model.

---

# 26. GENETIC ALGORITHM

Implement Genetic Algorithm for irrigation schedule optimization.

Possible chromosome:

```text
field + pump + time slot + water amount
```

Possible fitness factors:

```text
water efficiency
crop requirement satisfaction
time efficiency
pump utilization
constraint violations
```

Workflow:

```text
Population
↓
Fitness
↓
Selection
↓
Crossover
↓
Mutation
↓
New generation
↓
Best schedule
```

The result should be usable by the Irrigation Planner.

---

# 27. IRRIGATION PLANNER

Build an end-to-end irrigation workflow:

```text
Field Conditions
+
Weather
+
Water Resources
+
Pump Availability
        ↓
Decision Engine
        ↓
CSP
        ↓
AC-3
        ↓
Backtracking
        ↓
Genetic Algorithm optimization
        ↓
Final Schedule
```

Show:

* field
* recommended time
* pump
* water amount
* reason
* constraint status
* optimization information

---

# 28. AI ANALYSIS PAGE

Provide an organized workspace for:

```text
K-Means
Decision Tree
CNN
CSP + AC-3
A*
```

Each module should have:

* input section
* run/analyze button
* result section
* explanation
* loading state
* error state
* reset option where appropriate

The page should clearly distinguish:

```text
Algorithmic result
Demo model result
Trained model result
```

---

# 29. ALGORITHMS PAGE

Create a professional educational page.

Explain:

```text
K-Means
Decision Tree
CNN
CSP
AC-3
Backtracking
BFS
DFS
A*
Minimax
Genetic Algorithm
OpenCV
Pandas
NumPy
PyTorch
Scikit-learn
```

For each important algorithm show:

```text
What it is
Why AgroAI uses it
Input
Process
Output
```

Do not make it look like an IDE.

---

# 30. HISTORY & LOGS

Implement activity history.

Track important events:

```text
Farm created
Field created
Field updated
AI analysis executed
Disease analysis executed
Irrigation schedule generated
Weather updated
Resource changed
Settings changed
```

Use Firestore if configured.

Support:

* timestamp
* action
* category
* status
* details

---

# 31. SETTINGS

Implement functional settings.

Include:

```text
Firebase status
Backend/API status
Weather API status
Model status
Application information
```

Model status should distinguish:

```text
Not Trained
Demo Mode
Trained Model Available
```

Do not incorrectly show "AI Ready" if trained models do not exist.

---

# 32. API DESIGN

Keep API organization clean.

Example:

```text
/api/health
/api/fields
/api/farms
/api/weather

/api/ai/kmeans
/api/ai/decision-tree
/api/ai/cnn
/api/ai/csp
/api/ai/search
/api/ai/minimax
/api/ai/genetic
```

Do not remove existing working endpoints.

If endpoint structure needs improvement, maintain backward compatibility where practical.

Use Pydantic request/response models.

Validate inputs.

Return useful errors.

---

# 33. FRONTEND ↔ BACKEND

All AI pages must communicate through the API service layer.

Do not scatter raw fetch calls throughout components.

Prefer:

```text
frontend/src/services/api.ts
```

or an organized service structure.

Keep API types strongly typed in TypeScript.

---

# 34. ERROR HANDLING

Every major feature needs:

```text
Loading
Success
Empty
Error
Retry
```

Examples:

```text
Firebase unavailable
Backend unavailable
Weather API unavailable
Invalid image
Invalid field data
AI service unavailable
No irrigation solution
```

Errors should be understandable to normal users.

---

# 35. RESPONSIVENESS

The entire application must work on:

```text
Desktop
Tablet
Mobile
```

Do not allow:

* horizontal overflow
* broken tables
* unusable forms
* overlapping sidebar
* clipped buttons
* unreadable text

---

# 36. ACCESSIBILITY

Use:

* semantic HTML
* labels
* keyboard-friendly controls
* readable contrast
* useful button names
* image alt text
* focus states

---

# 37. DATA INTEGRITY

Never fabricate real-world data.

If data is:

```text
sample
mock
simulation
demo
```

label it clearly.

Never fabricate:

* trained model accuracy
* disease confidence
* real sensor readings
* live weather
* real farm GPS
* real IoT connectivity

---

# 38. SECURITY

Never:

* commit `.env`
* expose Firebase private credentials
* hard-code API secrets
* expose service-account JSON
* store secrets in frontend source

Use:

```text
.env
.env.example
```

Firebase security rules should protect user-owned data where applicable.

---

# 39. CODE QUALITY

Write code that is:

* simple
* readable
* modular
* maintainable
* beginner-friendly
* strongly typed where possible
* properly named

Avoid unnecessary abstractions.

Do not introduce libraries unless they solve a real project requirement.

Reuse existing components where possible.

---

# 40. DOCUMENTATION

Maintain:

```text
README.md
progress.md
docs/architecture.md
docs/firebase-schema.md
docs/ai-algorithms.md
docs/api.md
```

Documentation must accurately state:

```text
ML models are currently not trained.
The application contains model integration interfaces and demo/fallback behavior where required.
```

Never claim the project has trained AI models before training actually happens.

---

# 41. PROGRESS.MD

Update `progress.md` after meaningful implementation milestones.

Record:

```text
Completed
In Progress
Pending
Known Issues
```

Do not falsely mark model training as complete.

---

# 42. TESTING

Before declaring completion, test:

Frontend:

```text
npm install
npm run build
```

Backend:

```text
Python import/startup
/api/health
```

Test:

```text
routing
navigation
forms
CRUD
Firebase
API requests
loading states
error states
AI endpoints
CSP
AC-3
Backtracking
BFS
DFS
A*
Minimax
Genetic Algorithm
image upload
history
settings
```

Test the full workflow:

```text
Register
↓
Login
↓
Create Farm
↓
Add Field
↓
Enter Soil Data
↓
View Weather
↓
Analyze Field
↓
K-Means interface
↓
Decision Tree interface
↓
Disease Detection
↓
Irrigation Planner
↓
CSP
↓
AC-3
↓
Backtracking
↓
Genetic Algorithm
↓
Schedule
↓
History
```

---

# 43. GIT SAFETY

Before changes:

```text
git status
```

Never execute:

```text
git reset --hard
git clean -fd
git push --force
```

Do not delete user work.

Do not overwrite unrelated changes.

---

# 44. IMPLEMENTATION STRATEGY

The agent must work incrementally.

For each major area:

```text
Inspect
↓
Plan
↓
Implement
↓
Run/build/test
↓
Fix
↓
Continue
```

Do not rewrite the entire repository blindly.

Do not stop after creating UI mockups.

The final application must actually function.

---

# 45. COMPLETION STANDARD

The project is complete for this phase only when:

* UI is implemented from design_folder
* routes work
* navigation works
* Firebase integration works when configured
* CRUD works
* FastAPI works
* AI endpoints work
* non-ML algorithms work
* ML model interfaces work
* disease upload workflow works
* irrigation workflow works
* history works
* settings work
* loading/error/empty states work
* responsive design works
* documentation is updated
* build/tests pass

ML MODEL TRAINING IS NOT REQUIRED FOR COMPLETION OF THIS PHASE.

---

# 46. FINAL RULE

Do not ask the user to manually implement each small feature.

Inspect the repository and implement the complete application autonomously.

If a credential, API key, or external account connection is genuinely required and unavailable:

1. implement everything that does not require it
2. create the correct `.env.example`
3. provide a clear configuration state
4. continue implementing the remaining project
5. do not stop the entire project

Always preserve existing functionality.

Follow this SKILL.md strictly.
