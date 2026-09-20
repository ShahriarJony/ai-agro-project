# AgroAI — Constraint Satisfaction Problem (CSP) & AC-3 Module

## 1. What is CSP?
A **Constraint Satisfaction Problem (CSP)** consists of a set of variables, a domain of values for each variable, and a set of constraints that restrict the combinations of values the variables can take.

## 2. Core CSP Components in AgroAI
- **Variables**: Fields requiring irrigation assignment (`Field_A`, `Field_B`, `Field_C`, `Field_D`).
- **Domains**: Allowed time slots (`Morning`, `Afternoon`, `Evening`) and allocated pump stations (`Pump P1`, `Pump P2`).
- **Constraints**:
  1. *Single Pump Conflict*: A single pump cannot irrigate two fields at the exact same time slot.
  2. *Resource Limit*: Total water consumption per slot cannot exceed available reservoir capacity (8,200 L).
  3. *Crop Priority*: Critical fields (`Field_D`) must receive priority morning slots.

## 3. Why Use CSP for Irrigation?
Irrigation scheduling on modern farms involves multi-resource allocation with hard physical limits. CSP guarantees that any schedule produced does not violate hardware or water capacity bounds.

## 4. What Does AC-3 (Arc Consistency Algorithm #3) Do?
AC-3 is a domain reduction algorithm. It checks every pair of constrained variables $(X, Y)$ and removes values from $X$'s domain that have no valid partner in $Y$'s domain.
This significantly reduces the search space before running search algorithms.

## 5. What Will Backtracking Search Do in Week 2?
While AC-3 reduces domains, **Backtracking Search** (in Week 2) will perform systematic assignment to find complete, concrete irrigation schedules for all fields.
