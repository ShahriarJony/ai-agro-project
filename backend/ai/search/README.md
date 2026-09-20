# AgroAI — Search Algorithms Module (BFS, DFS, A*)

## Overview & Comparison Table

| Algorithm | Main Idea | Data Structure | AgroAI Application |
| :--- | :--- | :--- | :--- |
| **BFS** (Breadth-First Search) | Explores level-by-level | Queue (FIFO) | Uninformed search for shortest path decision trees |
| **DFS** (Depth-First Search) | Explores one path deeply | Stack / Recursion | Diagnostic search across environmental fault trees |
| **A\*** (Heuristic Search) | Cost $g(n) + h(n)$ | Priority Queue (Min-Heap) | Autonomous farm tractor pathfinding around obstacles |

## 1. Breadth-First Search (BFS)
Explores graph nodes layer by layer. Guarantees finding the shortest path in unweighted graphs.

## 2. Depth-First Search (DFS)
Explores branch paths deeply before backtracking. Memory efficient for deep decision trees.

## 3. A* Search Algorithm
Combines exact cost from start $g(n)$ with estimated distance to goal $h(n)$:
$$f(n) = g(n) + h(n)$$
In AgroAI, $h(n)$ uses Manhattan distance $|x_1 - x_2| + |y_1 - y_2|$ to guide farm vehicles around water tanks and field obstacles.
