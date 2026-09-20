"""
AgroAI - Search Algorithms: A* Pathfinding Algorithm
Module: Farm Tractor Autonomous Navigation

Educational Concept:
- g(n) = actual cost from start node to current node n
- h(n) = estimated heuristic cost from node n to goal (Manhattan distance)
- f(n) = g(n) + h(n) = total estimated path cost through node n
- Priority Queue (Min-Heap) expands lowest f(n) node first.
"""

import heapq

def manhattan_distance(p1: tuple, p2: tuple) -> int:
    """
    Calculate Manhattan distance heuristic h(n) = |x1 - x2| + |y1 - y2|
    Admissible for 4-directional grid movement.
    """
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def astar_search(grid: list, start: tuple, goal: tuple):
    """
    A* Search pathfinding on a 2D farm grid.
    
    :param grid: 2D list where '.' = Road, '#' = Obstacle, 'S' = Start, 'G' = Goal
    :param start: (row, col) tuple of starting position S
    :param goal: (row, col) tuple of destination field G
    :return: List of (row, col) grid steps representing optimal path
    """
    rows = len(grid)
    cols = len(grid[0])

    # Min-heap priority queue storing tuples: (f_score, g_score, current_node, path)
    initial_h = manhattan_distance(start, goal)
    pq = [(initial_h, 0, start, [start])]

    # Dictionary storing best known g_score per coordinate
    g_scores = {start: 0}
    visited = set()

    # 4-directional movement vectors: Up, Down, Left, Right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while pq:
        f_score, g_score, current, path = heapq.heappop(pq)

        if current == goal:
            return path

        if current in visited:
            continue
        visited.add(current)

        r, c = current

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc)

            # Check grid boundaries and obstacle collision
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != '#':
                tentative_g = g_score + 1  # Cost per step = 1

                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    h_score = manhattan_distance(neighbor, goal)
                    f_total = tentative_g + h_score
                    heapq.heappush(pq, (f_total, tentative_g, neighbor, path + [neighbor]))

    return None


if __name__ == "__main__":
    print("=== AgroAI Educational A* Grid Pathfinding Demonstration ===")

    # Farm Grid Map
    # S = Tractor Start [0, 0]
    # G = Target Field [2, 3]
    # # = Water Reservoir / Obstacle [1, 1]
    # . = Farm Road
    farm_grid = [
        ['S', '.', '.', '.'],
        ['.', '#', '.', '.'],
        ['.', '.', '.', 'G']
    ]

    start_pos = (0, 0)
    goal_pos = (2, 3)

    print("\n[1] Farm Map Grid Legend:")
    print("  S = Tractor Start  |  G = Goal Field  |  # = Obstacle  |  . = Road")
    for row in farm_grid:
        print("  " + " ".join(row))

    print(f"\n[2] Computing A* path from S{start_pos} to G{goal_pos} using f(n) = g(n) + h(n)...")
    path_coordinates = astar_search(farm_grid, start_pos, goal_pos)

    if path_coordinates:
        print("\n[3] Optimal Tractor Navigation Path:")
        print(f"  Steps Count: {len(path_coordinates) - 1}")
        print("  Coordinates: " + " -> ".join(f"[{r},{c}]" for r, c in path_coordinates))
    else:
        print("\n[!] Path blocked by obstacles.")
