"""
AgroAI - Search Algorithms: Depth-First Search (DFS)
Module: Farm Diagnostic Exploration

Educational Concept:
- DFS explores as deep as possible along each branch before backtracking.
- Uses a Stack (Last-In, First-Out) or recursion.
"""

def dfs(graph: dict, start: str, goal: str, path: list = None, visited: set = None):
    """
    Depth-First Search recursive implementation.
    
    :param graph: Dictionary representing adjacency list of farm nodes
    :param start: Current node being inspected
    :param goal: Target destination node
    :param path: Current exploration path trajectory
    :param visited: Set of already visited node identifiers
    :return: Path list if goal found, else None
    """
    if visited is None:
        visited = set()
    if path is None:
        path = [start]

    visited.add(start)

    # Base case: Goal node reached
    if start == goal:
        return path

    # Recurse deeply into unvisited neighboring nodes
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            result_path = dfs(graph, neighbor, goal, path + [neighbor], visited)
            if result_path:
                return result_path

    return None


if __name__ == "__main__":
    print("=== AgroAI Educational DFS Search Demonstration ===")

    # Sample Farm Network Graph
    farm_graph = {
        'Tractor_Hub': ['Road_Alpha', 'Water_Tank'],
        'Road_Alpha': ['Field_A', 'Field_B'],
        'Water_Tank': ['Pump_Station'],
        'Field_A': [],
        'Field_B': ['Field_C'],
        'Pump_Station': ['Field_D'],
        'Field_C': [],
        'Field_D': []
    }

    start_node = 'Tractor_Hub'
    goal_node = 'Field_D'

    print(f"\n[1] Searching path from '{start_node}' to '{goal_node}' using DFS...")
    discovered_path = dfs(farm_graph, start_node, goal_node)

    if discovered_path:
        print(f"\n[2] Path Found (Deep Exploration): {' -> '.join(discovered_path)}")
    else:
        print("\n[!] Path not reachable.")
