"""
AgroAI - Search Algorithms: Breadth-First Search (BFS)
Module: Farm Decision Node Exploration

Educational Concept:
- BFS explores graph nodes level by level using a Queue (First-In, First-Out).
- It guarantees finding the path with the fewest edges in unweighted graphs.
"""

from collections import deque

def bfs(graph: dict, start: str, goal: str):
    """
    Breadth-First Search implementation.
    
    :param graph: Dictionary representing adjacency list of farm nodes
    :param start: Starting node (e.g. 'Tractor_Hub')
    :param goal: Target destination node (e.g. 'Field_D')
    :return: List of nodes representing shortest path, or None
    """
    # Queue stores paths: initially containing path with just start node
    queue = deque([[start]])
    
    # Track visited nodes to prevent cycles
    visited = {start}
    
    while queue:
        # Pop the oldest path from front of queue (FIFO)
        path = queue.popleft()
        current_node = path[-1]
        
        # Check if goal destination reached
        if current_node == goal:
            return path
            
        # Explore unvisited neighbors level-by-level
        for neighbor in graph.get(current_node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
                
    return None


if __name__ == "__main__":
    print("=== AgroAI Educational BFS Search Demonstration ===")
    
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
    
    print(f"\n[1] Searching path from '{start_node}' to '{goal_node}' using BFS...")
    discovered_path = bfs(farm_graph, start_node, goal_node)
    
    if discovered_path:
        print(f"\n[2] Path Found (Level-by-Level): {' -> '.join(discovered_path)}")
    else:
        print("\n[!] Path not reachable.")
