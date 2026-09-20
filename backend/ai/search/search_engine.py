"""
AgroAI - Search Algorithms Engine (BFS, DFS, A*)
"""

from collections import deque
import heapq

def bfs_search(grid: list, start: tuple, goal: tuple):
    rows, cols = len(grid), len(grid[0])
    queue = deque([[start]])
    visited = {start}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        path = queue.popleft()
        r, c = path[-1]
        if (r, c) == goal:
            return path
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != '#' and neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return None


def dfs_search(grid: list, start: tuple, goal: tuple):
    rows, cols = len(grid), len(grid[0])
    stack = [[start]]
    visited = {start}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while stack:
        path = stack.pop()
        r, c = path[-1]
        if (r, c) == goal:
            return path
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != '#' and neighbor not in visited:
                visited.add(neighbor)
                stack.append(path + [neighbor])
    return None
