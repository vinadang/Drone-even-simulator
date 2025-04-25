import heapq
import random
import time

class Grid:
    def __init__(self, layout):
        """Initialize the grid from a list of strings.
        Symbols: 'G' = free, 'O' = no-fly, 'R' = obstacle, 'B' = base (start/goal)."""
        self.rows = len(layout)
        self.cols = len(layout[0]) if self.rows > 0 else 0
        self.grid = []
        self.start = None
        for i, row in enumerate(layout):
            row_cells = []
            for j, ch in enumerate(row):
                if ch == 'B':
                    # Record base position, treat as free
                    self.start = (i, j)
                    row_cells.append('G')
                else:
                    row_cells.append(ch)
            self.grid.append(row_cells)
        if self.start is None:
            raise ValueError("Layout must include a 'B' for the drone base/start position.")
    
    def in_bounds(self, cell):
        """Check if a cell coordinate is within grid bounds."""
        i, j = cell
        return 0 <= i < self.rows and 0 <= j < self.cols
    
    def is_free(self, cell):
        """Check if a cell is free (traversable)."""
        if not self.in_bounds(cell):
            return False
        i, j = cell
        return self.grid[i][j] == 'G'
    
    def neighbors(self, cell):
        """Yield all free neighboring cells (4-directionally adjacent)."""
        i, j = cell
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (i + di, j + dj)
            if self.is_free(neighbor):
                yield neighbor

class Drone:
    def __init__(self, start_position, time_budget):
        self.start_position = start_position
        self.position = start_position
        self.time_left = time_budget
        self.route = [start_position]
        self.detection_log = []  # (event_cell, p, p_eff, detected)
        self.combined_detection_prob = 0.0

class Planner:
    def __init__(self, grid, events, neighbor_detection_factor=0.9):
        self.grid = grid
        self.events = events
        self.neighbor_factor = neighbor_detection_factor
    
    def heuristic(self, a, b):
        """Manhattan distance heuristic for A*."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def astar(self, start, goal):
        """A* search from start to goal. Returns path list or None."""
        if not self.grid.is_free(start) or not self.grid.in_bounds(goal):
            return None
        open_heap = [(self.heuristic(start, goal), 0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        while open_heap:
            _, cost, current = heapq.heappop(open_heap)
            if current == goal:
                path = []
                node = current
                while node is not None:
                    path.append(node)
                    node = came_from[node]
                return path[::-1]
            if cost > cost_so_far.get(current, float('inf')):
                continue
            for nb in self.grid.neighbors(current):
                new_cost = cost + 1
                if new_cost < cost_so_far.get(nb, float('inf')):
                    cost_so_far[nb] = new_cost
                    came_from[nb] = current
                    heapq.heappush(open_heap, (new_cost + self.heuristic(nb, goal), new_cost, nb))
        return None

    def find_path(self, start, target):
        """Path to target or nearest free neighbor if target blocked."""
        if self.grid.is_free(target):
            return self.astar(start, target)
        best_path = None
        best_dist = float('inf')
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb = (target[0]+di, target[1]+dj)
            if not self.grid.is_free(nb):
                continue
            path = self.astar(start, nb)
            if path:
                dist = len(path)-1
                if dist < best_dist:
                    best_dist = dist
                    best_path = path
        return best_path

    def plan_mission(self, drone):
        visited = set()
        while drone.time_left > 0:
            best = None
            best_score = 0.0
            best_path = None
            best_p_eff = 0.0

            for evt, p in self.events.items():
                if evt in visited:
                    continue
                path_to_evt = self.find_path(drone.position, evt)
                if not path_to_evt:
                    continue
                cost_to_evt = len(path_to_evt) - 1
                if cost_to_evt < 1 or cost_to_evt > drone.time_left:
                    continue
                detect_pos = evt if self.grid.is_free(evt) else path_to_evt[-1]
                path_home = self.astar(detect_pos, self.grid.start)
                if not path_home:
                    continue
                cost_home = len(path_home) - 1
                if cost_to_evt + cost_home > drone.time_left:
                    continue
                p_eff = p if self.grid.is_free(evt) else self.neighbor_factor * p
                score = p_eff / cost_to_evt
                if score > best_score:
                    best_score = score
                    best = evt
                    best_path = path_to_evt
                    best_p_eff = p_eff

            if best is None:
                break

            for step in best_path[1:]:
                drone.position = step
                drone.route.append(step)
                drone.time_left -= 1

            detected = random.random() < best_p_eff
            drone.detection_log.append((best, self.events[best], best_p_eff, detected))
            visited.add(best)

        if drone.position != self.grid.start:
            home_path = self.astar(drone.position, self.grid.start)
            if home_path:
                for step in home_path[1:]:
                    drone.position = step
                    drone.route.append(step)
                    drone.time_left -= 1

        prob_none = 1.0
        for _, _, p_eff, _ in drone.detection_log:
            prob_none *= (1 - p_eff)
        drone.combined_detection_prob = 1 - prob_none


def print_ascii_grid(grid, events, route=None):
    """
    Prints the grid to the console:
    G = free, O = no-fly, R = obstacle, E = event, B = base, * = route
    """
    display = [row[:] for row in grid.grid]
    for (i, j), _ in events.items():
        if grid.in_bounds((i, j)):
            display[i][j] = 'E'
    if route:
        for (i, j) in route:
            display[i][j] = '*'
    bi, bj = grid.start
    display[bi][bj] = 'B'

    header = "   " + "".join(str(c % 10) for c in range(grid.cols))
    print(header)
    for i, row in enumerate(display):
        line = f"{i%10}  " + "".join(row)
        print(line)
    print()


def run_single(grid, events, budget, seed=None):
    """Run one simulation, return metrics."""
    if seed is not None:
        random.seed(seed)
    drone = Drone(grid.start, time_budget=budget)
    planner = Planner(grid, events)
    start_time = time.perf_counter()
    planner.plan_mission(drone)
    elapsed = time.perf_counter() - start_time
    detections = sum(1 for (_, _, _, detected) in drone.detection_log if detected)
    return {
        'route': drone.route,
        'time_left': drone.time_left,
        'attempts': len(drone.detection_log),
        'detections': detections,
        'combined_prob': drone.combined_detection_prob,
        'runtime_s': elapsed
    }

if __name__ == "__main__":
    # Sample grid 10x10
    layout = [
        "BOGGGGGGGG",
        "GOOOOOOGGG",
        "GGGGGOGGGG",
        "GGGRROGGGG",
        "GGGGRRGGGG",
        "GGGGRRGGGG",
        "GGGGGGGGGG",
        "GGOOOOOGGG",
        "GGORRRRGGG",
        "GGGGGGGGGG"
    ]
    grid = Grid(layout)
    events = {
        (2, 2): 0.8,
        (3, 8): 0.4,
        (7, 5): 0.6,
        (8, 4): 0.5
    }
    budget = 30
    runs = 10

    print("Legend:")
    print("  G = free cell (green)")
    print("  O = no-fly zone (orange)")
    print("  R = obstacle   (red)")
    print("  E = event      (probability cell)")
    print("  B = base       (start/finish)")
    print("  * = path taken by drone")
    print()

    print("Initial grid:")
    print_ascii_grid(grid, events)

    stats = {
        'time_left': [],
        'attempts': [],
        'detections': [],
        'combined_prob': [],
        'runtime_s': []
    }
    for _ in range(runs):
        res = run_single(grid, events, budget)
        for k in stats:
            stats[k].append(res[k])

    print(f"Performed {runs} simulation runs with time budget = {budget}\n")
    print("Average time left:       ", sum(stats['time_left'])/runs)
    print("Average event attempts:  ", sum(stats['attempts'])/runs)
    print("Average detections:      ", sum(stats['detections'])/runs)
    print("Average combined prob:   ", sum(stats['combined_prob'])/runs)
    print("Average runtime (sec):   ", sum(stats['runtime_s'])/runs)
