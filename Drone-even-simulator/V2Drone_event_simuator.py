import heapq
import itertools
import math
import random
import time

class Grid:
    def __init__(self, layout):
        """Initialize the grid from a list of strings.
        'G' = free, 'O' = no-fly, 'R' = obstacle, 'B' = base (start/goal)."""
        self.rows = len(layout)
        self.cols = len(layout[0]) if self.rows else 0
        self.grid = []
        self.start = None
        for i, row in enumerate(layout):
            cells = []
            for j, ch in enumerate(row):
                if ch == 'B':
                    self.start = (i, j)
                    cells.append('G')
                else:
                    cells.append(ch)
            self.grid.append(cells)
        if self.start is None:
            raise ValueError("Layout must include a 'B' for base/start.")

    def in_bounds(self, cell):
        i, j = cell
        return 0 <= i < self.rows and 0 <= j < self.cols

    def is_free(self, cell):
        if not self.in_bounds(cell):
            return False
        i, j = cell
        return self.grid[i][j] == 'G'

    def neighbors(self, cell):
        i, j = cell
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb = (i+di, j+dj)
            if self.is_free(nb):
                yield nb

class Drone:
    def __init__(self, start, time_budget):
        self.start = start
        self.position = start
        self.time_left = time_budget
        self.route = [start]
        # detection_log: (event_cell, p, p_eff, attempts, detected)
        self.detection_log = []
        self.combined_detection_prob = 0.0

class Planner:
    def __init__(self, grid, events, neighbor_factor=0.9):
        """
        grid: Grid instance
        events: dict cell->probability
        neighbor_factor: p_eff for adjacent detection
        """
        self.grid = grid
        self.events = events
        self.neighbor_factor = neighbor_factor

    def heuristic(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def astar(self, start, goal):
        if not self.grid.is_free(start) or not self.grid.in_bounds(goal):
            return None
        open_heap = [(self.heuristic(start, goal), 0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        while open_heap:
            _, cost, cur = heapq.heappop(open_heap)
            if cur == goal:
                path = []
                node = cur
                while node:
                    path.append(node)
                    node = came_from[node]
                return path[::-1]
            if cost > cost_so_far[cur]:
                continue
            for nb in self.grid.neighbors(cur):
                nc = cost + 1
                if nc < cost_so_far.get(nb, float('inf')):
                    cost_so_far[nb] = nc
                    came_from[nb] = cur
                    heapq.heappush(open_heap, (nc + self.heuristic(nb, goal), nc, nb))
        return None

    def find_path(self, start, target):
        if self.grid.is_free(target):
            return self.astar(start, target)
        best, bd = None, float('inf')
        for nb in self.grid.neighbors(target):
            p = self.astar(start, nb)
            if p:
                d = len(p)-1
                if d < bd:
                    best, bd = p, d
        return best

    def plan_mission(self, drone):
        """Greedy+A* simulation with 'stay to boost' behavior."""
        remaining = set()
        # original events: cell->p
        # use planner.events in code
        # store original event dict for reference
        events = self.events
        while drone.time_left > 0:
            best_evt = None
            best_path = None
            best_score = 0.0
            best_p, best_p_eff, cost_evt, cost_home = 0,0,0,0

            for evt, p in events.items():
                path_to = self.find_path(drone.position, evt)
                if not path_to:
                    continue
                ce = len(path_to)-1
                if ce < 1 or ce > drone.time_left:
                    continue
                detect_pos = evt if self.grid.is_free(evt) else path_to[-1]
                path_back = self.astar(detect_pos, drone.start)
                if not path_back:
                    continue
                ch = len(path_back)-1
                avail = drone.time_left - ce - ch
                if avail < 1:
                    continue
                p_eff = p if self.grid.is_free(evt) else self.neighbor_factor * p
                total_p = 1 - (1-p_eff)**avail
                score = total_p / (ce + avail)
                if score > best_score:
                    best_score = score
                    best_evt, best_path = evt, path_to
                    best_p, best_p_eff = p, p_eff
                    cost_evt, cost_home = ce, ch

            if not best_evt:
                break

            # move to event
            for step in best_path[1:]:
                drone.position = step
                drone.route.append(step)
                drone.time_left -= 1

            # stay-and-try
            max_stays = drone.time_left - cost_home
            attempts, detected = 0, False
            while attempts < max_stays and not detected:
                drone.time_left -= 1
                attempts += 1
                detected = (random.random() < best_p_eff)

            drone.detection_log.append((best_evt, best_p, best_p_eff, attempts, detected))

        # return home
        if drone.position != drone.start:
            home = self.astar(drone.position, drone.start)
            if home:
                for step in home[1:]:
                    drone.position = step
                    drone.route.append(step)
                    drone.time_left -= 1

        # combined detection prob
        prob_none = 1.0
        for _, _, p_eff, attempts, _ in drone.detection_log:
            prob_none *= (1 - p_eff)**attempts
        drone.combined_detection_prob = 1 - prob_none

    # --- New region-assignment methods ---

    def compute_region_detection(self):
        """
        Builds dict: region_cell -> {event_cell: p_eff}
        region candidates = each event + its free neighbors
        """
        region_probs = {}
        for evt, p in self.events.items():
            # direct region
            if evt not in region_probs:
                region_probs[evt] = {}
            region_probs[evt][evt] = p
            # neighbors
            for nb in self.grid.neighbors(evt):
                if nb not in region_probs:
                    region_probs[nb] = {}
                region_probs[nb][evt] = self.neighbor_factor * p
        return region_probs

    def greedy_region_assignment(self, K):
        """
        Greedy pick K region-cells:
        each step pick region maximizing sum p_eff over remaining events,
        then remove all events detectable by that region.
        """
        region_probs = self.compute_region_detection()
        remaining = set(self.events.keys())
        chosen = []
        for _ in range(K):
            best_r, best_val = None, 0.0
            for r, probs in region_probs.items():
                val = sum(probs[e] for e in remaining if e in probs)
                if val > best_val:
                    best_val, best_r = val, r
            if not best_r:
                break
            chosen.append(best_r)
            # remove events covered
            for e in list(remaining):
                if e in region_probs[best_r]:
                    remaining.remove(e)
        return chosen

    def optimal_region_assignment(self, K):
        """
        Brute-force all combinations of K regions, choose subset maximizing
        expected detections:
          sum over events of (1 - prod_{r in subset}(1 - p_eff[r][event]))
        """
        region_probs = self.compute_region_detection()
        regions = list(region_probs.keys())
        best_subset, best_val = None, -1.0
        for subset in itertools.combinations(regions, K):
            total = 0.0
            for evt in self.events:
                peffs = [region_probs[r].get(evt, 0.0) for r in subset]
                p_det = 1 - math.prod([(1-x) for x in peffs])
                total += p_det
            if total > best_val:
                best_val, best_subset = total, subset
        return list(best_subset), best_val

# --- ASCII grid printer ---
def print_ascii_grid(grid, events, route=None):
    """
    G=free, O=no-fly, R=obstacle, E=event, B=base, *=route
    """
    disp = [row[:] for row in grid.grid]
    for (i,j) in events:
        if grid.in_bounds((i,j)):
            disp[i][j] = 'E'
    if route:
        for (i,j) in route:
            disp[i][j] = '*'
    bi, bj = grid.start
    disp[bi][bj] = 'B'
    print("   " + "".join(str(c%10) for c in range(grid.cols)))
    for i, row in enumerate(disp):
        print(f"{i%10}  " + "".join(row))
    print()

if __name__ == "__main__":
    # --- Sample setup ---
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
        (8, 4): 0.5,
    }

    # --- Single-drone simulation (v1) ---
    drone = Drone(grid.start, time_budget=30)
    planner = Planner(grid, events, neighbor_factor=0.9)
    random.seed(1)

    print("Initial grid:")
    print_ascii_grid(grid, events)

    planner.plan_mission(drone)

    print("Final route:")
    print_ascii_grid(grid, events, drone.route)
    print("Time remaining:", drone.time_left)
    print("Detection log:")
    for evt, p, p_eff, attempts, det in drone.detection_log:
        status = "DETECTED" if det else "NOT DETECTED"
        print(f"  Event@{evt} p={p:.2f}, p_eff={p_eff:.2f}, attempts={attempts} -> {status}")
    print(f"Combined detection probability: {drone.combined_detection_prob:.2f}")
    print("\n" + "="*60 + "\n")

    # --- Region assignment demo (multi-drone) ---
    K = 2
    greedy_regions = planner.greedy_region_assignment(K)
    opt_regions, opt_val = planner.optimal_region_assignment(K)
    # calculate greedy expected detection via union formula
    region_probs = planner.compute_region_detection()
    greedy_val = 0.0
    for evt in events:
        peffs = [region_probs[r].get(evt, 0.0) for r in greedy_regions]
        greedy_val += 1 - math.prod([(1-x) for x in peffs])

    print(f"Greedy region selection (K={K}): {greedy_regions}")
    print(f"  Expected detections (union): {greedy_val:.2f}")
    print(f"Optimal region selection (K={K}): {opt_regions}")
    print(f"  Expected detections (union): {opt_val:.2f}")
    print(f"Greedy/Optimal ratio: {greedy_val/opt_val:.2f}")

