#!/usr/bin/env python3
import heapq
import itertools
import math
import random
import time

def generate_grid(rows, cols, base=(0,0)):
    """
    Return a layout list of strings with
    'G' for free and one 'B' at base position.
    """
    layout = [['G']*cols for _ in range(rows)]
    bi, bj = base
    if 0 <= bi < rows and 0 <= bj < cols:
        layout[bi][bj] = 'B'
    return [''.join(r) for r in layout]

class Grid:
    def __init__(self, layout):
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
        if not self.in_bounds(cell): return False
        i, j = cell
        return self.grid[i][j] == 'G'
    def neighbors(self, cell):
        i, j = cell
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb = (i+di, j+dj)
            if self.is_free(nb): yield nb

class Drone:
    def __init__(self, start, time_budget):
        self.start = start
        self.position = start
        self.time_left = time_budget
        self.route = [start]
        # each entry: (event_cell, p, p_eff, attempts, detected)
        self.detection_log = []
        self.combined_detection_prob = 0.0

class Planner:
    def __init__(self, grid, events, neighbor_factor=0.9):
        """
        events: dict mapping (i,j) -> base detection probability
        neighbor_factor: multiplier for adjacent‐cell detection
        """
        self.grid = grid
        self.events = events
        self.neighbor_factor = neighbor_factor

    # ——— core A* & path helpers —————————————————————
    def heuristic(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def astar(self, start, goal):
        if not self.grid.is_free(start) or not self.grid.in_bounds(goal):
            return None
        heap = [(self.heuristic(start,goal), 0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        while heap:
            _, cost, cur = heapq.heappop(heap)
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
                    heapq.heappush(heap, (nc + self.heuristic(nb,goal), nc, nb))
        return None

    def find_path(self, start, target):
        # try direct; else to nearest free neighbor
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

    # ——— v1 control‐loop simulation (“stay‐to‐boost”) —————————————
    def plan_mission(self, drone):
        while drone.time_left > 0:
            chosen_evt = None
            best_score = 0.0
            # store temporary best params
            best_path = None
            best_peff = 0.0
            best_cost_evt = best_cost_home = 0

            for evt, base_p in self.events.items():
                # skip already‐detected events
                if any(evt == rec[0] and rec[4] for rec in drone.detection_log):
                    continue

                path_to = self.find_path(drone.position, evt)
                if not path_to:
                    continue
                ce = len(path_to)-1
                if ce < 1 or ce > drone.time_left:
                    continue

                # cost back home
                detect_pos = evt if self.grid.is_free(evt) else path_to[-1]
                path_back = self.astar(detect_pos, drone.start)
                if not path_back:
                    continue
                ch = len(path_back)-1
                avail = drone.time_left - ce - ch
                if avail < 1:
                    continue

                # effective p
                peff = base_p if self.grid.is_free(evt) else self.neighbor_factor*base_p
                total_p = 1 - (1-peff)**avail
                score = total_p / (ce + avail)
                if score > best_score:
                    chosen_evt = evt
                    best_score = score
                    best_path = path_to
                    best_peff = peff
                    best_cost_evt, best_cost_home = ce, ch

            if chosen_evt is None:
                break

            # travel to event
            for step in best_path[1:]:
                drone.position = step
                drone.route.append(step)
                drone.time_left -= 1

            # stay & flip coin until either home‐cost or detected
            attempts = 0
            detected = False
            while attempts < (drone.time_left - best_cost_home) and not detected:
                drone.time_left -= 1
                attempts += 1
                detected = (random.random() < best_peff)

            drone.detection_log.append(
                (chosen_evt, self.events[chosen_evt], best_peff, attempts, detected)
            )

        # return home if needed
        if drone.position != drone.start:
            home_path = self.astar(drone.position, drone.start)
            if home_path:
                for step in home_path[1:]:
                    drone.position = step
                    drone.route.append(step)
                    drone.time_left -= 1

        # combined detection probability
        p_none = 1.0
        for _, _, peff, att, det in drone.detection_log:
            if det:
                p_none *= (1-peff)**att
            else:
                p_none *= 1.0  # undetected remains in future, but union formula holds
        drone.combined_detection_prob = 1 - p_none

    # ——— Section III: region assignments w/ true “coin‐flip” removal —————————
    def compute_region_detection(self):
        region_probs = {}
        for evt, p in self.events.items():
            region_probs.setdefault(evt, {})[evt] = p
            for nb in self.grid.neighbors(evt):
                region_probs.setdefault(nb, {})[evt] = self.neighbor_factor * p
        return region_probs

    def greedy_region_assignment(self, K):
        rps = self.compute_region_detection()
        remaining = set(self.events.keys())
        chosen = []
        for _ in range(K):
            best_region = None
            best_val = 0.0
            # pick region with highest marginal sum p_eff over remaining
            for r, probs in rps.items():
                val = sum(probs[e] for e in remaining if e in probs)
                if val > best_val:
                    best_val, best_region = val, r
            if not best_region:
                break
            chosen.append(best_region)
            # now flip coin and only remove detected events
            for e in list(remaining):
                if e in rps[best_region]:
                    if random.random() < rps[best_region][e]:
                        remaining.remove(e)
            # undetected events stay for next round
        return chosen

    def optimal_region_assignment(self, K):
        rps = self.compute_region_detection()
        regions = list(rps.keys())
        best_subset, best_val = None, -1.0
        for subset in itertools.combinations(regions, K):
            total = 0.0
            for evt in self.events:
                peffs = [rps[r].get(evt, 0.0) for r in subset]
                total += 1 - math.prod([(1-x) for x in peffs])
            if total > best_val:
                best_val, best_subset = total, subset
        return list(best_subset), best_val

# ——— ASCII printer & runner helpers —————————————————————————————
def print_ascii_grid(grid, events, route=None):
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
    for i,row in enumerate(disp):
        print(f"{i%10}  " + "".join(row))
    print()

def run_single(grid, events, budget, seed=None):
    if seed is not None:
        random.seed(seed)
    drone = Drone(grid.start, budget)
    planner = Planner(grid, events)
    t0 = time.perf_counter()
    planner.plan_mission(drone)
    t1 = time.perf_counter()
    detections = sum(1 for *_,det in drone.detection_log if det)
    return {
        'drone': drone,
        'route': drone.route,
        'detection_log': drone.detection_log,
        'time_left': drone.time_left,
        'attempts': len(drone.detection_log),
        'detections': detections,
        'combined_prob': drone.combined_detection_prob,
        'runtime_s': t1-t0
    }

if __name__ == "__main__":
    # Example: auto‐generate a 10×10 grid with base at (0,0)
    layout = generate_grid(10, 10, base=(0,0))
    grid = Grid(layout)
    events = { (2,2):0.8, (3,8):0.4, (7,5):0.6, (8,4):0.5 }
    budget = 30
    runs = 10
    K = 2

    print("Legend: G=free, B=base, E=event, *=route\n")
    print("Initial grid:")
    print_ascii_grid(grid, events)

    # v1 single‐run
    res0 = run_single(grid, events, budget, seed=1)
    print("Final route (v1):")
    print_ascii_grid(grid, events, res0['route'])
    print(f"Time left: {res0['time_left']}, detections: {res0['detections']}, combined_prob: {res0['combined_prob']:.2f}\n")
    print("="*60, "\n")

    # v1 multi‐run averages
    stats = {k:[] for k in ('time_left','attempts','detections','combined_prob','runtime_s')}
    for _ in range(runs):
        r = run_single(grid, events, budget)
        for k in stats:
            stats[k].append(r[k])
    print(f"Averages over {runs} runs (budget={budget}):")
    for k, v in stats.items():
        print(f"  {k:14s} {sum(v)/runs:.2f}")
    print("\n" + "="*60 + "\n")

    # v2/v3 region assignment
    planner = Planner(grid, events)
    greedy_choice = planner.greedy_region_assignment(K)
    opt_choice, opt_val = planner.optimal_region_assignment(K)
    rps = planner.compute_region_detection()
    greedy_val = sum(1 - math.prod([1-x for x in [rps[r].get(e,0) for r in greedy_choice]]) for e in events)

    print(f"Greedy regions (K={K}): {greedy_choice}, expected detections: {greedy_val:.2f}")
    print(f"Optimal regions (K={K}): {opt_choice}, expected detections: {opt_val:.2f}")
    print(f"Ratio greedy/optimal: {greedy_val/opt_val:.2f}")

