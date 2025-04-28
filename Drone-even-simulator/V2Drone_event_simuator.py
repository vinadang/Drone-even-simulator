#!/usr/bin/env python3
import heapq
import itertools
import math
import random
import time

class Grid:
    def __init__(self, layout):
        self.rows = len(layout)
        self.cols = len(layout[0]) if self.rows else 0
        self.grid = []
        self.start = None
        for i,row in enumerate(layout):
            cells=[]
            for j,ch in enumerate(row):
                if ch=='B':
                    self.start=(i,j)
                    cells.append('G')
                else:
                    cells.append(ch)
            self.grid.append(cells)
        if self.start is None:
            raise ValueError("Layout must include a 'B' for base/start.")
    def in_bounds(self,cell):
        i,j=cell
        return 0<=i<self.rows and 0<=j<self.cols
    def is_free(self,cell):
        if not self.in_bounds(cell): return False
        i,j=cell
        return self.grid[i][j]=='G'
    def neighbors(self,cell):
        i,j=cell
        for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb=(i+di,j+dj)
            if self.is_free(nb): yield nb

class Drone:
    def __init__(self,start,budget):
        self.start=start
        self.position=start
        self.time_left=budget
        self.route=[start]
        self.detection_log=[]
        self.combined_detection_prob=0.0

class Planner:
    def __init__(self,grid,events,neighbor_factor=0.9):
        self.grid, self.events = grid, events
        self.neighbor_factor = neighbor_factor
    def heuristic(self,a,b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
    def astar(self,start,goal):
        if not self.grid.is_free(start) or not self.grid.in_bounds(goal):
            return None
        heap=[(self.heuristic(start,goal),0,start)]
        came, cost = {start:None}, {start:0}
        while heap:
            _,c,cur = heapq.heappop(heap)
            if cur==goal:
                path=[]; n=cur
                while n: path.append(n); n=came[n]
                return path[::-1]
            if c>cost[cur]: continue
            for nb in self.grid.neighbors(cur):
                nc = c+1
                if nc < cost.get(nb,1e9):
                    cost[nb]=nc
                    came[nb]=cur
                    heapq.heappush(heap,(nc+self.heuristic(nb,goal),nc,nb))
        return None
    def find_path(self,start,target):
        if self.grid.is_free(target):
            return self.astar(start,target)
        best,bd=None,1e9
        for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb=(target[0]+di,target[1]+dj)
            if not self.grid.is_free(nb): continue
            p = self.astar(start,nb)
            if p:
                d=len(p)-1
                if d<bd: best,bd=p,d
        return best
    def plan_mission(self,drone):
        events=self.events
        while drone.time_left>0:
            be=None; bp=None; bs=0; bpe=0; bce=bch=0
            for evt,p in events.items():
                path=self.find_path(drone.position,evt)
                if not path: continue
                ce=len(path)-1
                if ce<1 or ce>drone.time_left: continue
                dp = evt if self.grid.is_free(evt) else path[-1]
                back=self.astar(dp,drone.start)
                if not back: continue
                ch=len(back)-1
                avail=drone.time_left-ce-ch
                if avail<1: continue
                pe=p if self.grid.is_free(evt) else self.neighbor_factor*p
                tp=1-(1-pe)**avail
                score=tp/(ce+avail)
                if score>bs:
                    be, bp, bs, bpe, bce, bch = evt, path, score, pe, ce, ch
            if be is None: break
            # travel
            for s in bp[1:]:
                drone.position=s; drone.route.append(s); drone.time_left-=1
            # stay/retry
            attempts=0; detected=False
            while attempts<drone.time_left-bch and not detected:
                drone.time_left-=1; attempts+=1
                detected = (random.random()<bpe)
            drone.detection_log.append((be,events[be],bpe,attempts,detected))
        # return home
        if drone.position!=drone.start:
            home=self.astar(drone.position,drone.start)
            if home:
                for s in home[1:]:
                    drone.position=s; drone.route.append(s); drone.time_left-=1
        # combined prob
        pn=1
        for _,_,pe,att,_ in drone.detection_log:
            pn*=(1-pe)**att
        drone.combined_detection_prob = 1-pn
    # --- Section III methods ---
    def compute_region_detection(self):
        rp={}
        for evt,p in self.events.items():
            rp.setdefault(evt,{})[evt]=p
            for nb in self.grid.neighbors(evt):
                rp.setdefault(nb,{})[evt]=self.neighbor_factor*p
        return rp
    def greedy_region_assignment(self,K):
        rps=self.compute_region_detection()
        rem=set(self.events)
        chosen=[]
        for _ in range(K):
            br,bv=None,0
            for r,probs in rps.items():
                val=sum(probs[e] for e in rem if e in probs)
                if val>bv: bv,br=val,r
            if not br: break
            chosen.append(br)
            for e in list(rem):
                if e in rps[br]: rem.remove(e)
        return chosen
    def optimal_region_assignment(self,K):
        rps=self.compute_region_detection()
        regs=list(rps)
        bestS,bestV=None,-1
        for comb in itertools.combinations(regs,K):
            tot=0
            for evt in self.events:
                peffs=[rps[r].get(evt,0) for r in comb]
                tot+=1-math.prod([1-x for x in peffs])
            if tot>bestV: bestV,bestS=tot,comb
        return list(bestS),bestV

def print_ascii_grid(grid,events,route=None):
    disp=[row[:] for row in grid.grid]
    for (i,j) in events: 
        if grid.in_bounds((i,j)): disp[i][j]='E'
    if route:
        for (i,j) in route: disp[i][j]='*'
    bi,bj=grid.start; disp[bi][bj]='B'
    print("   "+"".join(str(c%10) for c in range(grid.cols)))
    for i,row in enumerate(disp):
        print(f"{i%10}  "+"".join(row))
    print()

def run_single(grid,events,budget,seed=None):
    if seed is not None: random.seed(seed)
    drone=Drone(grid.start,budget)
    planner=Planner(grid,events)
    t0=time.perf_counter()
    planner.plan_mission(drone)
    t1=time.perf_counter()
    detections=sum(1 for *_,d in drone.detection_log if d)
    return {
        'drone':drone,
        'route':drone.route,
        'detection_log':drone.detection_log,
        'time_left':drone.time_left,
        'attempts':len(drone.detection_log),
        'detections':detections,
        'combined_prob':drone.combined_detection_prob,
        'runtime':t1-t0
    }

if __name__=="__main__":
    layout=[
        "BOGGGGGGGG","GOOOOOOGGG","GGGGGOGGGG","GGGRROGGGG",
        "GGGGRRGGGG","GGGGRRGGGG","GGGGGGGGGG","GGOOOOOGGG",
        "GGORRRRGGG","GGGGGGGGGG"
    ]
    grid=Grid(layout)
    events={(2,2):0.8,(3,8):0.4,(7,5):0.6,(8,4):0.5}
    budget=30; runs=10; K=2

    # Legend
    print("Legend:")
    print("  G = free cell")
    print("  O = no-fly zone")
    print("  R = obstacle")
    print("  E = event cell")
    print("  B = base")
    print("  * = route\n")

    # Initial grid
    print("Initial grid:")
    print_ascii_grid(grid,events)

    # Single-run detailed v1
    res0=run_single(grid,events,budget,seed=1)
    print("Final route:")
    print_ascii_grid(grid,events,res0['route'])
    print("Time remaining:",res0['time_left'])
    print("Detection log:")
    for evt,p,peff,att,det in res0['detection_log']:
        print(f"  Event@{evt} p={p:.2f}, p_eff={peff:.2f}, attempts={att} ->",
              "DETECTED" if det else "NOT DETECTED")
    print(f"Combined detection probability: {res0['combined_prob']:.2f}")
    print("\n"+"="*60+"\n")

    # Multi-run v1 metrics
    stats={k:[] for k in ('time_left','attempts','detections','combined_prob','runtime')}
    for i in range(runs):
        r=run_single(grid,events,budget,seed=None)
        for k in stats: stats[k].append(r[k])
    print(f"Averages over {runs} runs (time budget={budget}):")
    for k,v in stats.items():
        avg=sum(v)/runs
        print(f"  {k.replace('_',' ').capitalize():18s} {avg:.2f}")
    print("\n"+"="*60+"\n")

    # Section III –  (v2)
    planner=Planner(grid,events)
    greedy=planner.greedy_region_assignment(K)
    opt,oval=planner.optimal_region_assignment(K)
    # Calculate greedy expectation
    rps=planner.compute_region_detection()
    gval=sum(1-math.prod([1-x for x in [rps[r].get(e,0) for r in greedy]]) for e in events)
    print(f"Greedy region selection (K={K}): {greedy}")
    print(f"  Expected detections (union): {gval:.2f}")
    print(f"Optimal region selection (K={K}): {opt}")
    print(f"  Expected detections (union): {oval:.2f}")
    print(f"Greedy/Optimal ratio: {gval/oval:.2f}")

