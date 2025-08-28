import time, os, psutil, warnings, gc, threading, argparse, itertools
from typing import Dict, List, Tuple
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

def setup_arg_parser():
    parser = argparse.ArgumentParser(description="Benchmark a TSP solver for N runs with random problems.")
    parser.add_argument("-n", "--runs", type=int, default=1, help="Number of simulation runs to perform. (Default: 1)")
    parser.add_argument("-c", "--cities", type=int, default=4, help="Number of cities for the random TSP problems. (Default: 4)")
    return parser.parse_args()

# ===============================================
# 2. 공통 유틸리티 함수
# ===============================================
def generate_random_tsp(num_cities: int, min_dist: int = 10, max_dist: int = 100) -> np.ndarray:
    dist_matrix = np.zeros((num_cities, num_cities), dtype=int)
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            random_dist = np.random.randint(min_dist, max_dist)
            dist_matrix[i, j] = random_dist
            dist_matrix[j, i] = random_dist
    return dist_matrix

class PeakMemSampler:
    def __init__(self, interval: float = 0.005):
        self.interval = interval; self._stop = False; self._t = None; self.peak_rss_mb = 0.0
    def _run(self):
        proc = psutil.Process(os.getpid())
        while not self._stop:
            rss_mb = proc.memory_info().rss / (1024*1024)
            if rss_mb > self.peak_rss_mb: self.peak_rss_mb = rss_mb
            time.sleep(self.interval)
    def __enter__(self):
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True); self._t.start()
        return self
    def __exit__(self, exc_type, exc, tb):
        self._stop = True
        if self._t: self._t.join()

# ===============================================
# 3. 알고리즘별 핵심 로직 (Held-Karp)
# ===============================================
def held_karp_optimized_py(W: List[List[int]]):
    n = len(W); start = 0; INF = float('inf')
    dp_prev: Dict[Tuple[int,int], int] = {(1<<start, start): 0}
    prev_map: Dict[Tuple[int,int], int] = {}
    for s in range(2, n+1):
        dp_cur: Dict[Tuple[int,int], int] = {}
        cur_prev: Dict[Tuple[int,int], int] = {}
        for subset in itertools.combinations(range(1, n), s-1):
            mask = 1<<start
            for v in subset: mask |= (1<<v)
            for j in subset:
                best = INF; best_prev = -1
                mask_wo = mask ^ (1<<j)
                for u in subset:
                    if u == j: continue
                    pv = dp_prev.get((mask_wo, u))
                    if pv is not None:
                        cand = pv + W[u][j]
                        if cand < best: best, best_prev = cand, u
                if best < INF:
                    key = (mask, j); dp_cur[key] = best; cur_prev[key] = best_prev
        dp_prev = dp_cur; prev_map.update(cur_prev)
    full = (1<<n) - 1; best_cost = INF; last = -1
    for j in range(1, n):
        pv = dp_prev.get((full, j))
        if pv is not None:
            cand = pv + W[j][start]
            if cand < best_cost: best_cost = cand; last = j
    path = [];
    if last != -1:
        path = [last]; mask = full
        while path[-1] != start:
            j = path[-1]; pj = prev_map.get((mask, j), start)
            mask ^= (1<<j); path.append(pj)
        path.reverse(); path.append(start)
    return best_cost, path

def solve_held_karp(W: np.ndarray) -> Dict:
    W_list = W.tolist() # HK 함수는 리스트를 입력으로 받음
    cost, path = held_karp_optimized_py(W_list)
    return {'cost': cost, 'path': path}

# ===============================================
# 4. 표준 벤치마크 실행 함수
# ===============================================
def run_single_benchmark(num_cities: int) -> Dict[str, float]:
    W = generate_random_tsp(num_cities)
    solution, elapsed_time, peak_memory = None, 0, 0
    with PeakMemSampler() as sampler:
        t0 = time.perf_counter()
        solution = solve_held_karp(W)
        elapsed_time = time.perf_counter() - t0
        peak_memory = sampler.peak_rss_mb
    return {"cost": solution['cost'], "time": elapsed_time, "mem_peak_rss": peak_memory}

# ===============================================
# 5. 완전 동일한 메인 실행부
# ===============================================
def summarize_results(records: list, algorithm_name: str):
    num_runs = len(records)
    g = lambda k: [r.get(k, float('nan')) for r in records]
    c = g("cost"); t = g("time"); r = g("mem_peak_rss")
    valid_costs = [x for x in c if x != float('inf')]
    avg_cost = np.mean(valid_costs) if valid_costs else float('inf')
    std_cost = np.std(valid_costs) if valid_costs else float('nan')
    
    print("\n" + "="*40)
    print(f"📊 {algorithm_name} Benchmark Results (after {num_runs} runs)")
    print("="*40)
    print("🎯 Shortest Distance Found:")
    print(f"   - Average: {avg_cost:.4f} (±{std_cost:.4f})")
    print("-"*40)
    print("⏱️  Execution Time:")
    print(f"   - Average: {np.mean(t):.4f} seconds (±{np.std(t):.4f})")
    print("-"*40)
    print("💾 Peak Memory Usage:")
    print(f"   - Average: {np.mean(r):.4f} MB (±{np.std(r):.4f})")
    print("="*40)

def main(num_runs: int, num_cities: int):
    records = []
    for i in range(num_runs):
        print(f"Running simulation {i+1}/{num_runs}...")
        result_dict = run_single_benchmark(num_cities)
        records.append(result_dict)
    summarize_results(records, "Held-Karp")

if __name__ == "__main__":
    args = setup_arg_parser()
    gc.collect()
    print(" === Configuration ==="); print(f"-   Algorithm: Held-Karp"); print(f"-   N Runs = {args.runs}"); print(f"-   N Cities = {args.cities}"); print("")
    main(num_runs=args.runs, num_cities=args.cities)