import random
import time
import multiprocessing as mp


#  Instance generation
def generate_instance(num_tasks, num_vms):
    # Each task has some "work" amount
    tasks = [random.randint(10, 100) for _ in range(num_tasks)]
    # Each VM has a "speed" (higher = faster)
    vm_speeds = [random.uniform(0.8, 2.0) for _ in range(num_vms)]
    return tasks, vm_speeds

def initial_greedy_assignment(tasks, vm_speeds):
    num_vms = len(vm_speeds)
    loads = [0.0] * num_vms
    assignment = [[] for _ in range(num_vms)]

    for tid, work in enumerate(tasks):
        best_vm = min(
            range(num_vms),
            key=lambda v: loads[v] + work / vm_speeds[v]
        )
        assignment[best_vm].append(tid)
        loads[best_vm] += work / vm_speeds[best_vm]

    return assignment

def compute_makespan(assignment, tasks, vm_speeds):
    loads = []
    for v, tlist in enumerate(assignment):
        load = sum(tasks[tid] / vm_speeds[v] for tid in tlist)
        loads.append(load)
    return max(loads) if loads else 0.0


#  Best move for one VM
def best_move_for_vm(args):
    """
    Compute the best (most makespan-reducing) move
    for a single VM: move one of its tasks to another VM.
    """
    vm_idx, assignment, tasks, vm_speeds = args
    num_vms = len(vm_speeds)

    # current loads of all VMs
    loads = [
        sum(tasks[tid] / vm_speeds[v] for tid in assignment[v])
        for v in range(num_vms)
    ]
    current_makespan = max(loads) if loads else 0.0

    best_improvement = 0.0
    best_move = None  # (src_vm, dst_vm, task_id)

    for tid in assignment[vm_idx]:
        work = tasks[tid]
        for dst in range(num_vms):
            if dst == vm_idx:
                continue
            new_loads = list(loads)
            new_loads[vm_idx] -= work / vm_speeds[vm_idx]
            new_loads[dst]     += work / vm_speeds[dst]
            new_makespan = max(new_loads)
            improvement = current_makespan - new_makespan
            if improvement > best_improvement:
                best_improvement = improvement
                best_move = (vm_idx, dst, tid)

    return best_improvement, best_move


#  Sequential MMA
def seq_mma_allocation(assignment, tasks, vm_speeds,
                       max_rounds=80, min_improvement=1e-6):
    num_vms = len(vm_speeds)
    for _ in range(max_rounds):
        global_best_impr = 0.0
        global_best_move = None

        for vm_idx in range(num_vms):
            impr, move = best_move_for_vm((vm_idx, assignment, tasks, vm_speeds))
            if impr > global_best_impr:
                global_best_impr = impr
                global_best_move = move

        if not global_best_move or global_best_impr < min_improvement:
            break

        src, dst, tid = global_best_move
        assignment[src].remove(tid)
        assignment[dst].append(tid)

    return assignment


#  Parallel MMA
def par_mma_allocation(assignment, tasks, vm_speeds, n_workers=4,
                       max_rounds=80, min_improvement=1e-6):
    num_vms = len(vm_speeds)

    with mp.Pool(processes=n_workers) as pool:
        for _ in range(max_rounds):
            args = [
                (vm_idx, assignment, tasks, vm_speeds)
                for vm_idx in range(num_vms)
            ]
            results = pool.map(best_move_for_vm, args)

            global_best_impr = 0.0
            global_best_move = None
            for impr, move in results:
                if impr > global_best_impr:
                    global_best_impr = impr
                    global_best_move = move

            if not global_best_move or global_best_impr < min_improvement:
                break

            src, dst, tid = global_best_move
            assignment[src].remove(tid)
            assignment[dst].append(tid)

    return assignment


#  Experiment runner
def run_experiment(n_workers):
    random.seed(0)
    num_vms = 16
    task_sizes = [2000, 4000, 8000, 16000]

    print(f"\n=== n_workers = {n_workers} ===")
    print("tasks, seq_time, par_time, speedup, ms_seq, ms_par")

    for n_tasks in task_sizes:
        tasks, vm_speeds = generate_instance(n_tasks, num_vms)
        base_assign = initial_greedy_assignment(tasks, vm_speeds)

        # Sequential MMA
        assign_seq = [list(v) for v in base_assign]
        t0 = time.perf_counter()
        assign_seq = seq_mma_allocation(assign_seq, tasks, vm_speeds)
        t1 = time.perf_counter()
        ms_seq = compute_makespan(assign_seq, tasks, vm_speeds)

        # Parallel MMA
        assign_par = [list(v) for v in base_assign]
        t2 = time.perf_counter()
        assign_par = par_mma_allocation(assign_par, tasks, vm_speeds, n_workers=n_workers)
        t3 = time.perf_counter()
        ms_par = compute_makespan(assign_par, tasks, vm_speeds)

        seq_time = t1 - t0
        par_time = t3 - t2
        speedup = seq_time / par_time if par_time > 0 else 0.0

        print(f"{n_tasks}, {seq_time:.4f}, {par_time:.4f}, {speedup:.2f}, {ms_seq:.2f}, {ms_par:.2f}")

if __name__ == "__main__":
    mp.freeze_support()  # important on Windows

    # Try different worker counts
    for workers in [4, 8]:
        run_experiment(workers)
