import os
import csv
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class Task:
    id: int
    workload_mi: float
    cpu_req: float
    ram_req: float
    bw_req: float
    disk_req: float
    reliability_req: float
    security_req: float
    priority: int = 1

    def resource_vector(self):
        return (self.cpu_req, self.ram_req, self.bw_req, self.disk_req, self.reliability_req)


@dataclass
class VM:
    id: int
    cpu_cap: float
    ram_cap: float
    bw_cap: float
    disk_cap: float
    reliability: float
    security_level: float
    mips: float
    cost_per_time: float

    def resource_vector(self):
        return (self.cpu_cap, self.ram_cap, self.bw_cap, self.disk_cap, self.reliability)


# ============================================================
# BASELINE MMA
# ============================================================

class MMAScheduler:

    def __init__(self, max_rounds=50, load_threshold_factor=0.75, eps_improvement=1e-9, rng=None):
        self.max_rounds = max_rounds
        self.load_threshold_factor = load_threshold_factor
        self.eps_improvement = eps_improvement
        self.rng = rng or random.Random()

    @staticmethod
    def ect(task: Task, vm: VM):
        return task.workload_mi / vm.mips

    @staticmethod
    def ecc(task: Task, vm: VM):
        return MMAScheduler.ect(task, vm) * vm.cost_per_time

    @staticmethod
    def _passes_security_and_reliability(task: Task, vm: VM):
        return vm.security_level >= task.security_req and vm.reliability >= task.reliability_req

    def _feature_match_prob(self, t_val, v_val):
        if v_val <= 0:
            return 0
        if t_val > v_val:
            return 0
        return 1 / (1 + abs(t_val - v_val) / v_val)

    def _matching_degree(self, task, vm):
        t, v = task.resource_vector(), vm.resource_vector()
        probs = [self._feature_match_prob(t[i], v[i]) for i in range(4)]
        result = 1
        for p in probs:
            result *= p
        return result

    def matching_phase(self, tasks, vms):
        vm_index = {vm.id: vm for vm in vms}

        candidate_set = {
            t.id: [
                vm.id for vm in vms if self._passes_security_and_reliability(t, vm)]
            for t in tasks
        }

        ordered_tasks = sorted(
            tasks, key=lambda x: (x.priority, x.workload_mi))
        vm_load = {vm.id: 0 for vm in vms}
        first_alloc = {}

        for t in ordered_tasks:
            cands = candidate_set[t.id]
            if not cands:
                best_vm = max(vms, key=lambda x: x.security_level)
                first_alloc[t.id] = best_vm.id
                vm_load[best_vm.id] += self.ect(t, best_vm)
                continue

            best_vm, best_md = None, -1
            for vid in cands:
                md = self._matching_degree(t, vm_index[vid])
                if md > best_md:
                    best_md = md
                    best_vm = vid

            first_alloc[t.id] = best_vm
            vm_load[best_vm] += self.ect(t, vm_index[best_vm])

        return candidate_set, first_alloc

    def compute_stats(self, tasks, vms, alloc):
        vm_index = {vm.id: vm for vm in vms}
        vm_ect = {vm.id: 0 for vm in vms}
        vm_ecc = {vm.id: 0 for vm in vms}

        for t in tasks:
            vm = vm_index[alloc[t.id]]
            ect = self.ect(t, vm)
            vm_ect[vm.id] += ect
            vm_ecc[vm.id] += ect * vm.cost_per_time

        makespan = max(vm_ect.values())
        cost = sum(vm_ecc.values())

        avg = sum(vm_ect.values()) / len(vms)
        va = sum((vm_ect[vid] - avg)**2 for vid in vm_ect) / len(vms)

        util = sum(vm_ect.values()) / (makespan * len(vms))

        return makespan, cost, va, util

    def _compute_vm_stats(self, tasks, vms, alloc):
        vm_index = {vm.id: vm for vm in vms}
        vm_ect = {vm.id: 0 for vm in vms}

        for t in tasks:
            vm = vm_index[alloc[t.id]]
            vm_ect[vm.id] += self.ect(t, vm)

        avg = sum(vm_ect.values()) / len(vms)
        va = sum((vm_ect[v] - avg)**2 for v in vm_ect) / len(vms)

        vm_ecc = {vm.id: vm_ect[vm.id] * vm.cost_per_time for vm in vms}

        return vm_ect, vm_ecc, avg, va

    def allocation_phase(self, tasks, vms, cand, first):
        last = dict(first)
        vm_index = {vm.id: vm for vm in vms}

        for _ in range(self.max_rounds):
            vm_ect, _, avg, _ = self._compute_vm_stats(tasks, vms, last)
            threshold = avg * self.load_threshold_factor

            new = {}
            record = {vm.id: 0 for vm in vms}

            for t in tasks:
                candidates = cand[t.id]
                vm_id = last[t.id]

                if not candidates:
                    new[t.id] = vm_id
                    record[vm_id] += self.ect(t, vm_index[vm_id])
                    continue

                current_load = self.ect(t, vm_index[vm_id]) + record[vm_id]

                if current_load > threshold:
                    vm_id = min(candidates, key=lambda vid: self.ect(
                        t, vm_index[vid]) + record[vid])

                new[t.id] = vm_id
                record[vm_id] += self.ect(t, vm_index[vm_id])

            if new == last:
                break
            last = new

        return last

    def schedule(self, tasks, vms):
        cand, first = self.matching_phase(tasks, vms)
        return self.allocation_phase(tasks, vms, cand, first)


# ============================================================
# ADAPTIVE MMA
# ============================================================

class AdaptiveMMAScheduler(MMAScheduler):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.time_weight = 0.5
        self.cost_weight = 0.5
        self.gain_threshold = 0.001
        self.patience = 3

    def allocation_phase(self, tasks, vms, cand, first):
        last = dict(first)
        best_va = float("inf")
        fails = 0

        vm_index = {vm.id: vm for vm in vms}

        for _ in range(self.max_rounds):
            vm_ect, vm_ecc, avg, va = self._compute_vm_stats(tasks, vms, last)
            threshold = avg * self.load_threshold_factor

            new = {}
            record = {vm.id: 0 for vm in vms}

            for t in tasks:
                cands = cand[t.id]

                if not cands:
                    new[t.id] = last[t.id]
                    record[last[t.id]] += self.ect(t, vm_index[last[t.id]])
                    continue

                last_vm = last[t.id]
                overloaded = (
                    self.ect(t, vm_index[last_vm]) + record[last_vm]) > threshold

                if overloaded:
                    best = min(cands, key=lambda vid: self.ect(
                        t, vm_index[vid]) + record[vid])
                else:
                    best = last_vm
                    best_gain = 0

                    for vid in cands:
                        if vid == last_vm:
                            continue

                        old_vm = vm_index[last_vm]
                        new_vm = vm_index[vid]

                        gain = (
                            self.time_weight * (self.ect(t, old_vm) - self.ect(t, new_vm)) +
                            self.cost_weight *
                            (self.ecc(t, old_vm) - self.ecc(t, new_vm))
                        )
                        if gain > self.gain_threshold and gain > best_gain:
                            best_gain = gain
                            best = vid

                new[t.id] = best
                record[best] += self.ect(t, vm_index[best])

            _, _, _, new_va = self._compute_vm_stats(tasks, vms, new)

            if new_va < best_va:
                best_va = new_va
                fails = 0
            else:
                fails += 1

            if fails >= self.patience:
                break

            last = new

        return last


# ============================================================
# ENVIRONMENT BUILDER
# ============================================================

def build_example_environment(num_tasks=15, num_vms=5, rng=None):
    rng = rng or random.Random()

    vms = []
    for i in range(num_vms):
        vms.append(VM(
            id=i,
            cpu_cap=rng.uniform(0.3, 1.0),
            ram_cap=rng.uniform(0.3, 1.0),
            bw_cap=rng.uniform(0.3, 1.0),
            disk_cap=rng.uniform(0.3, 1.0),
            reliability=rng.uniform(0.9, 0.999),
            security_level=rng.uniform(1.0, 1.5),
            mips=rng.uniform(500, 2000),
            cost_per_time=rng.uniform(0.01, 0.05)
        ))

    tasks = []
    for i in range(num_tasks):
        tasks.append(Task(
            id=i,
            workload_mi=rng.uniform(1000, 10000),
            cpu_req=rng.uniform(0.1, 0.8),
            ram_req=rng.uniform(0.1, 0.8),
            bw_req=rng.uniform(0.1, 0.8),
            disk_req=rng.uniform(0.1, 0.8),
            reliability_req=rng.uniform(0.9, 0.97),
            security_req=rng.uniform(0.9, 1.0),
            priority=rng.randint(1, 3)
        ))

    return tasks, vms


# ============================================================
# RUN EXPERIMENTS
# ============================================================

def run_experiments(num_trials=6, num_tasks=50, num_vms=10):
    rows = []
    seeds = [123, 456, 789, 101112, 202223, 303344]

    for i in range(num_trials):
        seed = seeds[i]
        print(f"Running trial {i+1}/{num_trials} (seed={seed})")

        tasks, vms = build_example_environment(
            num_tasks=num_tasks,
            num_vms=num_vms,
            rng=random.Random(seed)
        )

        baseline = MMAScheduler(max_rounds=50, rng=np.random.default_rng(seed))
        alloc_base = baseline.schedule(tasks, vms)
        base_metrics = baseline.compute_stats(tasks, vms, alloc_base)

        adaptive = AdaptiveMMAScheduler(
            max_rounds=50, rng=np.random.default_rng(seed))
        alloc_adapt = adaptive.schedule(tasks, vms)
        adapt_metrics = adaptive.compute_stats(tasks, vms, alloc_adapt)

        rows.append([
            seed,
            *base_metrics,
            *adapt_metrics
        ])

    return rows


# ============================================================
# PLOT RESULTS
# ============================================================

def plot_results(rows, folder="mma_plots"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    rows = np.array(rows)
    base_makespan = rows[:, 1]
    adapt_makespan = rows[:, 5]

    base_cost = rows[:, 2]
    adapt_cost = rows[:, 6]

    base_va = rows[:, 3]
    adapt_va = rows[:, 7]

    def paired_plot(base, adapt, ylabel, filename):
        plt.figure(figsize=(7, 5))
        for i in range(len(base)):
            plt.plot([0, 1], [base[i], adapt[i]], marker="o")
        plt.xticks([0, 1], ["Baseline", "Adaptive"])
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.title(f"{ylabel}: Baseline vs Adaptive")
        plt.tight_layout()
        plt.savefig(os.path.join(folder, filename))
        plt.close()

    paired_plot(base_makespan, adapt_makespan,
                "Makespan", "makespan_paired.png")
    paired_plot(base_cost, adapt_cost, "Cost", "cost_paired.png")
    paired_plot(base_va, adapt_va, "Variance (VA)", "va_paired.png")

    print(f"Plots saved in: {folder}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    rows = run_experiments(num_trials=6, num_tasks=50, num_vms=10)
    plot_results(rows)
