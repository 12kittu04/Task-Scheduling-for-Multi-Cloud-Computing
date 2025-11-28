from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import math
import random


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
# BASELINE MMA SCHEDULER
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

    def _feature_match_prob(self, task_val, vm_val):
        if vm_val <= 0:
            return 0
        if task_val > vm_val:
            return 0
        return 1 / (1 + abs(task_val - vm_val) / vm_val)

    def _matching_degree(self, task: Task, vm: VM):
        t, v = task.resource_vector(), vm.resource_vector()
        p = [self._feature_match_prob(t[i], v[i]) for i in range(4)]
        prod = 1
        for x in p:
            prod *= x
        return prod

    def matching_phase(self, tasks: List[Task], vms: List[VM]):
        vm_index = {vm.id: vm for vm in vms}

        candidate_set = {
            t.id: [
                vm.id for vm in vms if self._passes_security_and_reliability(t, vm)]
            for t in tasks
        }

        ordered_tasks = sorted(
            tasks, key=lambda t: (t.priority, t.workload_mi))
        vm_loads = {vm.id: 0 for vm in vms}
        first_allocate = {}

        for task in ordered_tasks:
            candidates = candidate_set[task.id]

            if not candidates:
                best_vm = max(vms, key=lambda x: x.security_level)
                first_allocate[task.id] = best_vm.id
                vm_loads[best_vm.id] += self.ect(task, best_vm)
                continue

            best_vm, best_md = None, -1
            for vm_id in candidates:
                vm = vm_index[vm_id]
                md = self._matching_degree(task, vm)
                if md > best_md:
                    best_vm, best_md = vm_id, md

            first_allocate[task.id] = best_vm
            vm_loads[best_vm] += self.ect(task, vm_index[best_vm])

        return candidate_set, first_allocate

    def compute_stats(self, tasks, vms, allocation):
        vm_index = {vm.id: vm for vm in vms}
        vm_ect = {vm.id: 0 for vm in vms}
        vm_ecc = {vm.id: 0 for vm in vms}

        for t in tasks:
            vm_id = allocation[t.id]
            vm = vm_index[vm_id]
            ect = self.ect(t, vm)
            vm_ect[vm_id] += ect
            vm_ecc[vm_id] += ect * vm.cost_per_time

        makespan = max(vm_ect.values())
        total_cost = sum(vm_ecc.values())

        avg_ect = sum(vm_ect.values()) / len(vms)
        va = sum((vm_ect[vm] - avg_ect)**2 for vm in vm_ect) / len(vms)
        utilization = sum(vm_ect.values()) / (makespan * len(vms))

        return makespan, total_cost, va, utilization

    def allocation_phase(self, tasks, vms, candidate_set, first_allocate):
        vm_index = {vm.id: vm for vm in vms}
        last = dict(first_allocate)

        for _ in range(self.max_rounds):
            vm_ect, _, avg_ect, _ = self._compute_vm_stats(tasks, vms, last)
            threshold = avg_ect * self.load_threshold_factor

            new_alloc = {}
            record_load = {vm.id: 0 for vm in vms}

            for t in tasks:
                candidates = candidate_set[t.id]
                if not candidates:
                    vm_id = last[t.id]
                    new_alloc[t.id] = vm_id
                    record_load[vm_id] += self.ect(t, vm_index[vm_id])
                    continue

                vm_id = last[t.id]
                current_load = self.ect(
                    t, vm_index[vm_id]) + record_load[vm_id]

                if current_load > threshold:
                    best_vm = min(candidates, key=lambda vid: self.ect(
                        t, vm_index[vid]) + record_load[vid])
                    vm_id = best_vm

                new_alloc[t.id] = vm_id
                record_load[vm_id] += self.ect(t, vm_index[vm_id])

            if new_alloc == last:
                break
            last = new_alloc

        return last

    def _compute_vm_stats(self, tasks, vms, allocation):
        vm_index = {vm.id: vm for vm in vms}
        vm_ect = {vm.id: 0 for vm in vms}
        vm_ecc = {vm.id: 0 for vm in vms}

        for t in tasks:
            vm = vm_index[allocation[t.id]]
            ect = self.ect(t, vm)
            vm_ect[vm.id] += ect
            vm_ecc[vm.id] += ect * vm.cost_per_time

        h = len(vms)
        avg_ect = sum(vm_ect.values()) / h
        va = sum((vm_ect[vid] - avg_ect)**2 for vid in vm_ect) / h

        return vm_ect, vm_ecc, avg_ect, va

    def schedule(self, tasks, vms):
        cand, first = self.matching_phase(tasks, vms)
        alloc = self.allocation_phase(tasks, vms, cand, first)
        return alloc


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

    def allocation_phase(self, tasks, vms, candidate_set, first_allocate):
        vm_index = {vm.id: vm for vm in vms}
        task_index = {t.id: t for t in tasks}

        last = dict(first_allocate)
        no_improve = 0
        best_va = float("inf")

        for r in range(self.max_rounds):

            vm_ect, vm_ecc, avg_ect, va = self._compute_vm_stats(
                tasks, vms, last)
            threshold = avg_ect * self.load_threshold_factor

            new = {}
            record = {vm.id: 0 for vm in vms}

            for t in tasks:
                candidates = candidate_set[t.id]

                if not candidates:
                    vm_id = last[t.id]
                    new[t.id] = vm_id
                    record[vm_id] += self.ect(t, vm_index[vm_id])
                    continue

                last_vm = last[t.id]
                last_runtime = self.ect(t, vm_index[last_vm]) + record[last_vm]
                overloaded = last_runtime > threshold

                if overloaded:
                    best_vm = min(candidates, key=lambda vid: self.ect(
                        t, vm_index[vid]) + record[vid])
                else:
                    best_vm = last_vm
                    best_gain = 0

                    for vid in candidates:
                        if vid == last_vm:
                            continue
                        old_vm = vm_index[last_vm]
                        new_vm = vm_index[vid]

                        time_old = self.ect(t, old_vm)
                        time_new = self.ect(t, new_vm)

                        cost_old = self.ecc(t, old_vm)
                        cost_new = self.ecc(t, new_vm)

                        gain = (self.time_weight * (time_old - time_new) +
                                self.cost_weight * (cost_old - cost_new))

                        if gain > self.gain_threshold and gain > best_gain:
                            best_gain = gain
                            best_vm = vid

                new[t.id] = best_vm
                record[best_vm] += self.ect(t, vm_index[best_vm])

            vm_ect_new, vm_ecc_new, avg_ect_new, va_new = self._compute_vm_stats(
                tasks, vms, new)

            if va_new < best_va:
                best_va = va_new
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.patience:
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
