"""
Python implementation of the MMA algorithm from:

Q.-H. Zhu et al., "Task Scheduling for Multi-Cloud Computing
Subject to Security and Reliability Constraints", IEEE/CAA JAS, 2021.

This implementation captures the core logic:

  1. Matching Phase
     - Filter VMs by security and reliability constraints
     - Compute Bayesian-style matching degree between task requirements
       and VM capacities (CPU, RAM, bandwidth, disk)
     - Produce FirstAllocate: initial VM assignment per task

  2. Multi-Round Allocation Phase
     - Iteratively reassign tasks to reduce:
         * avgECT (average completion time)
         * VA (variance of completion times)
         * ECC (total execution cost)
     - Stops when no further improvement is possible or max_rounds reached.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import math
import random



@dataclass
class Task:
    id: int
    workload_mi: float  # workload in million instructions (MI)
    cpu_req: float      # normalized CPU demand (0..1 or similar scale)
    ram_req: float      # normalized RAM demand
    bw_req: float       # normalized bandwidth demand
    disk_req: float     # normalized disk demand
    reliability_req: float  # required reliability (0..1)
    security_req: float     # required security level (>= this)
    priority: int = 1       # lower = higher priority

    def resource_vector(self) -> Tuple[float, float, float, float, float]:
        return (self.cpu_req, self.ram_req, self.bw_req, self.disk_req, self.reliability_req)


@dataclass
class VM:
    id: int
    cpu_cap: float      # normalized CPU capacity
    ram_cap: float      # normalized RAM capacity
    bw_cap: float       # normalized bandwidth capacity
    disk_cap: float     # normalized disk capacity
    reliability: float  # reliability (0..1)
    security_level: float  # security level >= 1 means meets baseline
    mips: float         # computing power in MIPS for ECT
    cost_per_time: float  # cost per time unit

    def resource_vector(self) -> Tuple[float, float, float, float, float]:
        return (self.cpu_cap, self.ram_cap, self.bw_cap, self.disk_cap, self.reliability)


class MMAScheduler:

    def __init__(
        self,
        max_rounds: int = 100,
        load_threshold_factor: float = 0.75,
        eps_improvement: float = 1e-9,
        rng: Optional[random.Random] = None,
    ):

        self.max_rounds = max_rounds # Maximum number of allocation rounds in Phase 2
        self.load_threshold_factor = load_threshold_factor #Threshold factor used in Algorithm 4 (avgECT * factor)
        self.eps_improvement = eps_improvement #  Minimum improvement required to continue iterating
        self.rng = rng or random.Random() # Optional random generator (for deterministic behavior in ties)


    @staticmethod
    def ect(task: Task, vm: VM) -> float:
        if vm.mips <= 0:
            raise ValueError("VM mips must be > 0")
        return task.workload_mi / vm.mips

    @staticmethod
    def ecc(task: Task, vm: VM) -> float:
        return MMAScheduler.ect(task, vm) * vm.cost_per_time


    @staticmethod
    def _passes_security_and_reliability(task: Task, vm: VM) -> bool:
        if vm.security_level < task.security_req:
            return False
        if vm.reliability < task.reliability_req:
            return False
        return True

    @staticmethod
    def _feature_match_prob(task_val: float, vm_val: float) -> float:
        if vm_val <= 0:
            return 0.0
        if task_val > vm_val:
            return 0.0
        diff_ratio = abs(task_val - vm_val) / vm_val
        return 1.0 / (1.0 + diff_ratio)

    def _matching_degree(self, task: Task, vm: VM) -> float:
        t_vec = task.resource_vector()
        v_vec = vm.resource_vector()

        probs = []
        for alpha in range(4):  # CPU, RAM, BW, DISK only
            p = self._feature_match_prob(t_vec[alpha], v_vec[alpha])
            probs.append(p)

        prod = 1.0
        for p in probs:
            prod *= p
        return prod

    def matching_phase(
        self,
        tasks: List[Task],
        vms: List[VM],
    ) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
        """
        Phase 1: Matching.

        Returns:
            candidate_set: task_id -> list of candidate VM IDs (that pass rigid QoS).
            first_allocate: task_id -> chosen VM ID (highest matching degree, tie-breaking by load).
        """
        vm_index: Dict[int, VM] = {vm.id: vm for vm in vms}

        candidate_set: Dict[int, List[int]] = {}
        for task in tasks:
            candidates = [
                vm.id
                for vm in vms
                if self._passes_security_and_reliability(task, vm)
            ]
            candidate_set[task.id] = candidates

        ordered_tasks = sorted(tasks, key=lambda t: (t.priority, t.workload_mi))

        vm_loads: Dict[int, float] = {vm.id: 0.0 for vm in vms}

        first_allocate: Dict[int, int] = {}

        for task in ordered_tasks:
            candidates = candidate_set[task.id]
            if not candidates:
                best_vm = max(vms, key=lambda vm: vm.security_level)
                first_allocate[task.id] = best_vm.id
                vm_loads[best_vm.id] += self.ect(task, best_vm)
                continue


            best_vm_id = None
            best_match = -1.0

            for vm_id in candidates:
                vm = vm_index[vm_id]
                match_degree = self._matching_degree(task, vm)

                if match_degree <= 0:
                    continue

                if match_degree > best_match:
                    best_match = match_degree
                    best_vm_id = vm_id
                elif math.isclose(match_degree, best_match, rel_tol=1e-9):
                    current_vm = vm_index[best_vm_id] if best_vm_id is not None else None
                    cand_time = self.ect(task, vm) + vm_loads[vm_id]
                    if current_vm is None:
                        best_vm_id = vm_id
                    else:
                        best_time = self.ect(task, current_vm) + vm_loads[best_vm_id]
                        if cand_time < best_time:
                            best_vm_id = vm_id

            if best_vm_id is None:
                best_vm_id = min(candidates, key=lambda vid: vm_loads[vid])

            first_allocate[task.id] = best_vm_id
            vm_loads[best_vm_id] += self.ect(task, vm_index[best_vm_id])

        return candidate_set, first_allocate


    def _compute_vm_stats(
        self,
        tasks: List[Task],
        vms: List[VM],
        allocation: Dict[int, int],
    ) -> Tuple[Dict[int, float], Dict[int, float], float, float]:
        """
        Compute:

        - vm_ect: VM ID -> total estimated completion time
        - vm_ecc: VM ID -> cost for its tasks
        - avg_ect: average ECT across all VMs
        - va: variance of ECT across VMs
        """
        vm_index = {vm.id: vm for vm in vms}
        vm_ect = {vm.id: 0.0 for vm in vms}
        vm_ecc = {vm.id: 0.0 for vm in vms}

        for task in tasks:
            vm_id = allocation.get(task.id)
            if vm_id is None:
                continue
            vm = vm_index[vm_id]
            t_ect = self.ect(task, vm)
            vm_ect[vm_id] += t_ect
            vm_ecc[vm_id] += t_ect * vm.cost_per_time

        h = len(vms)
        total_ect_sum = sum(vm_ect.values())
        avg_ect = total_ect_sum / h if h > 0 else 0.0

        va = 0.0
        for vm_id in vm_ect:
            diff = vm_ect[vm_id] - avg_ect
            va += diff * diff
        va = va / h if h > 0 else 0.0

        return vm_ect, vm_ecc, avg_ect, va

    def allocation_phase(
        self,
        tasks: List[Task],
        vms: List[VM],
        candidate_set: Dict[int, List[int]],
        first_allocate: Dict[int, int],
    ) -> Dict[int, int]:
        """
        Phase 2: Multi-round iterative allocation.

        Returns:
            best_allocation: task_id -> VM ID
        """
        vm_index = {vm.id: vm for vm in vms}
        task_index = {t.id: t for t in tasks}

        last_allocate: Dict[int, int] = dict(first_allocate)

        vm_ect, vm_ecc, avg_ect, va = self._compute_vm_stats(tasks, vms, last_allocate)
        total_ecc = sum(vm_ecc.values())

        ordered_task_ids = [
            t.id for t in sorted(tasks, key=lambda t: (t.priority, t.workload_mi))
        ]

        best_allocation = dict(last_allocate)
        best_va = va
        best_ecc = total_ecc

        for round_idx in range(self.max_rounds):
            record_load: Dict[int, float] = {vm.id: 0.0 for vm in vms}
            new_allocate: Dict[int, int] = {}

            threshold = avg_ect * self.load_threshold_factor

            for tid in ordered_task_ids:
                task = task_index[tid]
                candidates = candidate_set.get(tid, [])
                if not candidates:
                    vm_id = last_allocate.get(tid)
                    if vm_id is not None:
                        new_allocate[tid] = vm_id
                        record_load[vm_id] += self.ect(task, vm_index[vm_id])
                    continue

                if len(candidates) == 1:
                    vm_id = candidates[0]
                    new_allocate[tid] = vm_id
                    record_load[vm_id] += self.ect(task, vm_index[vm_id])
                    continue

                last_vm_id = last_allocate.get(tid)
                if last_vm_id is None:
                    overloaded = True
                    last_vm_runtime = float("inf")
                else:
                    last_vm = vm_index[last_vm_id]
                    last_vm_runtime = self.ect(task, last_vm) + record_load[last_vm_id]
                    overloaded = last_vm_runtime > threshold

                if overloaded:
                    best_vm_id = None
                    best_load = float("inf")
                    for vm_id in candidates:
                        vm = vm_index[vm_id]
                        cand_time = self.ect(task, vm) + record_load[vm_id]
                        if cand_time < best_load:
                            best_load = cand_time
                            best_vm_id = vm_id
                    chosen_vm_id = best_vm_id
                else:
                    # Keep previous VM
                    chosen_vm_id = last_vm_id

                new_allocate[tid] = chosen_vm_id
                record_load[chosen_vm_id] += self.ect(task, vm_index[chosen_vm_id])

            vm_ect_new, vm_ecc_new, avg_ect_new, va_new = self._compute_vm_stats(
                tasks, vms, new_allocate
            )
            total_ecc_new = sum(vm_ecc_new.values())


            improved_va = va - va_new
            improved_ecc = total_ecc - total_ecc_new

            if improved_va <= self.eps_improvement and improved_ecc <= self.eps_improvement:
                break

            last_allocate = new_allocate
            vm_ect, vm_ecc = vm_ect_new, vm_ecc_new
            avg_ect, va, total_ecc = avg_ect_new, va_new, total_ecc_new

            if va < best_va or (
                math.isclose(va, best_va, rel_tol=1e-9) and total_ecc < best_ecc
            ):
                best_allocation = dict(last_allocate)
                best_va = va
                best_ecc = total_ecc

        return best_allocation


    def schedule(self, tasks: List[Task], vms: List[VM]) -> Dict[int, int]:
        """
        Run full MMA (Matching + Multi-Round Allocation) and return:

            allocation: task_id -> vm_id
        """
        candidate_set, first_allocate = self.matching_phase(tasks, vms)
        final_allocate = self.allocation_phase(tasks, vms, candidate_set, first_allocate)
        return final_allocate



def build_example_environment(
    num_tasks: int = 20,
    num_vms: int = 6,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Task], List[VM]]:
    # Build a toy environment to test the MMA scheduler. This is just for demo; in a real experiment you would define tasks/VMs according to your scenario or dataset.
    rng = rng or random.Random(42)

    vms: List[VM] = []
    for i in range(num_vms):
        cpu = rng.uniform(0.3, 1.0)
        ram = rng.uniform(0.3, 1.0)
        bw = rng.uniform(0.3, 1.0)
        disk = rng.uniform(0.3, 1.0)
        rel = rng.uniform(0.90, 0.999)
        sec = rng.uniform(1.0, 1.5)  # >= 1 means at/above baseline
        mips = rng.uniform(500, 2000)
        cost = rng.uniform(0.01, 0.05)
        vms.append(
            VM(
                id=i,
                cpu_cap=cpu,
                ram_cap=ram,
                bw_cap=bw,
                disk_cap=disk,
                reliability=rel,
                security_level=sec,
                mips=mips,
                cost_per_time=cost,
            )
        )

    tasks: List[Task] = []
    for i in range(num_tasks):
        # Smaller tasks
        workload = rng.uniform(1_000, 10_000)  
        cpu = rng.uniform(0.1, 0.8)
        ram = rng.uniform(0.1, 0.8)
        bw = rng.uniform(0.1, 0.8)
        disk = rng.uniform(0.1, 0.8)
        rel_req = rng.uniform(0.9, 0.97)  
        sec_req = rng.uniform(0.9, 1.0)    
        priority = rng.randint(1, 3)
        tasks.append(
            Task(
                id=i,
                workload_mi=workload,
                cpu_req=cpu,
                ram_req=ram,
                bw_req=bw,
                disk_req=disk,
                reliability_req=rel_req,
                security_req=sec_req,
                priority=priority,
            )
        )

    return tasks, vms


if __name__ == "__main__":
    # Demo run of MMA
    rng = random.Random(123)
    tasks, vms = build_example_environment(num_tasks=15, num_vms=5, rng=rng)

    scheduler = MMAScheduler(max_rounds=50, rng=rng)
    allocation = scheduler.schedule(tasks, vms)

    # Print basic results
    print("=== MMA Allocation Result ===")
    for t in sorted(tasks, key=lambda t: t.id):
        vm_id = allocation.get(t.id)
        print(f"Task {t.id:02d} (prio={t.priority}) -> VM {vm_id}")
