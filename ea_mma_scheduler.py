"""
Python implementation of the MMA algorithm and an Energy-Aware MMA (EA-MMA)
extension based on:

Q.-H. Zhu et al., "Task Scheduling for Multi-Cloud Computing
Subject to Security and Reliability Constraints", IEEE/CAA JAS, 2021.

- MMAScheduler:
    Original MMA logic (matching + multi-round allocation)
    optimizing time & cost (via load balancing and cost reduction).

- EnergyAwareMMAScheduler:
    Extends MMAScheduler by adding an energy term:
        F = α * T + β * C + γ * E
    where:
        T = average completion time (avgECT)
        C = total cost
        E = total energy consumption of all VMs
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import math
import random


# -------------------------
# Data Models
# -------------------------

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
        return (
            self.cpu_req,
            self.ram_req,
            self.bw_req,
            self.disk_req,
            self.reliability_req,
        )


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
    power_watts: float = 150.0  # average power draw in Watts (for EA-MMA)

    def resource_vector(self) -> Tuple[float, float, float, float, float]:
        return (
            self.cpu_cap,
            self.ram_cap,
            self.bw_cap,
            self.disk_cap,
            self.reliability,
        )


# -------------------------
# Original MMA Scheduler
# -------------------------

class MMAScheduler:
    """
    Baseline MMA scheduler (original algorithm).
    """

    def __init__(
        self,
        max_rounds: int = 100,
        load_threshold_factor: float = 0.75,
        eps_improvement: float = 1e-9,
        rng: Optional[random.Random] = None,
    ):
        # Maximum number of allocation rounds in Phase 2
        self.max_rounds = max_rounds
        # Threshold factor used in Algorithm 4 (avgECT * factor)
        self.load_threshold_factor = load_threshold_factor
        # Minimum improvement required to continue iterating
        self.eps_improvement = eps_improvement
        # Optional random generator (for deterministic behavior in ties)
        self.rng = rng or random.Random()

    # --- Time & Cost ---

    @staticmethod
    def ect(task: Task, vm: VM) -> float:
        """Estimated completion time for this task on this VM."""
        if vm.mips <= 0:
            raise ValueError("VM mips must be > 0")
        return task.workload_mi / vm.mips

    @staticmethod
    def ecc(task: Task, vm: VM) -> float:
        """Estimated execution cost for this task on this VM."""
        return MMAScheduler.ect(task, vm) * vm.cost_per_time

    # --- Rigid Constraints (Security + Reliability) ---

    @staticmethod
    def _passes_security_and_reliability(task: Task, vm: VM) -> bool:
        if vm.security_level < task.security_req:
            return False
        if vm.reliability < task.reliability_req:
            return False
        return True

    # --- Matching Degree ---

    @staticmethod
    def _feature_match_prob(task_val: float, vm_val: float) -> float:
        """
        Bayesian-like matching function:
        P(t_alpha | y_alpha) = 1 / (1 + |t - y| / y), if t <= y; 0 otherwise.
        """
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
        # Only CPU, RAM, BW, DISK (4 dimensions) used for matching degree
        for alpha in range(4):
            p = self._feature_match_prob(t_vec[alpha], v_vec[alpha])
            probs.append(p)

        prod = 1.0
        for p in probs:
            prod *= p
        return prod

    # --- Phase 1: Matching ---

    def matching_phase(
        self,
        tasks: List[Task],
        vms: List[VM],
    ) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
        """
        Phase 1: Matching (Algorithms 1–3).

        Returns:
            candidate_set: task_id -> list of candidate VM IDs (passing rigid QoS).
            first_allocate: task_id -> chosen VM ID (highest matching degree,
                            tie-broken by lower current load).
        """
        vm_index: Dict[int, VM] = {vm.id: vm for vm in vms}

        # Step 1: Build candidate set that passes security + reliability
        candidate_set: Dict[int, List[int]] = {}
        for task in tasks:
            candidates = [
                vm.id
                for vm in vms
                if self._passes_security_and_reliability(task, vm)
            ]
            candidate_set[task.id] = candidates

        # Sort tasks by priority (low number = high priority), then by workload
        ordered_tasks = sorted(tasks, key=lambda t: (t.priority, t.workload_mi))

        # Keep track of VM loads during initial assignment
        vm_loads: Dict[int, float] = {vm.id: 0.0 for vm in vms}

        first_allocate: Dict[int, int] = {}

        for task in ordered_tasks:
            candidates = candidate_set[task.id]

            # If no candidate passes rigid constraints, fall back to "best" VM by security level
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
                    # Tie-break by smaller completion time (ECT + current load)
                    current_vm = vm_index[best_vm_id] if best_vm_id is not None else None
                    cand_time = self.ect(task, vm) + vm_loads[vm_id]
                    if current_vm is None:
                        best_vm_id = vm_id
                    else:
                        best_time = self.ect(task, current_vm) + vm_loads[best_vm_id]
                        if cand_time < best_time:
                            best_vm_id = vm_id

            # If everything failed (all match_degree 0), pick least loaded VM among candidates
            if best_vm_id is None:
                best_vm_id = min(candidates, key=lambda vid: vm_loads[vid])

            first_allocate[task.id] = best_vm_id
            vm_loads[best_vm_id] += self.ect(task, vm_index[best_vm_id])

        return candidate_set, first_allocate

    # --- VM Stats (Time & Cost) ---

    def _compute_vm_stats(
        self,
        tasks: List[Task],
        vms: List[VM],
        allocation: Dict[int, int],
    ) -> Tuple[Dict[int, float], Dict[int, float], float, float]:
        """
        Compute:

            vm_ect: VM ID -> total estimated completion time
            vm_ecc: VM ID -> cost for its tasks
            avg_ect: average ECT across all VMs
            va: variance of ECT across VMs
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

    # --- Phase 2: Multi-Round Allocation ---

    def allocation_phase(
        self,
        tasks: List[Task],
        vms: List[VM],
        candidate_set: Dict[int, List[int]],
        first_allocate: Dict[int, int],
    ) -> Dict[int, int]:
        """
        Phase 2: Multi-round iterative allocation (Algorithm 4).

        Returns:
            best_allocation: task_id -> VM ID
        """
        vm_index = {vm.id: vm for vm in vms}
        task_index = {t.id: t for t in tasks}

        last_allocate: Dict[int, int] = dict(first_allocate)

        vm_ect, vm_ecc, avg_ect, va = self._compute_vm_stats(tasks, vms, last_allocate)
        total_ecc = sum(vm_ecc.values())

        # Order tasks by priority and workload
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

                # Only one candidate => fixed
                if len(candidates) == 1:
                    vm_id = candidates[0]
                    new_allocate[tid] = vm_id
                    record_load[vm_id] += self.ect(task, vm_index[vm_id])
                    continue

                # Check if currently assigned VM is overloaded
                last_vm_id = last_allocate.get(tid)
                if last_vm_id is None:
                    overloaded = True
                    last_vm_runtime = float("inf")
                else:
                    last_vm = vm_index[last_vm_id]
                    last_vm_runtime = self.ect(task, last_vm) + record_load[last_vm_id]
                    overloaded = last_vm_runtime > threshold

                if overloaded:
                    # Try to find better VM among candidates with minimal load
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

            # Compute new stats
            vm_ect_new, vm_ecc_new, avg_ect_new, va_new = self._compute_vm_stats(
                tasks, vms, new_allocate
            )
            total_ecc_new = sum(vm_ecc_new.values())

            improved_va = va - va_new
            improved_ecc = total_ecc - total_ecc_new

            # Stop if no further improvement in variance or cost
            if improved_va <= self.eps_improvement and improved_ecc <= self.eps_improvement:
                break

            # Update for next round
            last_allocate = new_allocate
            vm_ect, vm_ecc = vm_ect_new, vm_ecc_new
            avg_ect, va, total_ecc = avg_ect_new, va_new, total_ecc_new

            # Track best allocation
            if va < best_va or (
                math.isclose(va, best_va, rel_tol=1e-9) and total_ecc < best_ecc
            ):
                best_allocation = dict(last_allocate)
                best_va = va
                best_ecc = total_ecc

        return best_allocation

    # --- Public API ---

    def schedule(self, tasks: List[Task], vms: List[VM]) -> Dict[int, int]:
        """
        Run full MMA (Matching + Multi-Round Allocation).

        Returns:
            allocation: task_id -> vm_id
        """
        candidate_set, first_allocate = self.matching_phase(tasks, vms)
        final_allocate = self.allocation_phase(tasks, vms, candidate_set, first_allocate)
        return final_allocate


# -------------------------
# Energy-Aware MMA (EA-MMA)
# -------------------------

class EnergyAwareMMAScheduler(MMAScheduler):
    """
    Energy-Aware MMA (EA-MMA):

        - Keeps the same matching phase as MMA.
        - In the allocation phase, considers energy as well.
        - Objective:
                F = α * T + β * C + γ * E

          where:
                T = average completion time (avgECT)
                C = total cost
                E = total energy consumption of all VMs
    """

    def __init__(
        self,
        alpha: float = 1.0,   # weight for time
        beta: float = 1.0,    # weight for cost
        gamma: float = 1.0,   # weight for energy
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _compute_vm_stats_with_energy(
        self,
        tasks: List[Task],
        vms: List[VM],
        allocation: Dict[int, int],
    ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], float, float, float]:
        """
        Extends the parent stats with energy:

            vm_ect: VM ID -> total estimated completion time
            vm_ecc: VM ID -> total cost on that VM
            vm_energy: VM ID -> total energy on that VM
            avg_ect: average ECT across VMs
            va: variance of ECT
            total_energy: sum of all vm_energy
        """
        vm_ect, vm_ecc, avg_ect, va = super()._compute_vm_stats(tasks, vms, allocation)

        vm_index = {vm.id: vm for vm in vms}
        vm_energy = {vm.id: 0.0 for vm in vms}
        for vm_id in vm_ect:
            time_on_vm = vm_ect[vm_id]
            vm = vm_index[vm_id]
            # Simple energy model: E = P_avg * t_exec
            vm_energy[vm_id] = time_on_vm * vm.power_watts

        total_energy = sum(vm_energy.values())
        return vm_ect, vm_ecc, vm_energy, avg_ect, va, total_energy

    def allocation_phase(
        self,
        tasks: List[Task],
        vms: List[VM],
        candidate_set: Dict[int, List[int]],
        first_allocate: Dict[int, int],
    ) -> Dict[int, int]:
        """
        Phase 2 for EA-MMA:

        Iteratively reallocate tasks to minimize
            F = α * T + β * C + γ * E
        """
        vm_index = {vm.id: vm for vm in vms}
        task_index = {t.id: t for t in tasks}

        last_allocate: Dict[int, int] = dict(first_allocate)

        vm_ect, vm_ecc, vm_energy, avg_ect, va, total_energy = (
            self._compute_vm_stats_with_energy(tasks, vms, last_allocate)
        )
        total_ecc = sum(vm_ecc.values())

        # Scalar objective
        def objective(avg_ect_val: float, total_cost_val: float, total_energy_val: float) -> float:
            return (
                self.alpha * avg_ect_val
                + self.beta * total_cost_val
                + self.gamma * total_energy_val
            )

        current_F = objective(avg_ect, total_ecc, total_energy)

        # Order tasks by priority and workload
        ordered_task_ids = [
            t.id for t in sorted(tasks, key=lambda t: (t.priority, t.workload_mi))
        ]

        best_allocation = dict(last_allocate)
        best_F = current_F

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

                # Single candidate: forced allocation
                if len(candidates) == 1:
                    vm_id = candidates[0]
                    new_allocate[tid] = vm_id
                    record_load[vm_id] += self.ect(task, vm_index[vm_id])
                    continue

                # Check if the previous VM is overloaded relative to avgECT
                last_vm_id = last_allocate.get(tid)
                if last_vm_id is None:
                    overloaded = True
                    last_vm_runtime = float("inf")
                else:
                    last_vm = vm_index[last_vm_id]
                    last_vm_runtime = self.ect(task, last_vm) + record_load[last_vm_id]
                    overloaded = last_vm_runtime > threshold

                if overloaded:
                    # Re-choose VM among candidates: here we use "future" load
                    # defined by record_load + ect(task, vm)
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
                    # Keep previous assignment
                    chosen_vm_id = last_vm_id

                new_allocate[tid] = chosen_vm_id
                record_load[chosen_vm_id] += self.ect(task, vm_index[chosen_vm_id])

            # Compute new stats and objective
            (
                vm_ect_new,
                vm_ecc_new,
                vm_energy_new,
                avg_ect_new,
                va_new,
                total_energy_new,
            ) = self._compute_vm_stats_with_energy(tasks, vms, new_allocate)

            total_ecc_new = sum(vm_ecc_new.values())
            new_F = objective(avg_ect_new, total_ecc_new, total_energy_new)

            # If we didn't improve F enough, stop
            if current_F - new_F <= self.eps_improvement:
                break

            # Accept new allocation
            last_allocate = new_allocate
            vm_ect, vm_ecc = vm_ect_new, vm_ecc_new
            vm_energy = vm_energy_new
            avg_ect, va, total_energy = avg_ect_new, va_new, total_energy_new
            total_ecc = total_ecc_new
            current_F = new_F

            # Update best solution
            if new_F < best_F:
                best_F = new_F
                best_allocation = dict(last_allocate)

        return best_allocation

    # schedule() is inherited from MMAScheduler and works the same.


# -------------------------
# Example Environment Builder
# -------------------------

def build_example_environment(
    num_tasks: int = 20,
    num_vms: int = 6,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Task], List[VM]]:
    """
    Build a toy environment to test the schedulers.
    In a real experiment you would define tasks/VMs from your dataset or scenario.
    """
    rng = rng or random.Random(42)

    # Build VMs
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
        power = rng.uniform(80, 250)  # watts (simulated)
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
                power_watts=power,
            )
        )

    # Build Tasks
    tasks: List[Task] = []
    for i in range(num_tasks):
        workload = rng.uniform(1_000, 10_000)  # MI
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


# -------------------------
# Demo main (optional)
# -------------------------

if __name__ == "__main__":
    rng = random.Random(123)
    tasks, vms = build_example_environment(num_tasks=15, num_vms=5, rng=rng)

    # Baseline MMA
    mma = MMAScheduler(max_rounds=50, rng=rng)
    mma_alloc = mma.schedule(tasks, vms)
    print("=== Baseline MMA Allocation ===")
    for t in sorted(tasks, key=lambda t: t.id):
        print(f"Task {t.id:02d} -> VM {mma_alloc.get(t.id)}")

    # Energy-Aware MMA
    ea_mma = EnergyAwareMMAScheduler(
        alpha=1.0, beta=1.0, gamma=1.0, max_rounds=50, rng=rng
    )
    ea_alloc = ea_mma.schedule(tasks, vms)
    print("\n=== Energy-Aware MMA Allocation ===")
    for t in sorted(tasks, key=lambda t: t.id):
        print(f"Task {t.id:02d} -> VM {ea_alloc.get(t.id)}")