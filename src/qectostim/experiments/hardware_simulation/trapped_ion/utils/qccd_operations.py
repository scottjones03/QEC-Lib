
import numpy as np
from typing import (
    Sequence,
    List,
    Optional,
    Callable,
    Any,
    Mapping,
    Set,
    Dict,
    Iterable,
    Union,
    Tuple
)
import abc
from .qccd_nodes import *
from .physics import DEFAULT_CALIBRATION, DEFAULT_FIDELITY_MODEL, CalibrationConstants


class Operation:
    KEY: Operations

    def __init__(
        self,
        run: Callable[[Any], None],
        involvedComponents: Sequence[QCCDComponent],
        **kwargs,
    ) -> None:
        self._run = run
        self._kwargs = dict(kwargs)
        self._involvedIonsForLabel: List[Ion] = []
        self._involvedComponents: List[QCCDComponent] = involvedComponents
        self._addOns = ""
        self._fidelity: float = 1.0
        self._dephasingFidelity: float = 1.0
        self._operationTime: float = 0.0

    def addComponent(self, component: QCCDComponent) -> None:
        self._involvedComponents.append(component)

    @property
    def involvedComponents(self) -> Sequence[QCCDComponent]:
        return self._involvedComponents

    @property
    def color(self) -> str:
        return "lightgreen"

    @property
    def involvedIonsForLabel(self) -> Sequence[Ion]:
        return self._involvedIonsForLabel

    @property
    def label(self) -> str:
        return self.KEY.name + self._addOns

    @property
    @abc.abstractmethod
    def isApplicable(self) -> bool:
        return all(self.KEY in component.allowedOperations for component in self.involvedComponents)
    
    @abc.abstractmethod
    def _checkApplicability(self) -> None:
        for component in self.involvedComponents:
            if self.KEY not in component.allowedOperations:
                raise ValueError(f"Component {component} with index {component.idx} cannot complete {self.KEY.name}")

    @classmethod
    @abc.abstractmethod
    def physicalOperation(cls) -> "Operation": ...

    @abc.abstractmethod
    def calculateFidelity(self) -> None: ...

    @abc.abstractmethod
    def calculateDephasingFidelity(self) -> None: ...

    @abc.abstractmethod
    def calculateOperationTime(self) -> None: ...

    @abc.abstractmethod
    def _generateLabelAddOns(self) -> None: ...

    def run(self) -> None:
        self._checkApplicability()
        self.calculateOperationTime()
        self.calculateFidelity()
        self.calculateDephasingFidelity()
        self._run(())
        self._generateLabelAddOns()

    def dephasingFidelity(self) -> float:
        # Deprecated!
        return self._dephasingFidelity

    def fidelity(self) -> float:
        return self._fidelity
    
    def operationTime(self) -> float:
        return self._operationTime


class CrystalOperation(Operation):
    def __init__(
        self,
        run: Callable[[Any], None],
        involvedComponents: Sequence[QCCDComponent],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents, **kwargs)
        self._trap: Trap = kwargs["trap"]


    def calculateFidelity(self) -> None:
        self._fidelity = 1  # NOISE INCORPORATED INTO HEATING MODEL

    def calculateDephasingFidelity(self) -> None:
        self.calculateOperationTime()
        self._dephasingFidelity = DEFAULT_FIDELITY_MODEL.dephasing_fidelity(self._operationTime)

    @property
    def ionsInfluenced(self) -> Sequence[Ion]:
        return self._trap.ions
    
    def _generateLabelAddOns(self) -> None:
        self._involvedIonsForLabel = list(self._trap.ions)
        self._addOns = ""
        for ion in self._involvedIonsForLabel:
            self._addOns += f" {ion.label}"


class GlobalReconfigurations(Operation):
    KEY = Operations.GLOBAL_RECONFIG

    def __init__(
        self,
        run: Callable[[Any], None],
        involvedComponents: Sequence[QCCDComponent],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
        self._wiseArch: QCCDWiseArch = kwargs['wiseArch']
        self._reconfigTime: float = kwargs['reconfigTime']

    def calculateOperationTime(self) -> None:
        self._operationTime =  self._reconfigTime

    def calculateFidelity(self) -> None:
        self._fidelity = 1  # NOISE INCORPORATED INTO HEATING MODEL

    def calculateDephasingFidelity(self) -> None:
        self.calculateOperationTime()
        self._dephasingFidelity = DEFAULT_FIDELITY_MODEL.dephasing_fidelity(self._operationTime)


    def _generateLabelAddOns(self) -> None:
        self._addOns = f""

    @property
    def isApplicable(self) -> bool:
        return True
    
    def _checkApplicability(self) -> None:
        return True

    @classmethod
    def physicalOperation(
        cls,
        arrangement: Mapping[Trap, Sequence[Ion]],
        wiseArch: QCCDWiseArch,
        oldAssignment: Sequence[Sequence[int]],
        newAssignment: Sequence[Sequence[int]],
        schedule: Optional[List[Dict[str, Any]]] = None,
        initial_placement: bool = False,
    ):
        # DEBUG: trace usage of schedule/sat_schedule for reconfiguration
        # print("[DEBUG GlobalReconfigurations.physicalOperation] initial_placement =", initial_placement)
        # if schedule is None:
        #     print("[DEBUG GlobalReconfigurations.physicalOperation] schedule is None (likely cached block first step)")
        # else:
        #     try:
        #         print("[DEBUG GlobalReconfigurations.physicalOperation] schedule len =", len(schedule))
        #         if schedule and isinstance(schedule[0], list):
        #             print("[DEBUG GlobalReconfigurations.physicalOperation] schedule[0] passes len =", len(schedule[0]))
        #     except Exception as e:
        #         print("[DEBUG GlobalReconfigurations.physicalOperation] error inspecting schedule:", repr(e))

        heatingRates, reconfigTime = cls._runOddEvenReconfig(
            wiseArch,
            arrangement,
            oldAssignment,
            newAssignment,
            sat_schedule=schedule,
            initial_placement=initial_placement,
        )
        reconfigTime = 1e-20 if initial_placement else reconfigTime
        def run():
            for trap in arrangement.keys():
                while trap.ions:
                    trap.removeIon(trap.ions[0])
            for trap, ions in arrangement.items():
                for i, ion in enumerate(ions):
                    trap.addIon(ion, offset=i)
                    if initial_placement: 
                        continue
                    ion.addMotionalEnergy(heatingRates[ion.idx])
        return cls(
            run=lambda _: run(),
            involvedComponents=list(arrangement.keys()),
            wiseArch=wiseArch,
            reconfigTime=reconfigTime

        )



    
    @classmethod
    def _runOddEvenReconfig(
        cls,
        wiseArch: "QCCDWiseArch",
        arrangement: Mapping["Trap", Sequence["Ion"]],
        oldAssignment: Sequence[Sequence[int]],
        newAssignment: Sequence[Sequence[int]],
        ignoreSpectators: bool = False,
        sat_schedule: Optional[List[Dict[str, Any]]] = None,   # NEW: decoded schedule from RC2
        initial_placement: bool = False
    ) -> Tuple[Mapping[int, float], float]:
        # DEBUG: entry into _runOddEvenReconfig
        # print("[DEBUG _runOddEvenReconfig] called")
        # try:
        #     print("  initial_placement =", initial_placement)
        # except Exception:
        #     pass
        # try:
        #     if hasattr(oldAssignment, "shape"):
        #         print("  oldAssignment shape =", oldAssignment.shape)
        #     else:
        #         print("  oldAssignment len =", len(oldAssignment))
        # except Exception:
        #     pass
        # try:
        #     if hasattr(newAssignment, "shape"):
        #         print("  newAssignment shape =", newAssignment.shape)
        #     else:
        #         print("  newAssignment len =", len(newAssignment))
        # except Exception:
        #     pass
        # try:
        #     if sat_schedule is None:
        #         print("  sat_schedule is None (layout-only reconfig; cached block first step)")
        #     else:
        #         print("  sat_schedule type =", type(sat_schedule))
        #         print("  sat_schedule len =", len(sat_schedule))
        #         if sat_schedule and isinstance(sat_schedule[0], list):
        #             print("  sat_schedule[0] passes len =", len(sat_schedule[0]))
        # except Exception as e:
        #     print("[DEBUG _runOddEvenReconfig] error inspecting sat_schedule:", repr(e))

        # Schedule-aware reconfiguration fallback when SAT results are available.
        heatingRates: Dict[int, float] = {}
        for _, ions in arrangement.items():
            for ion in ions:
                heatingRates[ion.idx] = 0.0

        heatingRates: Mapping[int, float] = {}

        # Initialise from arrangement (all ions we know about physically)
        for _, ions in arrangement.items():
            for ion in ions:
                heatingRates[ion.idx] = 0.0

        # ALSO initialise any ions present in the layout matrices A/T
        try:
            A_all = np.array(oldAssignment, dtype=int)
            for ion_id in set(int(x) for x in A_all.flatten()):
                if ion_id not in heatingRates:
                    # These might be spectators / padding but we still give them an entry
                    heatingRates[ion_id] = 0.0
        except Exception as e:
            print("[DEBUG _runOddEvenReconfig] error while initialising heatingRates from assignment:", repr(e))
        timeElapsed = 0.0

        row_swap_time = (
            Move.MOVING_TIME
            + Merge.MERGING_TIME
            + CrystalRotation.ROTATION_TIME
            + Split.SPLITTING_TIME
            + Move.MOVING_TIME
        )
        row_swap_heating = (
            Move.MOVING_TIME * Move.HEATING_RATE
            + Merge.MERGING_TIME * Merge.HEATING_RATE
            + CrystalRotation.ROTATION_TIME * CrystalRotation.HEATING_RATE
            + Split.SPLITTING_TIME * Split.HEATING_RATE
            + Move.MOVING_TIME * Move.HEATING_RATE
        )
        col_swap_time = (2 * JunctionCrossing.CROSSING_TIME) + (
            4 * JunctionCrossing.CROSSING_TIME + Move.MOVING_TIME
        ) * 2
        col_swap_heating_rate = (
            6 * JunctionCrossing.CROSSING_TIME * JunctionCrossing.HEATING_RATE
            + Move.MOVING_TIME * Move.HEATING_RATE
        )

        n = wiseArch.n  # rows
        m = wiseArch.m * wiseArch.k  # full columns (matches CNF m if built that way)
        k = wiseArch.k  # column stride for junction batching

        A = np.array(oldAssignment, dtype=int)  # current layout
        T = np.array(newAssignment, dtype=int)  # target layout

        spectatorIons: List[int] = []
        for ions in arrangement.values():
            spectatorIons.extend([ion.idx for ion in ions if isinstance(ion, SpectatorIon)])

        # ==========================================================
        # Phase A: parallel split (this is arch-specific overhead)
        # ==========================================================
        timeElapsed += Split.SPLITTING_TIME
        for idx in heatingRates.keys():
            heatingRates[idx] += Split.HEATING_RATE * Split.SPLITTING_TIME

        # If we have a SAT schedule, use it directly and SKIP Phases B/C/D.
        if sat_schedule is not None:
            acc_passes = 0
            # print("[DEBUG _runOddEvenReconfig] entering schedule loop, len =", len(sat_schedule))

            for pass_idx, info in enumerate(sat_schedule):
                # try:
                #     print(
                #         f"[DEBUG _runOddEvenReconfig] round {pass_idx}: passes type={type(info)}, "
                #         f"h_swaps={len(info.get('h_swaps', [])) if hasattr(info, 'get') else 'NA'}, "
                #         f"v_swaps={len(info.get('v_swaps', [])) if hasattr(info, 'get') else 'NA'}"
                #     )
                # except Exception as e:
                #     print("[DEBUG _runOddEvenReconfig] error inspecting schedule entry:", repr(e))

                phase = info.get("phase", "H")
                h_swaps = info.get("h_swaps", [])
                v_swaps = info.get("v_swaps", [])

                # NOTE: to stay consistent with the SAT model, we SHOULD NOT
                #       skip swaps just because both ions are spectators.
                #       If you want spectators to be immobile, that must be
                #       encoded in the SAT itself.
                did_any_swap = False

                if phase == "H":
                    # All horizontal swaps in this pass are parallel
                    for (r, c) in h_swaps:
                        a = int(A[r, c])
                        b = int(A[r, c + 1])
                        # Optionally respect ignoreSpectators here, but that
                        # can deviate from the SAT layout. Safer to ignore it
                        # when using a SAT schedule:
                        if ignoreSpectators and (a in spectatorIons and b in spectatorIons):
                            continue
                        # Perform swap
                        A[r, c], A[r, c + 1] = b, a
                        heatingRates[a] += row_swap_heating
                        heatingRates[b] += row_swap_heating
                        did_any_swap = True

                    if did_any_swap:
                        timeElapsed += row_swap_time
                        acc_passes += 1

                elif phase == "V":
                    # All vertical swaps in this pass are parallel
                    for (r, c) in v_swaps:
                        a = int(A[r, c])
                        b = int(A[r + 1, c])
                        if ignoreSpectators and (a in spectatorIons and b in spectatorIons):
                            continue
                        A[r, c], A[r + 1, c] = b, a
                        heatingRates[a] += col_swap_heating_rate
                        heatingRates[b] += col_swap_heating_rate
                        did_any_swap = True

                    if did_any_swap:
                        timeElapsed += col_swap_time
                        acc_passes += 1

                else:
                    # Should never happen; phase is either H or V
                    pass

            # After executing the SAT schedule, check we reached the target.
            if not np.array_equal(A, T):
                # print("[WARN] SAT-driven reconfig: final layout does NOT match newAssignment!")
                # print("  A (final):")
                # print(A)
                # print("  T (target):")
                # print(T)
                # You can raise if you want:
                raise RuntimeError("SAT schedule did not realise target layout")
            # if not initial_placement:
            #     print(
            #         f"RECONFIGURATION (SAT schedule): {acc_passes} passes were needed for the current reconfiguration round, "
            #         f"taking {timeElapsed} time and {heatingRates} heating"
            #     )
            return heatingRates, timeElapsed
        # else:
            # sat_schedule is None: fall back to heuristic odd-even reconfiguration.
            # try:
            #     diff = int(np.sum(A != T))
            # except Exception:
            #     diff = "NA"
            # print(
            #     "[DEBUG _runOddEvenReconfig] sat_schedule is None; using heuristic odd-even reconfig "
            #     f"(Phase B/C/D). layout_diffs={diff}"
            # )

                # ---------- helper: odd-even passes ----------
        def row_pass_by_rank(even_phase: bool, row_rank: List[Dict[int, int]]) -> bool:
            maxSwapsInRow = 0
            start = 0 if even_phase else 1
            phase_label = "even" if even_phase else "odd"
            swaps_this_phase = 0

            for r in range(n):
                swapsInRow = 0
                rank = row_rank[r]
                for c in range(start, m - 1, 2):
                    a = int(A[r, c])
                    b = int(A[r, c + 1])
                    if rank[a] > rank[b]:
                        A[r, c], A[r, c + 1] = b, a
                        heatingRates[a] += row_swap_heating
                        heatingRates[b] += row_swap_heating
                        swapsInRow += 1
                if swapsInRow > maxSwapsInRow:
                    maxSwapsInRow = swapsInRow
                swaps_this_phase += swapsInRow

            # print(
            #     f"[DEBUG _runOddEvenReconfig][row_pass_by_rank] "
            #     f"phase={phase_label}, swaps_total={swaps_this_phase}, "
            #     f"max_swaps_row={maxSwapsInRow}"
            # )
            return maxSwapsInRow > 0

        def col_bucket_pass(
            even_phase: bool, bucket_mod: int, ion_to_dest_row: Dict[int, int]
        ) -> bool:
            maxSwapsInCol = 0
            start = 0 if even_phase else 1
            phase_label = "even" if even_phase else "odd"
            swaps_this_phase = 0

            for c in range(bucket_mod, m, k):
                swapsInCol = 0
                for r in range(start, n - 1, 2):
                    a = int(A[r, c])
                    b = int(A[r + 1, c])
                    if ion_to_dest_row[a] > ion_to_dest_row[b]:
                        A[r, c], A[r + 1, c] = b, a
                        heatingRates[a] += col_swap_heating_rate
                        heatingRates[b] += col_swap_heating_rate
                        swapsInCol += 1
                if swapsInCol > maxSwapsInCol:
                    maxSwapsInCol = swapsInCol
                swaps_this_phase += swapsInCol

            # print(
            #     f"[DEBUG _runOddEvenReconfig][col_bucket_pass] "
            #     f"bucket_mod={bucket_mod}, phase={phase_label}, "
            #     f"swaps_total={swaps_this_phase}, max_swaps_col={maxSwapsInCol}"
            # )
            return maxSwapsInCol > 0

        # ---------- Phase B greedy target layout ----------
        B = GlobalReconfigurations.phaseB_greedy_layout(A, T)[1]
        # print(f"A={A}, B={B}, T={T}")

        # ---------- destination row/col maps (ion -> dest row/col) ----------
        ion_to_dest_row: Dict[int, int] = {}
        ion_to_dest_col: Dict[int, int] = {}
        for r in range(n):
            for c in range(m):
                ion = int(T[r, c])
                ion_to_dest_row[ion] = r
                ion_to_dest_col[ion] = c

        # Sanity 1: each row of B must be a permutation of the row of A
        try:
            for r in range(n):
                if set(B[r]) != set(A[r]):
                    raise AssertionError("row permutation mismatch")
        except AssertionError:
            print(
                "[DEBUG _runOddEvenReconfig][Phase B] assertion failed: "
                "B row not a permutation of A row",
                "row_idx=", r,
                "A_row=", list(A[r]),
                "B_row=", list(B[r]),
            )
            raise

        # Sanity 2: for each column of B, destination rows must be all distinct
        try:
            for c in range(m):
                dest_rows_col = [ion_to_dest_row[int(B[r, c])] for r in range(n)]
                if len(set(dest_rows_col)) != n:
                    raise AssertionError("dest_rows column clash")
        except AssertionError:
            print(
                "[DEBUG _runOddEvenReconfig][Phase B] assertion failed: "
                "duplicate dest_rows in a column",
                "m=", m,
                "n=", n,
                "dest_rows_col0=",
                [ion_to_dest_row[int(B[r, 0])] for r in range(n)] if m > 0 else [],
            )
            raise

        # Phase B target row order is exactly B
        desired_row_order = B.copy()

        # Row ranks for Phase B permutation
        row_rank_phaseB: List[Dict[int, int]] = []
        for r in range(n):
            row_rank_phaseB.append(
                {ion: idx for idx, ion in enumerate(desired_row_order[r])}
            )

        acc_cost = 0
        # print(
        #     f"[DEBUG _runOddEvenReconfig] Phase B start: m={m}, "
        #     f"current_vs_target_diffs={int(np.sum(A != T))}"
        # )

        # Execute ≤ m odd–even steps to realise the permutation per row
        for _ in range(m):
            oddpass = row_pass_by_rank(True, row_rank_phaseB)
            evenpass = row_pass_by_rank(False, row_rank_phaseB)
            timeElapsed += oddpass * row_swap_time
            timeElapsed += evenpass * row_swap_time
            acc_cost += int(oddpass) + int(evenpass)

        # print(f"A={A}, B={B}, T={T}")

        # ==========================================================
        # Phase C: vertical odd–even with k-way parallel buckets.
        # ==========================================================
        # print(
        #     f"[DEBUG _runOddEvenReconfig] Phase C start: k={k}, "
        #     f"current_vs_target_diffs={int(np.sum(A != T))}"
        # )
        for t in range(k):
            for _ in range(n):
                oddpass = col_bucket_pass(True, t, ion_to_dest_row)
                evenpass = col_bucket_pass(False, t, ion_to_dest_row)
                timeElapsed += oddpass * col_swap_time
                timeElapsed += evenpass * col_swap_time
                acc_cost += int(oddpass) + int(evenpass)
            if t < k - 1:
                # "Parallel row reconfig"
                timeElapsed += k * row_swap_time
                for idx in heatingRates.keys():
                    heatingRates[idx] += row_swap_heating

        # print(f"A={A}, B={B}, T={T}")

        # ==========================================================
        # Phase D: final row-wise odd–even to exact target order.
        # ==========================================================
        row_rank_final: List[Dict[int, int]] = []
        for r in range(n):
            row_rank_final.append({ion: idx for idx, ion in enumerate(T[r, :])})

        # print(
        #     f"[DEBUG _runOddEvenReconfig] Phase D start: rows={n}, "
        #     f"current_vs_target_diffs={int(np.sum(A != T))}"
        # )
        for _ in range(m):
            oddpass = row_pass_by_rank(True, row_rank_final)
            evenpass = row_pass_by_rank(False, row_rank_final)
            timeElapsed += oddpass * row_swap_time
            timeElapsed += evenpass * row_swap_time
            acc_cost += int(oddpass) + int(evenpass)

        # print(f"A={A}, B={B}, T={T}")

        # Final sanity: how close are we to the target layout?
        try:
            final_diff = int(np.sum(A != T))
        except Exception:
            final_diff = "NA"

        # print(
        #     f"[DEBUG _runOddEvenReconfig] Phase D end: acc_cost={acc_cost}, "
        #     f"final_layout_diffs={final_diff}"
        # )

        if not initial_placement:
            print(
                f"RECONFIGURATION: {acc_cost} passes were needed for the "
                f"current reconfiguration round, taking {timeElapsed} time and "
                f"{heatingRates} heating"
            )

        return heatingRates, timeElapsed


 

    @staticmethod
    def phaseB_greedy_layout(
        A_in: np.ndarray,
        T_in: np.ndarray,
    ) -> Tuple[int, np.ndarray]:
        """
        Phase-B layout helper.

        Given current layout A_in (n×m) and target layout T_in (n×m),
        construct a row-wise permutation B of A_in such that:

          • Each column of B contains ions with *all distinct* destination rows
            (according to T_in).
          • Returns (max_horizontal_displacement, B), where the displacement
            is |orig_col - new_col| measured per ion within its row.

        This assumes A_in and T_in contain exactly the same ion IDs.
        """
        A = np.asarray(A_in, dtype=int)
        T = np.asarray(T_in, dtype=int)
        n, m = A.shape
        if T.shape != (n, m):
            raise ValueError("phaseB_greedy_layout: A and T must have same shape")

        # ---- dest row for every ion id (from T) ----
        ion_to_dest_row: Dict[int, int] = {}
        for r in range(n):
            for c in range(m):
                ion_to_dest_row[int(T[r, c])] = r

        # sanity: same ions
        if set(A.flatten()) != set(T.flatten()):
            raise ValueError(
                "phaseB_greedy_layout: A and T must contain the same ion IDs"
            )

        # ---- counts[r, d] & ions_per_row_dest[r][d] ----
        counts = np.zeros((n, n), dtype=int)
        ions_per_row_dest: List[List[List[int]]] = [[[] for _ in range(n)] for _ in range(n)]

        # inverse position map per row (ion -> original column)
        invpos: List[Dict[int, int]] = []
        for r in range(n):
            mp: Dict[int, int] = {}
            for c in range(m):
                ion = int(A[r, c])
                mp[ion] = c
            invpos.append(mp)

        for r in range(n):
            for c in range(m):
                ion = int(A[r, c])
                d = ion_to_dest_row[ion]
                counts[r, d] += 1
                ions_per_row_dest[r][d].append(ion)

        # consistency check between counts and buckets
        for r in range(n):
            for d in range(n):
                if counts[r, d] != len(ions_per_row_dest[r][d]):
                    raise RuntimeError(
                        "phaseB_greedy_layout: counts/buckets mismatch at "
                        f"(row={r}, dest={d}): counts={counts[r,d]}, "
                        f"bucket_len={len(ions_per_row_dest[r][d])}"
                    )

        # row/col sums must be m
        for r in range(n):
            if counts[r, :].sum() != m:
                raise RuntimeError(
                    f"phaseB_greedy_layout: row {r} total count {counts[r,:].sum()} != m={m}"
                )
        for d in range(n):
            if counts[:, d].sum() != m:
                raise RuntimeError(
                    f"phaseB_greedy_layout: dest-row {d} total count {counts[:,d].sum()} != m={m}"
                )

        desired = np.zeros_like(A)
        max_disp = 0

        # ---------- internal helper: perfect matching from counts ----------
        def _perfect_matching_from_counts(counts_mat: np.ndarray) -> List[int]:
            """
            Given current counts[r, d] (non-negative ints) with
            sum_d counts[r,d] == m' for all r and sum_r counts[r,d] == m' for all d,
            find a perfect matching on the graph of edges where counts[r,d] > 0.

            Returns match_row_to_dest[r] = d for all rows r.
            Raises RuntimeError if no perfect matching exists.
            """
            nn = counts_mat.shape[0]
            # adjacency: neighbors[r] = [d | counts[r,d]>0]
            neighbors: List[List[int]] = []
            for rr in range(nn):
                nbrs = [dd for dd in range(nn) if counts_mat[rr, dd] > 0]
                neighbors.append(nbrs)

            match_to_row = [-1] * nn  # dest d -> row r

            def dfs(r: int, seen: List[bool]) -> bool:
                for d in neighbors[r]:
                    if seen[d]:
                        continue
                    seen[d] = True
                    if match_to_row[d] == -1 or dfs(match_to_row[d], seen):
                        match_to_row[d] = r
                        return True
                return False

            for r in range(nn):
                seen = [False] * nn
                if not dfs(r, seen):
                    raise RuntimeError(
                        "phaseB_greedy_layout: no perfect matching for current counts"
                    )

            # convert dest->row into row->dest
            match_row_to_dest = [-1] * nn
            for d, r in enumerate(match_to_row):
                if r == -1:
                    continue
                match_row_to_dest[r] = d

            if any(d == -1 for d in match_row_to_dest):
                raise RuntimeError(
                    "phaseB_greedy_layout internal: incomplete matching"
                )
            return match_row_to_dest

        # ---- build columns one by one via perfect matchings ----
        for c in range(m):
            # find any perfect matching consistent with current counts
            match = _perfect_matching_from_counts(counts)

            # reconstruct column c
            for r in range(n):
                d = int(match[r])

                # sanity: we must have capacity here
                if counts[r, d] <= 0:
                    raise RuntimeError(
                        "phaseB_greedy_layout: internal bug — picked (row={r}, dest={d}) "
                        "with zero capacity despite matching only on counts>0; "
                        f"(r={r}, d={d}, counts[r,d]={counts[r,d]})"
                    )

                bucket = ions_per_row_dest[r][d]
                if not bucket:
                    raise RuntimeError(
                        "phaseB_greedy_layout: empty bucket for "
                        f"(row={r}, dest={d}) while reconstructing column {c}; "
                        f"counts[r,d]={counts[r,d]}"
                    )

                # take any ion with dest row d from this row
                ion = bucket.pop()
                counts[r, d] -= 1
                desired[r, c] = ion

                # update displacement
                j0 = invpos[r][ion]
                disp = abs(j0 - c)
                if disp > max_disp:
                    max_disp = disp

        # final sanity checks
        for r in range(n):
            if set(desired[r, :]) != set(A[r, :]):
                raise RuntimeError(
                    f"phaseB_greedy_layout: desired row {r} not a permutation of A row {r}"
                )

        for c in range(m):
            dests_col = [ion_to_dest_row[int(desired[r, c])] for r in range(n)]
            if len(set(dests_col)) != n:
                raise RuntimeError(
                    f"phaseB_greedy_layout: column {c} does not have unique "
                    f"destination rows; dest_rows_col={dests_col}"
                )

        return max_disp, desired

    @staticmethod
    def _optimal_QMR_for_WISE(*args, **kwargs):
        """Delegation stub — implementation moved to qccd_SAT_WISE_odd_even_sorter."""
        from ..compiler.qccd_SAT_WISE_odd_even_sorter import optimal_QMR_for_WISE
        return optimal_QMR_for_WISE(*args, **kwargs)
    



# Re-export so `from .qccd_operations import *` picks it up
from ..compiler.qccd_SAT_WISE_odd_even_sorter import NoFeasibleLayoutError  # noqa: E402, F401


class Split(CrystalOperation):
    KEY = Operations.SPLIT
    # Constants now sourced from physics.py DEFAULT_CALIBRATION
    # Kept as class attributes for backward compatibility
    SPLITTING_TIME = DEFAULT_CALIBRATION.split_time
    HEATING_RATE = DEFAULT_CALIBRATION.split_heating_rate

    def __init__(
        self,
        run: Callable[[Any], None],
        involvedComponents: Sequence[QCCDComponent],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
        self._crossing: Crossing = kwargs["crossing"]
        self._ion: Ion = kwargs["ion"]

    def calculateOperationTime(self) -> None:
        self._operationTime = self.SPLITTING_TIME

    def _generateLabelAddOns(self) -> None:
        self._involvedIonsForLabel = [self._crossing.ion]
        self._addOns = f" {self._crossing.ion.label}"

    @property
    def isApplicable(self) -> bool:
        if not self._crossing.hasTrap(self._trap):
            return False
        if self._crossing.ion is not None:
            return False
        if len(self._trap.ions) == 0:
            return False
        if self._crossing.getEdgeIon(self._trap) != self._ion:
            return False
        return super().isApplicable
    
    def _checkApplicability(self) -> None:
        if not self._crossing.hasTrap(self._trap):
            raise ValueError(f"Split: crossing does not include trap {self._trap.idx}")
        if self._crossing.ion is not None:
            raise ValueError(
                f"Split: crossing is already occupied by ion {self._crossing.ion.idx}"
            )
        if len(self._trap.ions) == 0:
            raise ValueError(f"Split: trap {self._trap.idx} has no ions")
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(
        cls, trap: Trap, crossing: Crossing, ion: Optional[Ion] = None
    ):
        def run():
            ion = crossing.getEdgeIon(trap)
            trap.removeIon(ion)
            crossing.setIon(ion, trap)
            trap.addMotionalEnergy(cls.HEATING_RATE * cls.SPLITTING_TIME)
            ion.addMotionalEnergy(cls.HEATING_RATE * cls.SPLITTING_TIME)
            # 3N mode heating for the remaining crystal
            ms = getattr(trap, 'mode_structure', None)
            if ms is not None:
                ms.heat_modes(DEFAULT_CALIBRATION.transport_heating["split"])

        return cls(
            run=lambda _: run(),
            ion=ion,
            trap=trap,
            crossing=crossing,
            involvedComponents=[trap, crossing, *crossing.connection],
        )


class Merge(CrystalOperation):
    KEY = Operations.MERGE
    # Constants now sourced from physics.py DEFAULT_CALIBRATION
    MERGING_TIME = DEFAULT_CALIBRATION.merge_time
    HEATING_RATE = DEFAULT_CALIBRATION.merge_heating_rate

    def __init__(
        self,
        run: Callable[[Any], bool],
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
        self._crossing: Crossing = kwargs["crossing"]
        self._ion: Ion = kwargs["ion"]

    def calculateOperationTime(self) -> None:
        self._operationTime = self.MERGING_TIME

    def _generateLabelAddOns(self) -> None:
        self._involvedIonsForLabel = [self._crossing.getEdgeIon(self._trap)]
        self._addOns = f" {self._crossing.getEdgeIon(self._trap).label}"

    @property
    def isApplicable(self) -> bool:
        if not self._crossing.hasTrap(self._trap):
            return False
        if self._crossing.ion is None:
            return False
        if self._crossing.ion != self._ion:
            return False
        return super().isApplicable

    def _checkApplicability(self) -> None:
        if not self._crossing.hasTrap(self._trap):
            raise ValueError(f"Merge: crossing does not include trap {self._trap.idx}")
        if self._crossing.ion is None:
            raise ValueError(f"Merge: crossing is empty")
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(
        cls, trap: Trap, crossing: Crossing, ion: Optional[Ion] = None
    ):
        def run():
            ion = crossing.ion
            crossing.clearIon()
            edge_ion = crossing.getEdgeIon(trap) if trap.ions else None
            # Use crossing geometry (which side the ion approaches from)
            # to decide insertion order.  The previous delta-based logic
            # compared ion.pos to edge_ion.pos *after* setIon(), which
            # places them fractions of a spacing unit apart — too close
            # for a reliable sign check.  Instead, check whether the
            # other node in the crossing is to the left/above or
            # right/below the destination trap.
            _horiz = getattr(trap, '_isHorizontal', True)
            if edge_ion is not None:
                src, tgt = crossing.connection
                other = src if tgt == trap else tgt
                approach = (
                    (other.pos[0] - trap.pos[0]) * _horiz
                    + (other.pos[1] - trap.pos[1]) * (1 - _horiz)
                )
                offset = 0 if approach < 0 else 1
            else:
                offset = 0
            trap.addIon(ion, adjacentIon=edge_ion, offset=offset)
            trap.addMotionalEnergy(cls.HEATING_RATE * cls.MERGING_TIME)
            trap.addMotionalEnergy(cls.HEATING_RATE * cls.MERGING_TIME)
            # 3N mode heating for the merged crystal
            ms = getattr(trap, 'mode_structure', None)
            if ms is not None:
                ms.heat_modes(DEFAULT_CALIBRATION.transport_heating["merge"])

        return cls(
            run=lambda _: run(),
            ion=ion,
            crossing=crossing,
            trap=trap,
            involvedComponents=[trap, crossing, *crossing.connection],
        )


class CrystalRotation(CrystalOperation):
    KEY = Operations.CRYSTAL_ROTATION
    # Constants now sourced from physics.py DEFAULT_CALIBRATION
    ROTATION_TIME = DEFAULT_CALIBRATION.rotation_time
    HEATING_RATE = DEFAULT_CALIBRATION.rotation_heating_rate

    def __init__(
        self,
        run: Callable[[Any], bool],
        trap: Trap,
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
        self._trap: Trap = trap

    def calculateOperationTime(self) -> None:
        self._operationTime = self.ROTATION_TIME

    @property
    def isApplicable(self) -> bool:
        return super().isApplicable
    
    def _checkApplicability(self) -> None:
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(cls, trap: Trap):
        def run():
            ions = list(trap.ions).copy()[::-1]
            for ion in ions:
                trap.removeIon(ion)
            for i, ion in enumerate(ions):
                trap.addIon(ion, offset=i)
            trap.addMotionalEnergy(cls.HEATING_RATE * cls.ROTATION_TIME)

        return cls(
            run=lambda _: run(),
            trap=trap,
            involvedComponents=[trap],
        )



class CoolingOperation(CrystalOperation):
    KEY = Operations.RECOOLING
    # Constants now sourced from physics.py DEFAULT_CALIBRATION
    COOLING_TIME = DEFAULT_CALIBRATION.recool_time
    HEATING_RATE = DEFAULT_CALIBRATION.cooling_heating_rate

    def __init__(
        self,
        run: Callable[[Any], bool],
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
    
    def calculateOperationTime(self) -> None:
        self._operationTime =  self.COOLING_TIME

    @property
    def isApplicable(self) -> bool:
        if not self._trap.hasCoolingIon:
            return False
        return super().isApplicable

    def _checkApplicability(self) -> None:
        if not self._trap.hasCoolingIon:
            raise ValueError(f"CoolingOperation: trap {self._trap.idx} does not include a cooling ion")
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(
        cls, trap: Trap
    ):
        def run():
            trap.coolTrap()
            trap.addMotionalEnergy(cls.HEATING_RATE * cls.COOLING_TIME)

        return cls(
            run=lambda _: run(),
            trap=trap,
            involvedComponents=[trap],
        )




class Move(Operation):
    KEY = Operations.MOVE
    # Constants now sourced from physics.py DEFAULT_CALIBRATION
    MOVING_TIME = DEFAULT_CALIBRATION.shuttle_time
    HEATING_RATE = DEFAULT_CALIBRATION.shuttle_heating_rate

    def __init__(
        self,
        run: Callable[[Any], bool],
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
        self._crossing: Crossing = kwargs["crossing"]
        self._ion: Ion = kwargs["ion"]

    def calculateOperationTime(self) -> None:
        self._operationTime = self.MOVING_TIME

    def calculateFidelity(self) -> None:
        self._fidelity = 1  # NOISE INCORPORATED INTO HEATING MODEL

    def calculateDephasingFidelity(self) -> None:
        self._dephasingFidelity= 1 # little to no idling due to shuttling being fast

    def _generateLabelAddOns(self) -> None:
        self._involvedIonsForLabel = [self._crossing.ion]
        self._addOns = f" {self._crossing.ion.label}"

    @property
    def isApplicable(self) -> bool:
        return bool(self._crossing.ion) and self._ion == self._crossing.ion and super().isApplicable

    def _checkApplicability(self) -> None:
        if not self._crossing.ion:
            raise ValueError(f"Move: crossing does not contain ion")
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(cls, crossing: Crossing, ion: Optional[Ion] = None):
        def run():
            crossing.ion.addMotionalEnergy(cls.HEATING_RATE * cls.MOVING_TIME)
            crossing.moveIon()

        return cls(
            run=lambda _: run(),
            ion=ion,
            crossing=crossing,
            involvedComponents=[crossing],
        )

# TODO: junction crossing should really go over the junction to the next crossing
class JunctionCrossing(Operation):
    KEY = Operations.JUNCTION_CROSSING
    # Constants now sourced from physics.py DEFAULT_CALIBRATION
    CROSSING_TIME = DEFAULT_CALIBRATION.junction_time
    HEATING_RATE = DEFAULT_CALIBRATION.junction_heating_rate

    def __init__(
        self,
        run: Callable[[Any], bool],
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
        self._crossing: Crossing = kwargs["crossing"]
        self._junction: Junction = kwargs["junction"]
        self._ion: Ion = kwargs["ion"]

    def calculateOperationTime(self) -> None:
        self._operationTime = self.CROSSING_TIME

    def calculateFidelity(self) -> None:
        self._fidelity = 1  # NOISE INCORPORATED INTO HEATING MODEL

    def calculateDephasingFidelity(self) -> None:
        self.calculateOperationTime()
        self._dephasingFidelity = DEFAULT_FIDELITY_MODEL.dephasing_fidelity(self._operationTime)

    def _generateLabelAddOns(self) -> None:
        self._involvedIonsForLabel = [self._ion] if self._ion else []
        self._addOns = f" {self._ion.label}" if self._ion else ""

    @property
    def isApplicable(self) -> bool:
        if not self._crossing.hasJunction(self._junction):
            return False
        if not self._crossing.ion and len(self._junction.ions) == 0:
            return False
        if self._crossing.ion and self._crossing.ion != self._ion:
            return False
        if self._junction.ions and self._junction.ions[0] != self._ion:
            return False
        if self._crossing.ion and len(self._junction.ions) == self._junction.DEFAULT_CAPACITY:
            return False
        return super().isApplicable
    
    def _checkApplicability(self) -> None:
        if not self._crossing.hasJunction(self._junction):
            raise ValueError(
                f"JunctionCrossing: crossing does not contain junction {self._junction.idx}"
            )
        if not self._crossing.ion and len(self._junction.ions) == 0:
            raise ValueError(
                f"JunctionCrossing: neither junction nor crossing has an ion"
            )
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(
        cls, junction: Junction, crossing: Crossing, ion: Optional[Ion] = None
    ):
        def run():
            if not crossing.ion and len(junction.ions) > 0:
                ion = junction.ions[0]
                crossing.setIon(ion, junction)
                junction.removeIon(ion)
            else:
                ion = crossing.ion
                crossing.clearIon()
                junction.addIon(ion)
            ion.addMotionalEnergy(cls.HEATING_RATE * cls.CROSSING_TIME)

        return cls(
            run=lambda _: run(),
            ion=ion,
            junction=junction,
            crossing=crossing,
            involvedComponents=[junction, crossing],
        )






class PhysicalCrossingSwap(Operation):
    KEY = Operations.JUNCTION_CROSSING
    # Constants now sourced from physics.py DEFAULT_CALIBRATION
    CROSSING_TIME = DEFAULT_CALIBRATION.crossing_swap_time
    HEATING_RATE = DEFAULT_CALIBRATION.crossing_swap_heating_rate

    def __init__(
        self,
        run: Callable[[Any], bool],
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
        self._crossing: Crossing = kwargs["crossing"]
        self._junction: Junction = kwargs["junction"]
        self._ion: Ion = kwargs["ion"]

    def calculateOperationTime(self) -> None:
        self._operationTime = self.CROSSING_TIME

    def calculateFidelity(self) -> None:
        self._fidelity = 1  # NOISE INCORPORATED INTO HEATING MODEL

    def calculateDephasingFidelity(self) -> None:
        self.calculateOperationTime()
        self._dephasingFidelity = DEFAULT_FIDELITY_MODEL.dephasing_fidelity(self._operationTime)

    def _generateLabelAddOns(self) -> None:
        self._involvedIonsForLabel = [self._ion] if self._ion else []
        self._addOns = f" {self._ion.label}" if self._ion else ""

    @property
    def isApplicable(self) -> bool:
        if not self._crossing.hasJunction(self._junction):
            return False
        if not self._crossing.ion and len(self._junction.ions) == 0:
            return False
        if self._crossing.ion and self._crossing.ion != self._ion:
            return False
        if self._junction.ions and self._junction.ions[0] != self._ion:
            return False
        if self._crossing.ion and len(self._junction.ions) == self._junction.DEFAULT_CAPACITY:
            return False
        return super().isApplicable
    
    def _checkApplicability(self) -> None:
        if not self._crossing.hasJunction(self._junction):
            raise ValueError(
                f"JunctionCrossing: crossing does not contain junction {self._junction.idx}"
            )
        if not self._crossing.ion and len(self._junction.ions) == 0:
            raise ValueError(
                f"JunctionCrossing: neither junction nor crossing has an ion"
            )
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(
        cls, junction: Junction, crossing: Crossing, ion: Optional[Ion] = None
    ):
        def run():
            if not crossing.ion and len(junction.ions) > 0:
                ion = junction.ions[0]
                crossing.setIon(ion, junction)
                junction.removeIon(ion)
            else:
                ion = crossing.ion
                crossing.clearIon()
                junction.addIon(ion)
            ion.addMotionalEnergy(cls.HEATING_RATE * cls.CROSSING_TIME)

        return cls(
            run=lambda _: run(),
            ion=ion,
            junction=junction,
            crossing=crossing,
            involvedComponents=[junction, crossing],
        )


class ParallelOperation(Operation):
    KEY = Operations.PARALLEL

    def __init__(
        self, run: Callable[[Any], bool], operations: Sequence[Operation], **kwargs
    ) -> None:
        super().__init__(run, **kwargs, operations=operations)
        self._operations = operations

    def calculateOperationTime(self) -> None:
        for op in self._operations:
            op.calculateOperationTime()
        self._operationTime = max(op.operationTime() for op in self._operations)

    def calculateDephasingFidelity(self) -> None:
        for op in self._operations:
            op.calculateDephasingFidelity()
        self._dephasingFidelity = float(max([op.dephasingFidelity() for op in self._operations]))


    def calculateFidelity(self) -> None:
        for op in self._operations:
            op.calculateFidelity()
        # assuming independence between parallel operations
        self._fidelity = float(np.prod([op.fidelity() for op in self._operations]))

    def _generateLabelAddOns(self) -> None:
        self._addOns = ""
        for op in self._operations:
            self._addOns += f" {op.KEY.name}"

    @property
    def isApplicable(self) -> bool:
        return all(op.isApplicable for op in self.operations)
    
    def _checkApplicability(self) -> None:
        return True

    @property
    def operations(self) -> Sequence[Operation]:
        return self._operations

    @classmethod
    def physicalOperation(cls, operationsToStart: Sequence[Operation], operationsStarted: Sequence[Operation]):
        def run():
            for op in np.random.permutation(operationsToStart):
                op.run()

        involvedComponents = []
        operations = list(operationsStarted)+list(operationsToStart)
        for op in operations:
            involvedComponents += list(op.involvedComponents)
        return cls(
            run=lambda _: run(),
            operations=operations,
            involvedComponents=set(involvedComponents),
        )


# ============================================================================
# Simplified aliases for compiler compatibility
# ============================================================================
# The WISE / junction compilers use simplified constructors (ion, gate_type=...)
# matching trapped_ion/operations.py API.  These wrappers delegate to the
# qccd_operations_on_qubits types that the old routing pipeline recognises.

from .qccd_operations_on_qubits import (
    TwoQubitMSGate as _TwoQubitMSGate,
    OneQubitGate as _OneQubitGate,
    XRotation as _XRotation,
    YRotation as _YRotation,
    Measurement as _OldMeasurement,
    QubitReset as _OldQubitReset,
    GateSwap,
)

# MSGate: direct alias — .qubitOperation(ion1, ion2) works unchanged.
MSGate = _TwoQubitMSGate


class SingleQubitGate(_OneQubitGate):
    """OneQubitGate with simplified (ion, gate_type=) constructor."""

    def __init__(self, ion, *, gate_type="R", **kwargs):
        def _noop(trap):
            ...

        super().__init__(run=_noop, ion=ion, involvedComponents=[ion], **kwargs)
        self._gate_type = gate_type

    @property
    def label(self) -> str:
        return f"{self._gate_type}({self._ions[0].label})" + self._addOns


class Measurement(_OldMeasurement):
    """Measurement with simplified (ion,) constructor."""

    def __init__(self, ion, **kwargs):
        def _noop(trap):
            ...

        super().__init__(run=_noop, ion=ion, involvedComponents=[ion], **kwargs)


class QubitReset(_OldQubitReset):
    """QubitReset with simplified (ion,) constructor."""

    def __init__(self, ion, **kwargs):
        def _noop(trap):
            ...

        super().__init__(run=_noop, ion=ion, involvedComponents=[ion], **kwargs)
    
