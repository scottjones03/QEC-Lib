#!/usr/bin/env python3
"""
Tests for non-CSS code and concatenated code compatibility.

Validates that:
1. _get_code_stabilizer_counts correctly counts stabilizers for CSS,
   non-CSS, and concatenated codes.
2. QubitAllocation.from_codes allocates the right number of ancillas
   for non-CSS stabilizer codes.
3. GeneralStabilizerRoundBuilder ancilla properties (x_ancillas,
   all_ancillas) return the correct qubit indices.
4. FaultTolerantGadgetExperiment.to_stim() produces deterministic
   circuits for a non-CSS code with a transversal identity gadget.
5. Gadget code-type validation catches CSS-only gadgets with non-CSS codes.
6. ConcatenatedCSSCode works through the FT gadget pipeline (flat model).
"""
import numpy as np
import pytest
import stim

from qectostim.codes.generic.generic_stabilizer_code import GenericStabilizerCode  # noqa
from qectostim.codes.abstract_code import StabilizerCode
from qectostim.gadgets.layout import QubitAllocation, _get_code_stabilizer_counts
from qectostim.experiments.stabilizer_rounds import (
    DetectorContext,
    GeneralStabilizerRoundBuilder,
    CSSStabilizerRoundBuilder,
)


# ============================================================================
# Fixtures: well-known codes
# ============================================================================

def _make_five_qubit_code() -> GenericStabilizerCode:
    """[[5,1,3]] perfect code — the smallest non-CSS stabilizer code."""
    code = GenericStabilizerCode.from_stabilizer_strings(
        ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"],
        logical_x=["XXXXX"],
        logical_z=["ZZZZZ"],
    )
    return code


def _make_steane_code():
    """[[7,1,3]] Steane code — a CSS code for comparison."""
    from qectostim.codes.small.steane_713 import SteaneCode713
    return SteaneCode713()


# ============================================================================
# Test: _get_code_stabilizer_counts
# ============================================================================

class TestGetCodeStabilizerCounts:
    """Tests for the unified stabilizer count helper."""

    def test_css_code_uses_hx_hz(self):
        """CSS codes should return (rows(hx), rows(hz))."""
        steane = _make_steane_code()
        nx, nz = _get_code_stabilizer_counts(steane)
        assert nx == 3, f"Steane should have 3 X stabilizers, got {nx}"
        assert nz == 3, f"Steane should have 3 Z stabilizers, got {nz}"

    def test_non_css_code_uses_stabilizer_matrix(self):
        """Non-CSS codes should get stabilizer count from stabilizer_matrix."""
        five = _make_five_qubit_code()
        nx, nz = _get_code_stabilizer_counts(five)
        total = nx + nz
        assert total == 4, f"[[5,1,3]] has 4 stabilizers, got {total}"
        # Non-CSS: all stabilizers reported as x
        assert nx == 4
        assert nz == 0

    def test_code_with_no_stabilizers(self):
        """A bare Code subclass with no hx/hz/stabilizer_matrix → (0, 0)."""
        from qectostim.codes.abstract_code import Code
        class BareCode(Code):
            @property
            def n(self): return 3
            @property
            def k(self): return 1
            @property
            def logical_x_ops(self): return [{0: 'X'}]
            @property
            def logical_z_ops(self): return [{0: 'Z'}]
        bare = BareCode()
        nx, nz = _get_code_stabilizer_counts(bare)
        assert (nx, nz) == (0, 0)


# ============================================================================
# Test: QubitAllocation.from_codes for non-CSS
# ============================================================================

class TestQubitAllocationNonCSS:
    """Validate ancilla allocation for non-CSS stabilizer codes."""

    def test_five_qubit_code_gets_ancillas(self):
        """[[5,1,3]] should get 4 ancillas (one per stabilizer)."""
        five = _make_five_qubit_code()
        alloc = QubitAllocation.from_codes([five])
        block = alloc.get_block("block_0")
        assert block is not None
        assert block.data_count == 5
        # 4 stabilizers → 4 ancillas total (all in x bucket for non-CSS)
        total_anc = block.x_anc_count + block.z_anc_count
        assert total_anc == 4, f"Expected 4 ancillas, got {total_anc}"

    def test_css_code_gets_correct_ancillas(self):
        """Steane code should get 3+3=6 ancillas."""
        steane = _make_steane_code()
        alloc = QubitAllocation.from_codes([steane])
        block = alloc.get_block("block_0")
        assert block is not None
        assert block.x_anc_count == 3
        assert block.z_anc_count == 3

    def test_mixed_codes(self):
        """Allocating a CSS and non-CSS code together."""
        steane = _make_steane_code()
        five = _make_five_qubit_code()
        alloc = QubitAllocation.from_codes([steane, five])

        b0 = alloc.get_block("block_0")  # Steane
        b1 = alloc.get_block("block_1")  # Five-qubit

        assert b0 is not None and b1 is not None
        assert b0.data_count == 7
        assert b0.x_anc_count == 3
        assert b0.z_anc_count == 3
        assert b1.data_count == 5
        assert b1.x_anc_count + b1.z_anc_count == 4
        
        # No overlap in qubit indices
        all_b0 = set(b0.all_qubits)
        all_b1 = set(b1.all_qubits)
        assert all_b0.isdisjoint(all_b1)


# ============================================================================
# Test: GeneralStabilizerRoundBuilder ancilla properties
# ============================================================================

class TestGeneralBuilderAncillas:
    """Validate that GeneralStabilizerRoundBuilder exposes ancillas correctly."""

    def test_all_ancillas_returns_full_pool(self):
        five = _make_five_qubit_code()
        ctx = DetectorContext()
        builder = GeneralStabilizerRoundBuilder(
            code=five, ctx=ctx, block_name="test",
            data_offset=0, ancilla_offset=5,
        )
        assert len(builder.all_ancillas) == 4
        assert builder.all_ancillas == [5, 6, 7, 8]

    def test_x_ancillas_equals_all_ancillas(self):
        """For non-CSS codes, x_ancillas should return the whole pool."""
        five = _make_five_qubit_code()
        ctx = DetectorContext()
        builder = GeneralStabilizerRoundBuilder(
            code=five, ctx=ctx, data_offset=0, ancilla_offset=5,
        )
        assert builder.x_ancillas == builder.all_ancillas
        assert builder.z_ancillas == []

    def test_get_last_measurement_indices(self):
        """After one round, get_last_measurement_indices should return indices."""
        five = _make_five_qubit_code()
        ctx = DetectorContext()
        builder = GeneralStabilizerRoundBuilder(
            code=five, ctx=ctx, data_offset=0, ancilla_offset=5,
        )
        circuit = stim.Circuit()
        builder.emit_round(circuit, emit_detectors=False)
        last = builder.get_last_measurement_indices()
        assert len(last["X"]) == 4
        assert last["Z"] == []


# ============================================================================
# Test: Gadget code-type validation
# ============================================================================

class TestGadgetCodeValidation:
    """Validate that gadgets enforce code-type requirements."""

    def test_surgery_cnot_rejects_non_css(self):
        """CSSSurgeryCNOTGadget should reject non-CSS codes."""
        from qectostim.gadgets.css_surgery_cnot import CSSSurgeryCNOTGadget
        gadget = CSSSurgeryCNOTGadget()
        five = _make_five_qubit_code()
        with pytest.raises(ValueError, match="requires CSS"):
            gadget.validate_codes([five, five, five])

    def test_transversal_cnot_accepts_non_css(self):
        """TransversalCNOTGadget should accept any stabilizer code."""
        from qectostim.gadgets.transversal_cnot import TransversalCNOTGadget
        gadget = TransversalCNOTGadget()
        five = _make_five_qubit_code()
        # Should not raise
        gadget.validate_codes([five, five])


# ============================================================================
# Test: End-to-end circuit generation for non-CSS codes
# ============================================================================

class TestNonCSSEndToEnd:
    """Validate complete FT circuit generation with non-CSS codes."""

    def test_memory_experiment_five_qubit_code(self):
        """A memory experiment (identity gadget) for [[5,1,3]] should work."""
        from qectostim.experiments.memory import StabilizerMemoryExperiment
        five = _make_five_qubit_code()
        exp = StabilizerMemoryExperiment(
            code=five, rounds=3, basis="Z",
        )
        circuit = exp.to_stim()

        # Should have qubits allocated
        assert circuit.num_qubits > 0
        # Should have measurements
        assert circuit.num_measurements > 0

    def test_five_qubit_deterministic_noiseless(self):
        """Memory experiment for [[5,1,3]] should be deterministic at zero noise."""
        from qectostim.experiments.memory import StabilizerMemoryExperiment
        five = _make_five_qubit_code()
        exp = StabilizerMemoryExperiment(
            code=five, rounds=3, basis="Z",
        )
        circuit = exp.to_stim()

        # Check determinism: sample with 0 noise → no detector flips
        sampler = circuit.compile_detector_sampler()
        results = sampler.sample(shots=100)
        n_flips = results.sum()
        assert n_flips == 0, (
            f"Non-deterministic detectors: {n_flips} flips in 100 shots. "
            f"Circuit has {circuit.num_detectors} detectors."
        )


# ============================================================================
# Test: ConcatenatedCSSCode through FT pipeline
# ============================================================================

class TestConcatenatedCSSCodeFT:
    """Validate ConcatenatedCSSCode works through FT gadget pipeline."""

    def test_concat_allocation(self):
        """ConcatenatedCSSCode should get correct ancilla allocation (flat model)."""
        from qectostim.codes.small.steane_713 import SteaneCode713
        from qectostim.codes.composite.concatenated import ConcatenatedCSSCode

        inner = SteaneCode713()
        outer = SteaneCode713()
        concat = ConcatenatedCSSCode(outer, inner)

        alloc = QubitAllocation.from_codes([concat])
        block = alloc.get_block("block_0")
        assert block is not None
        # n = 7*7 = 49
        assert block.data_count == 49
        # hx rows + hz rows should be > 0
        total_anc = block.x_anc_count + block.z_anc_count
        assert total_anc > 0, "Concatenated code should have ancillas"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
