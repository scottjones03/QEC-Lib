#!/usr/bin/env python
"""
End-to-end runner for the trapped-ion QCCD framework.

Exercises the full pipeline on a d=2 rotated surface code for both
Augmented Grid and WISE architectures:

  build_ideal_circuit → compile → display → animate → noise → simulate

Outputs
-------
- ``aug_grid_layout.png``   — static architecture figure
- ``wise_layout.png``       — static architecture figure
- ``aug_grid_transport.mp4`` — animation (requires ffmpeg)
- ``wise_transport.mp4``     — animation (requires ffmpeg)
- Compilation & simulation metrics printed to stdout

Usage
-----
::

    python -m qectostim.experiments.hardware_simulation.trapped_ion.demo.run
    # or
    python <path-to-this-file>/run.py [--output-dir DIR] [--no-animate] [--no-simulate]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
import stim

# ── QECToStim imports ────────────────────────────────────────────────
from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
from qectostim.experiments.memory import CSSMemoryExperiment

# ── Trapped-ion module imports ───────────────────────────────────────
from qectostim.experiments.hardware_simulation.trapped_ion.utils import (
    AugmentedGridArchitecture,
    WISEArchitecture,
    TrappedIonCompiler,
    TrappedIonExperiment,
    TrappedIonNoiseModel,
)
from qectostim.experiments.hardware_simulation.trapped_ion.utils.qccd_nodes import (
    QCCDWiseArch,
    SpectatorIon,
)
from qectostim.experiments.hardware_simulation.trapped_ion.viz import (
    display_architecture,
    animate_transport,
    render_animation,
)

# =====================================================================
# Helpers
# =====================================================================


def build_viz_mappings(
    compiler: TrappedIonCompiler,
) -> Tuple[Dict[int, str], Dict[int, int], Dict[int, int]]:
    """Extract ion_roles, physical_to_logical, ion_idx_remap from a compiler.

    Parameters
    ----------
    compiler : TrappedIonCompiler
        A compiler whose ``decompose_to_native`` (and optionally full
        ``compile``) has already run so that ``ion_mapping`` is populated.

    Returns
    -------
    ion_roles : dict
        ``{ion.idx: 'D' | 'M' | 'P'}``
    physical_to_logical : dict
        ``{ion.idx: stim_qubit_index}``
    ion_idx_remap : dict
        ``{ion.idx: dense_index}``  (0-based)
    """
    ion_roles: Dict[int, str] = {}
    physical_to_logical: Dict[int, int] = {}

    for stim_idx, (ion, _coords) in compiler.ion_mapping.items():
        label = getattr(ion, "_label", "U")
        if isinstance(ion, SpectatorIon):
            ion_roles[ion.idx] = "P"
        else:
            ion_roles[ion.idx] = label[0] if label else "U"
        physical_to_logical[ion.idx] = stim_idx

    arch = compiler.architecture
    sorted_ions = sorted(arch.ions.values(), key=lambda i: i.idx)
    ion_idx_remap = {ion.idx: i for i, ion in enumerate(sorted_ions)}

    return ion_roles, physical_to_logical, ion_idx_remap


def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# =====================================================================
# Augmented Grid pipeline
# =====================================================================


def run_augmented_grid(
    ideal: stim.Circuit,
    output_dir: Path,
    *,
    animate: bool = True,
) -> Tuple[Any, TrappedIonCompiler, Any]:
    """Compile + visualise on an Augmented Grid architecture.

    Returns (arch, compiler, compiled_circuit).
    """
    _print_section("Augmented Grid — compile")

    # ── architecture + compiler ──────────────────────────────────────
    arch = AugmentedGridArchitecture(
        trap_capacity=2, rows=3, cols=3, padding=0,
    )
    compiler = TrappedIonCompiler(arch, is_wise=False)

    t0 = time.perf_counter()
    compiled = compiler.compile(ideal)
    t1 = time.perf_counter()

    # ── viz mappings ─────────────────────────────────────────────────
    ion_roles, physical_to_logical, ion_idx_remap = build_viz_mappings(compiler)

    # ── metrics ──────────────────────────────────────────────────────
    batches = compiled.scheduled.batches          # list[ParallelOperation]
    all_ops = compiled.scheduled.metadata.get("all_operations", [])
    metrics = compiled.compute_metrics()

    print(f"  Compile time:   {t1 - t0:.2f} s")
    print(f"  Traps:          {len(arch._manipulationTraps)}")
    print(f"  Routed ops:     {len(all_ops)}")
    print(f"  Parallel steps: {len(batches)}")
    print(f"  Depth:          {metrics.get('depth', '?')}")
    print(f"  Duration (µs):  {metrics.get('duration_us', '?')}")
    print(f"  Ion roles:      {ion_roles}")
    if batches:
        print(f"  First batch:    {batches[0].label}")

    # ── static figure ────────────────────────────────────────────────
    _print_section("Augmented Grid — static layout")
    arch.resetArrangement()
    arch.refreshGraph()

    fig, ax = display_architecture(
        arch,
        title="Augmented Grid — initial layout",
        show_junctions=True,
        show_edges=True,
        show_ions=True,
        show_labels=False,
        show_legend=True,
        ion_roles=ion_roles,
        ion_idx_remap=ion_idx_remap,
        physical_to_logical=physical_to_logical,
    )
    fig_path = output_dir / "aug_grid_layout.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")

    # ── animation ────────────────────────────────────────────────────
    if animate and batches:
        _print_section("Augmented Grid — animation")
        arch.resetArrangement()
        arch.refreshGraph()

        anim = animate_transport(
            arch=arch,
            operations=batches,
            interval=600,
            show_labels=False,
            interp_frames=2,
            gate_hold_frames=1,
            stim_circuit=ideal,
            figsize=(16, 10),
            title_prefix="Augmented Grid",
            repeat=False,
            ion_roles=ion_roles,
            ion_idx_remap=ion_idx_remap,
            physical_to_logical=physical_to_logical,
        )

        mp4_path = output_dir / "aug_grid_transport.mp4"
        render_animation(
            anim,
            output_path=str(mp4_path),
            fps=5,
            dpi=120,
            display_inline=False,
            fallback_to_jshtml=False,
        )

    return arch, compiler, compiled


# =====================================================================
# WISE pipeline
# =====================================================================


def run_wise(
    ideal: stim.Circuit,
    output_dir: Path,
    *,
    animate: bool = True,
) -> Tuple[Any, TrappedIonCompiler, Any]:
    """Compile + visualise on a WISE architecture.

    Returns (arch, compiler, compiled_circuit).
    """
    _print_section("WISE — compile")

    # ── architecture + compiler ──────────────────────────────────────
    wise_cfg = QCCDWiseArch(m=5, n=5, k=2)
    arch = WISEArchitecture(
        wise_config=wise_cfg,
        add_spectators=True,
        compact_clustering=True,
    )
    compiler = TrappedIonCompiler(
        arch, is_wise=True, wise_config=wise_cfg,
    )

    t0 = time.perf_counter()
    compiled = compiler.compile(ideal)
    t1 = time.perf_counter()

    # ── viz mappings ─────────────────────────────────────────────────
    ion_roles, physical_to_logical, ion_idx_remap = build_viz_mappings(compiler)

    # ── metrics ──────────────────────────────────────────────────────
    batches = compiled.scheduled.batches
    all_ops = compiled.scheduled.metadata.get("all_operations", [])
    metrics = compiled.compute_metrics()

    print(f"  Compile time:   {t1 - t0:.2f} s")
    print(f"  Traps:          {len(arch._manipulationTraps)}")
    print(f"  Routed ops:     {len(all_ops)}")
    print(f"  Parallel steps: {len(batches)}")
    print(f"  Depth:          {metrics.get('depth', '?')}")
    print(f"  Duration (µs):  {metrics.get('duration_us', '?')}")
    print(f"  Ion roles:      {ion_roles}")
    if batches:
        print(f"  First batch:    {batches[0].label}")

    # ── static figure ────────────────────────────────────────────────
    _print_section("WISE — static layout")
    arch.resetArrangement()
    arch.refreshGraph()

    fig, ax = display_architecture(
        arch,
        title="WISE — initial layout",
        show_junctions=True,
        show_edges=True,
        show_ions=True,
        show_labels=True,
        show_legend=True,
        ion_roles=ion_roles,
        ion_idx_remap=ion_idx_remap,
        physical_to_logical=physical_to_logical,
    )
    fig_path = output_dir / "wise_layout.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")

    # ── animation ────────────────────────────────────────────────────
    if animate and batches:
        _print_section("WISE — animation")
        arch.resetArrangement()
        arch.refreshGraph()

        anim = animate_transport(
            arch=arch,
            operations=batches,
            interval=600,
            show_labels=True,
            interp_frames=2,
            gate_hold_frames=1,
            stim_circuit=ideal,
            figsize=(16, 12),
            title_prefix="WISE",
            repeat=False,
            ion_roles=ion_roles,
            ion_idx_remap=ion_idx_remap,
            physical_to_logical=physical_to_logical,
        )

        mp4_path = output_dir / "wise_transport.mp4"
        render_animation(
            anim,
            output_path=str(mp4_path),
            fps=5,
            dpi=120,
            display_inline=False,
            fallback_to_jshtml=False,
        )

    return arch, compiler, compiled


# =====================================================================
# Noise + simulation
# =====================================================================


def run_simulation(
    code: Any,
    ideal: stim.Circuit,
    arch: Any,
    compiler: TrappedIonCompiler,
    arch_label: str = "AugmentedGrid",
    num_shots: int = 1_000,
) -> None:
    """Apply noise and decode with PyMatching."""
    _print_section(f"{arch_label} — noisy simulation")

    noise = TrappedIonNoiseModel()

    # Re-build architecture for a fresh compile
    arch.resetArrangement()
    arch.refreshGraph()

    experiment = TrappedIonExperiment(
        code=code,
        architecture=arch,
        compiler=compiler,
        hardware_noise=noise,
        rounds=1,
        basis="z",
    )

    # Build → compile → noise → decode
    ideal_circ = experiment.build_ideal_circuit()
    experiment._compiled = experiment.compiler.compile(ideal_circ)
    noisy_circuit = experiment.apply_hardware_noise()

    print(f"  Noisy circuit instructions: {len(noisy_circuit)}")
    print(f"  Noisy circuit has noise:    "
          f"{any('ERROR' in str(inst) for inst in noisy_circuit)}")

    # Decode with PyMatching
    try:
        from qectostim.decoders.pymatching_decoder import PyMatchingDecoder

        dem = noisy_circuit.detector_error_model(
            decompose_errors=True,
            approximate_disjoint_errors=True,
        )
        decoder = PyMatchingDecoder(dem=dem)

        sampler = noisy_circuit.compile_detector_sampler()
        detection_events, observable_flips = sampler.sample(
            num_shots, separate_observables=True,
        )
        predictions = decoder.decode_batch(detection_events)
        num_errors = int(
            np.sum(np.any(predictions != observable_flips, axis=1))
        )
        logical_error_rate = num_errors / num_shots

        print(f"  Shots:              {num_shots}")
        print(f"  Logical errors:     {num_errors}")
        print(f"  Logical error rate: {logical_error_rate:.4f}")
    except Exception as exc:
        print(f"  Simulation failed: {exc}")
        print("  (PyMatching or detector model may not be available)")


# =====================================================================
# main
# =====================================================================


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end trapped-ion QCCD demo",
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Directory for output files (default: cwd)",
    )
    parser.add_argument(
        "--no-animate", action="store_true",
        help="Skip animation (much faster — figures + metrics only)",
    )
    parser.add_argument(
        "--no-simulate", action="store_true",
        help="Skip noisy simulation (compile + visualise only)",
    )
    parser.add_argument(
        "--shots", type=int, default=1_000,
        help="Number of shots for simulation (default 1000)",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    animate = not args.no_animate
    simulate = not args.no_simulate

    # ── 1. Build ideal circuit ───────────────────────────────────────
    _print_section("Build ideal circuit (d=2, rotated surface code)")

    code = RotatedSurfaceCode(distance=2)
    mem = CSSMemoryExperiment(
        code=code, rounds=1, noise_model=None, basis="z",
    )
    ideal = mem.to_stim()

    print(f"  Code:      d={code.distance}, n={code.n}")
    print(f"  Circuit:   {len(ideal)} instructions, "
          f"{ideal.num_qubits} qubits")

    # ── 2. Augmented Grid ────────────────────────────────────────────
    arch_ag, compiler_ag, compiled_ag = run_augmented_grid(
        ideal, output_dir, animate=animate,
    )

    # ── 3. WISE ──────────────────────────────────────────────────────
    arch_w, compiler_w, compiled_w = run_wise(
        ideal, output_dir, animate=animate,
    )

    # ── 4. Simulation ────────────────────────────────────────────────
    if simulate:
        run_simulation(
            code, ideal, arch_ag, compiler_ag,
            arch_label="Augmented Grid", num_shots=args.shots,
        )

    _print_section("Done")
    print(f"  Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
