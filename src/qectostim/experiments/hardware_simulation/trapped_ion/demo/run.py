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
# Gadget Animation Support
# =====================================================================


def compile_gadget_for_animation(
    ideal_gadget_circuit: stim.Circuit,
    *,
    qec_metadata: Any = None,
    gadget: Any = None,
    qubit_allocation: Any = None,
    trap_capacity: int = 2,
    lookahead: int = 2,
    subgridsize: Optional[Tuple[int, int, int]] = None,
    base_pmax_in: int = 1,
    max_inner_workers: int | None = None,
    show_progress: bool = False,
    routing_config: Optional["WISERoutingConfig"] = None,
    replay_level: Optional[int] = None,
) -> Tuple[Any, TrappedIonCompiler, Any, List[Any], Dict[int, str], Dict[int, int], Dict[int, int]]:
    """Compile a gadget circuit through the full WISE pipeline for animation.

    This bridges :func:`route_full_experiment` (analytics-only, layout arrays)
    and :func:`animate_transport` (needs architecture + operation objects).

    It builds a correctly-sized WISE architecture, runs the compiler
    pipeline manually (decompose → map → route → schedule), and injects
    ``qec_metadata``, ``gadget``, and ``qubit_allocation`` into the
    ``NativeCircuit`` metadata so that ``TrappedIonCompiler.route()``
    can dispatch to ``ionRoutingGadgetArch`` for phase-aware routing.

    Parameters
    ----------
    ideal_gadget_circuit : stim.Circuit
        Ideal circuit from ``FaultTolerantGadgetExperiment.to_stim()``.
    qec_metadata : QECMetadata, optional
        Phase/block information from ``ft_experiment.qec_metadata``.
        **Required** for phase-aware gadget routing.
    gadget : Gadget, optional
        Gadget object (e.g. ``TransversalCNOTGadget()``).
    qubit_allocation : QubitAllocation, optional
        Unified allocation from ``ft_experiment._unified_allocation``.
    trap_capacity : int
        Ions per trap (k).
    lookahead : int
        SAT solver lookahead window.  Ignored if *routing_config* is set.
    subgridsize : tuple or None
        ``(width, height, increment)`` for patch decomposition.
        ``None`` means no cap (full-grid SAT).  Ignored if *routing_config* is set.
    base_pmax_in : int
        Base pass horizon for SAT solver.  Ignored if *routing_config* is set.
    max_inner_workers : int | None
        Max parallel SAT workers per patch (None = all cores, 1 = serial).
    show_progress : bool
        Display tqdm progress bars during routing.
    routing_config : WISERoutingConfig, optional
        Pre-built routing config.  When provided, *lookahead*,
        *subgridsize*, *base_pmax_in*, and *replay_level* are ignored.
    replay_level : int, optional
        Cache replay aggressiveness.  ``0`` = no caching (pure SAT,
        slowest, guaranteed optimal); ``1`` = per-round caching
        (fastest, good approximation); ``d`` (e.g. ``2`` for d=2) =
        cache per EC block.  Default (``None``) inherits from
        *routing_config*, or ``1`` if constructing a fresh config.
        Ignored when *routing_config* is provided.

    Returns
    -------
    arch : WISEArchitecture
        The architecture instance (needed by ``animate_transport``).
    compiler : TrappedIonCompiler
        The compiler (has ``ion_mapping`` for ``build_viz_mappings``).
    compiled : CompiledCircuit
        Full compilation result.
    batches : list
        ``compiled.scheduled.batches`` — operation objects for animation.
    ion_roles : dict
        ``{ion_idx: role}`` for viz coloring.
    physical_to_logical : dict
        ``{ion_idx: stim_qubit_idx}``.
    ion_idx_remap : dict
        ``{ion_idx: dense_index}``.
    """
    from ..compiler.routing_config import WISERoutingConfig
    from qectostim.experiments.hardware_simulation.core.pipeline import (
        CompiledCircuit,
    )
    from ..utils.gadget_routing import partition_grid_for_blocks

    # ------------------------------------------------------------------
    # Grid sizing: per-block sub-grids if metadata available, else global
    # ------------------------------------------------------------------
    _use_per_block = (
        qec_metadata is not None
        and qubit_allocation is not None
        and hasattr(qec_metadata, 'block_allocations')
        and len(qec_metadata.block_allocations) >= 2
    )

    if qec_metadata is not None and qubit_allocation is None:
        import warnings
        warnings.warn(
            "compile_gadget_for_animation: qec_metadata provided but "
            "qubit_allocation is None — falling back to global mapping "
            "which will mangle block ions together. Pass qubit_allocation "
            "for correct per-block spatial partitioning.",
            stacklevel=2,
        )

    if _use_per_block:
        # §3.1.1: allocate disjoint sub-grid regions per block
        sub_grids = partition_grid_for_blocks(
            qec_metadata, qubit_allocation, k=trap_capacity,
        )
        max_r = max(sg.grid_region[2] for sg in sub_grids.values())
        max_c = max(sg.grid_region[3] for sg in sub_grids.values())
        n_traps = max_r   # rows
        m_traps = max_c   # trap columns
    else:
        sub_grids = None
        # Fallback: global grid sizing (same formula as _route_and_simulate)
        num_qubits = ideal_gadget_circuit.num_qubits
        nqubits_needed = 4 * (np.ceil(num_qubits / 3))
        n_traps = int(np.ceil(np.sqrt(nqubits_needed)))
        m_traps = int(np.ceil(n_traps / trap_capacity))

    # Build architecture + compiler
    wise_cfg = QCCDWiseArch(m=m_traps, n=n_traps, k=trap_capacity)
    arch = WISEArchitecture(
        wise_config=wise_cfg,
        add_spectators=True,
        compact_clustering=True,
    )
    compiler = TrappedIonCompiler(
        arch, is_wise=True, wise_config=wise_cfg,
        show_progress=False,  # We handle progress display ourselves below
    )

    # ── Progress: prefer ipywidgets in notebooks, tqdm elsewhere ──
    _progress_close_fn = None
    if show_progress:
        try:
            from ..utils.progress_table import (
                make_notebook_widget_progress_callback,
                _in_notebook,
            )
            if _in_notebook():
                from IPython.display import display as _ipy_display
                _widget_container, _widget_cb, _progress_close_fn = (
                    make_notebook_widget_progress_callback("WISE Routing")
                )
                _ipy_display(_widget_container)
                _user_cb = _widget_cb
            else:
                raise ImportError("not in notebook")
        except Exception:
            # Fallback: tqdm text bars
            from ..compiler.routing_config import make_triple_tqdm_progress_callback
            _user_cb, _progress_close_fn = make_triple_tqdm_progress_callback(
                round_desc="MS Rounds",
                patch_desc="Patches",
                sat_desc="SAT Configs",
            )
    else:
        _user_cb = None

    # Build or reuse routing config
    if routing_config is not None:
        _rc = routing_config
    else:
        _rc_kwargs: dict = dict(
            lookahead=lookahead,
            subgridsize=subgridsize if subgridsize is not None else (4, 3, 0),
            base_pmax_in=base_pmax_in,
            show_progress=False,  # Disable internal tqdm — we set our own
        )
        if replay_level is not None:
            _rc_kwargs["replay_level"] = replay_level
        _rc = WISERoutingConfig.default(**_rc_kwargs)
    compiler.routing_kwargs = dict(
        routing_config=_rc,
        max_inner_workers=max_inner_workers,
    )

    # Inject our widget/tqdm callback into the routing config
    if _user_cb is not None:
        _rc.progress_callback = _user_cb
        _rc._progress_close = _progress_close_fn

    # Run compiler pipeline MANUALLY so we can inject gadget metadata
    # into the NativeCircuit's metadata dict before route() reads it.
    native = compiler.decompose_to_native(ideal_gadget_circuit)

    # Inject gadget metadata so route() dispatches to ionRoutingGadgetArch
    if qec_metadata is not None:
        native.qec_metadata = qec_metadata
        native.metadata["qec_metadata"] = qec_metadata
    if gadget is not None:
        native.metadata["gadget"] = gadget
    if qubit_allocation is not None:
        native.metadata["qubit_allocation"] = qubit_allocation

    # ------------------------------------------------------------------
    # Qubit-to-ion mapping: per-block or global
    # ------------------------------------------------------------------
    # Delegate to the compiler's map_qubits(), which handles both the
    # per-block (_map_qubits_per_block) and global paths.  The injected
    # metadata (qec_metadata, qubit_allocation) will trigger the
    # per-block path automatically when multiple blocks are present.
    mapped = compiler.map_qubits(native)

    try:
        routed = compiler.route(mapped)
    finally:
        # Close progress display even if routing raises
        if _progress_close_fn is not None:
            try:
                _progress_close_fn()
            except Exception:
                pass

    scheduled = compiler.schedule(routed)

    compiled = CompiledCircuit(
        scheduled=scheduled,
        mapping=routed.final_mapping or mapped.mapping,
        original_circuit=ideal_gadget_circuit,
    )
    compiled.compute_metrics()

    # Extract animation artifacts
    batches = compiled.scheduled.batches
    ion_roles, physical_to_logical, ion_idx_remap = build_viz_mappings(compiler)

    return arch, compiler, compiled, batches, ion_roles, physical_to_logical, ion_idx_remap


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
