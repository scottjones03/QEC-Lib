#!/usr/bin/env python3
"""Fast Code × Gadget sweep — saves results to JSON for notebook use."""
import sys, os, time, json, signal, traceback
import numpy as np

# Ensure src/ is on path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_repo_root, 'src'))

# ── Imports ──────────────────────────────────────────────────────────
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.noise.models import CircuitDepolarizingNoise
from qectostim.decoders.concat_mle_decoder import ConcatMLEDecoder

# Small codes
from qectostim.codes.small.steane_713 import SteaneCode713
from qectostim.codes.small.hamming_css import HammingCSSCode
from qectostim.codes.small.four_two_two import FourQubit422Code
from qectostim.codes.small.six_two_two import SixQubit622Code
from qectostim.codes.small.eight_three_two import EightThreeTwoCode
from qectostim.codes.small.shor_code import ShorCode91
from qectostim.codes.small.reed_muller_code import ReedMullerCode151
from qectostim.codes.small.repetition_codes import RepetitionCode

# Color codes
from qectostim.codes.color.triangular_colour import TriangularColourCode
from qectostim.codes.color.colour_code import ColourCode488
from qectostim.codes.color.hexagonal_colour import HexagonalColourCode
from qectostim.codes.color.extended_color import (
    TruncatedTrihexColorCode, HyperbolicColorCode,
    BallColorCode, CubicHoneycombColorCode, TetrahedralColorCode,
)
from qectostim.codes.color.color_3d import ColorCode3D, ColorCode3DPrism

# Surface codes
from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
from qectostim.codes.surface.xzzx_surface import XZZXSurfaceCode
from qectostim.codes.surface.toric_code import ToricCode33
from qectostim.codes.surface.toric_code_general import ToricCode
from qectostim.codes.surface.toric_3d import ToricCode3D, ToricCode3DFaces
from qectostim.codes.surface.toric_4d import ToricCode4D
from qectostim.codes.surface.four_d_surface_code import FourDSurfaceCode
from qectostim.codes.surface.hyperbolic import (
    Hyperbolic45Code, Hyperbolic57Code, Hyperbolic38Code,
    FreedmanMeyerLuoCode, GuthLubotzkyCode, GoldenCode,
)
from qectostim.codes.surface.exotic_surface import (
    FractalSurfaceCode, TwistedToricCode, ProjectivePlaneSurfaceCode,
    LCSCode, LoopToricCode4D,
)
from qectostim.codes.topological.fracton_codes import XCubeCode, HaahCode, CheckerboardCode
from qectostim.codes.subsystem.bacon_shor import BaconShorCode
from qectostim.codes.generic.css_generic import GenericCSSCode

# Gadgets
from qectostim.gadgets.transversal_cnot import TransversalCNOTGadget
from qectostim.gadgets.teleportation_h_gadgets import CNOTHTeleportGadget
from qectostim.gadgets.knill_ec import KnillECGadget
from qectostim.gadgets.css_surgery_cnot import CSSSurgeryCNOTGadget

# ── Config ───────────────────────────────────────────────────────────
P = 3e-3
SHOTS = 100
MAX_N = 50        # skip codes with n > this
TIMEOUT = 20      # per-combo timeout (seconds)
MAX_DET = 600     # skip if detector count exceeds this

# ── Helpers ──────────────────────────────────────────────────────────
def _code_distance(code):
    if hasattr(code, 'distance'):
        d = code.distance
        if isinstance(d, int) and d >= 1:
            return d
    return 3

def _build_dem(circuit):
    for strategy in ['decompose', 'ignore_failures', 'no_decompose']:
        try:
            if strategy == 'decompose':
                return circuit.detector_error_model(decompose_errors=True)
            elif strategy == 'ignore_failures':
                return circuit.detector_error_model(
                    decompose_errors=True,
                    ignore_decomposition_failures=True,
                )
            else:
                return circuit.detector_error_model(decompose_errors=False)
        except Exception:
            continue
    raise RuntimeError('Cannot build DEM with any strategy')

class _Timeout(Exception):
    pass

def _alarm_handler(signum, frame):
    raise _Timeout("timeout")

# ── Code registry ────────────────────────────────────────────────────
_steane_hx = SteaneCode713().hx
_steane_hz = SteaneCode713().hz

ALL_CODES = [
    ("Steane [[7,1,3]]",        lambda: SteaneCode713(),              "Small"),
    ("Hamming CSS",             lambda: HammingCSSCode(),             "Small"),
    ("[[4,2,2]]",               lambda: FourQubit422Code(),           "Small"),
    ("[[6,2,2]]",               lambda: SixQubit622Code(),            "Small"),
    ("[[8,3,2]]",               lambda: EightThreeTwoCode(),          "Small"),
    ("Shor [[9,1,3]]",          lambda: ShorCode91(),                 "Small"),
    ("Reed-Muller [[15,1,5]]",  lambda: ReedMullerCode151(),          "Small"),
    ("Repetition(3)",           lambda: RepetitionCode(3),            "Small"),
    ("Tri-Colour d=3",          lambda: TriangularColourCode(distance=3), "Color"),
    ("Colour-488 d=3",          lambda: ColourCode488(distance=3),        "Color"),
    ("Hex-Colour d=2",          lambda: HexagonalColourCode(distance=2),  "Color"),
    ("Trunc-Trihex",            lambda: TruncatedTrihexColorCode(),       "Color"),
    ("Hyperbolic-Color",        lambda: HyperbolicColorCode(),            "Color"),
    ("Ball-Color",              lambda: BallColorCode(),                  "Color"),
    ("Cubic-Honeycomb",         lambda: CubicHoneycombColorCode(),        "Color"),
    ("Tetrahedral-Color",       lambda: TetrahedralColorCode(),           "Color"),
    ("Color-3D",                lambda: ColorCode3D(),                    "Color"),
    ("Color-3D-Prism",          lambda: ColorCode3DPrism(),               "Color"),
    ("RotSurf d=3",             lambda: RotatedSurfaceCode(3),            "Surface"),
    ("XZZX d=3",                lambda: XZZXSurfaceCode(distance=3),      "Surface"),
    ("Toric [[18,2,3]]",        lambda: ToricCode33(),                    "Surface"),
    ("Toric 3×3",               lambda: ToricCode(3, 3),                  "Surface"),
    ("Toric-3D",                lambda: ToricCode3D(),                    "Surface"),
    ("Toric-3D-Faces",          lambda: ToricCode3DFaces(),               "Surface"),
    ("Toric-4D",                lambda: ToricCode4D(),                    "Surface"),
    ("4D-Surface",              lambda: FourDSurfaceCode(),               "Surface"),
    ("Hyperbolic {4,5}",        lambda: Hyperbolic45Code(),               "Surface"),
    ("Hyperbolic {5,7}",        lambda: Hyperbolic57Code(),               "Surface"),
    ("Hyperbolic {3,8}",        lambda: Hyperbolic38Code(),               "Surface"),
    ("Freedman-Meyer-Luo",      lambda: FreedmanMeyerLuoCode(),           "Surface"),
    ("Guth-Lubotzky",           lambda: GuthLubotzkyCode(),               "Surface"),
    ("Golden",                  lambda: GoldenCode(),                     "Surface"),
    ("Fractal-Surface",         lambda: FractalSurfaceCode(),             "Surface"),
    ("Twisted-Toric",           lambda: TwistedToricCode(),               "Surface"),
    ("Projective-Plane",        lambda: ProjectivePlaneSurfaceCode(),     "Surface"),
    ("LCS",                     lambda: LCSCode(),                        "Surface"),
    ("Loop-Toric-4D",           lambda: LoopToricCode4D(),                "Surface"),
    ("X-Cube L=2",              lambda: XCubeCode(L=2),                   "Fracton"),
    ("Haah L=2",                lambda: HaahCode(L=2),                    "Fracton"),
    ("Checkerboard L=2",        lambda: CheckerboardCode(L=2),            "Fracton"),
    ("Bacon-Shor(3)",           lambda: BaconShorCode(3),                 "Other"),
    ("GenericCSS (Steane)",     lambda: GenericCSSCode(hx=_steane_hx, hz=_steane_hz), "Other"),
]

ALL_GADGETS = [
    ("TransversalCNOT",
     lambda d: TransversalCNOTGadget()),
    ("CNOT H-Teleport",
     lambda d: CNOTHTeleportGadget(num_ec_rounds=max(d, 2))),
    ("Knill EC",
     lambda d: KnillECGadget(num_ec_rounds=max(d, 2))),
    ("CSS Surgery CNOT",
     lambda d: CSSSurgeryCNOTGadget(control_state="0", target_state="0", merge_rounds=1)),
]

# ── Main sweep ───────────────────────────────────────────────────────
if __name__ == '__main__':
    rows = []
    total = len(ALL_CODES) * len(ALL_GADGETS)
    done = 0
    t_start = time.time()

    print(f"Phase 5 Sweep: {len(ALL_CODES)} codes × {len(ALL_GADGETS)} gadgets = {total} combos")
    print(f"p={P:.0e}, shots={SHOTS}, max_n={MAX_N}, timeout={TIMEOUT}s, max_det={MAX_DET}")
    print("=" * 90)

    for code_name, code_fn, family in ALL_CODES:
        try:
            code = code_fn()
        except Exception as e:
            for gname, _ in ALL_GADGETS:
                done += 1
                rows.append(dict(Code=code_name, Family=family, n=-1, k=-1,
                                 Gadget=gname, LER=None,
                                 Status="CODE_ERR", Detail=str(e)[:80], Time=-1))
            print(f"  {code_name:25s}  ⏭ CODE_ERR: {e}")
            continue

        n_q = code.n
        k_q = code.k
        d = _code_distance(code)

        if n_q > MAX_N:
            for gname, _ in ALL_GADGETS:
                done += 1
                rows.append(dict(Code=code_name, Family=family, n=n_q, k=k_q,
                                 Gadget=gname, LER=None,
                                 Status="SKIP_LARGE", Detail=f"n={n_q}", Time=-1))
            print(f"  {code_name:25s}  ⏭ SKIP (n={n_q})")
            continue

        for gadget_name, gadget_fn in ALL_GADGETS:
            done += 1
            label = f"[{done:3d}/{total}] {code_name:25s} × {gadget_name:18s}"
            t0 = time.time()

            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(TIMEOUT)
            try:
                gadget = gadget_fn(d)
                noise = CircuitDepolarizingNoise(p1=P, p2=P)
                rb = d + 1 if isinstance(gadget, (CNOTHTeleportGadget, KnillECGadget)) else d
                exp = FaultTolerantGadgetExperiment(
                    codes=[code], gadget=gadget, noise_model=noise,
                    num_rounds_before=rb, num_rounds_after=d,
                )
                circuit = exp.to_stim()

                nd = circuit.num_detectors
                if nd > MAX_DET:
                    raise _Timeout(f"det={nd}>{MAX_DET}")

                dem = _build_dem(circuit)
                decoder = ConcatMLEDecoder(dem=dem)
                sampler = dem.compile_sampler()

                det_samples, obs_samples, _ = sampler.sample(shots=SHOTS)
                corrections = decoder.decode_batch(det_samples)
                predicted_obs = corrections % 2
                actual_obs = obs_samples.astype(np.uint8)
                errors = np.any(predicted_obs != actual_obs, axis=1)
                num_errors = int(errors.sum())
                ler = num_errors / SHOTS
                dt = time.time() - t0

                print(f"  {label}  LER={ler:.4f} ({num_errors:3d}/{SHOTS})  "
                      f"n={n_q} det={nd:4d}  {dt:.1f}s")
                rows.append(dict(Code=code_name, Family=family, n=n_q, k=k_q,
                                 Gadget=gadget_name, LER=ler,
                                 Status="OK", Detail=f"{num_errors}/{SHOTS}",
                                 Time=round(dt, 1)))

            except _Timeout as te:
                dt = time.time() - t0
                print(f"  {label}  ⏭ TIMEOUT ({dt:.0f}s) {te}")
                rows.append(dict(Code=code_name, Family=family, n=n_q, k=k_q,
                                 Gadget=gadget_name, LER=None,
                                 Status="TIMEOUT", Detail=str(te)[:60], Time=round(dt, 1)))
            except Exception as e:
                dt = time.time() - t0
                print(f"  {label}  ✗ ERR: {str(e)[:60]}  {dt:.1f}s")
                rows.append(dict(Code=code_name, Family=family, n=n_q, k=k_q,
                                 Gadget=gadget_name, LER=None,
                                 Status="ERROR", Detail=str(e)[:60], Time=round(dt, 1)))
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    elapsed = time.time() - t_start
    n_ok   = sum(1 for r in rows if r['Status'] == 'OK')
    n_skip = sum(1 for r in rows if r['Status'] in ('SKIP_LARGE', 'TIMEOUT'))
    n_err  = sum(1 for r in rows if r['Status'] in ('ERROR', 'CODE_ERR'))
    print(f"\n{'='*90}")
    print(f"Done in {elapsed:.0f}s: {n_ok} OK, {n_skip} skipped/timeout, {n_err} errors")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), '_sweep_results.json')
    with open(out_path, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"Results saved to {out_path}")
