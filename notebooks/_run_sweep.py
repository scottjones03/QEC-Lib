#!/usr/bin/env python3
"""
Fast Code × Gadget sweep with process-level hard timeout.
Saves results to _sweep_results.json for notebook consumption.
"""
import sys, os, time, json, multiprocessing
import numpy as np

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_src = os.path.join(_repo_root, 'src')
if _src not in sys.path:
    sys.path.insert(0, _src)

# ── Config ───────────────────────────────────────────────────────────
P       = 3e-3
SHOTS   = 100
MAX_N   = 50      # skip codes with n > this
TIMEOUT = 15      # hard kill per combo (seconds)
MAX_DET = 500     # skip combos with more detectors than this

# ── Helpers (picklable — used in both main + worker) ─────────────────
def _code_distance(code):
    if hasattr(code, 'distance'):
        d = code.distance
        if isinstance(d, int) and d >= 1:
            return d
    return 3

def _build_dem(circuit):
    for strat in ['decompose', 'ignore', 'raw']:
        try:
            if strat == 'decompose':
                return circuit.detector_error_model(decompose_errors=True)
            elif strat == 'ignore':
                return circuit.detector_error_model(
                    decompose_errors=True, ignore_decomposition_failures=True)
            else:
                return circuit.detector_error_model(decompose_errors=False)
        except Exception:
            continue
    raise RuntimeError('Cannot build DEM with any strategy')

# ── Worker — runs in a forked subprocess ─────────────────────────────
def _worker(code_eval, gadget_eval, code_name, gadget_name, family, q):
    """Evaluate code+gadget from strings, build circuit, decode, put result."""
    try:
        # All imports happen inside the worker
        from qectostim.codes.small.steane_713 import SteaneCode713
        from qectostim.codes.small.hamming_css import HammingCSSCode
        from qectostim.codes.small.four_two_two import FourQubit422Code
        from qectostim.codes.small.six_two_two import SixQubit622Code
        from qectostim.codes.small.eight_three_two import EightThreeTwoCode
        from qectostim.codes.small.shor_code import ShorCode91
        from qectostim.codes.small.reed_muller_code import ReedMullerCode151
        from qectostim.codes.small.repetition_codes import RepetitionCode
        from qectostim.codes.color.triangular_colour import TriangularColourCode
        from qectostim.codes.color.colour_code import ColourCode488
        from qectostim.codes.color.hexagonal_colour import HexagonalColourCode
        from qectostim.codes.color.extended_color import (
            TruncatedTrihexColorCode, HyperbolicColorCode,
            BallColorCode, CubicHoneycombColorCode, TetrahedralColorCode)
        from qectostim.codes.color.color_3d import ColorCode3D, ColorCode3DPrism
        from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
        from qectostim.codes.surface.xzzx_surface import XZZXSurfaceCode
        from qectostim.codes.surface.toric_code import ToricCode33
        from qectostim.codes.surface.toric_code_general import ToricCode
        from qectostim.codes.surface.toric_3d import ToricCode3D, ToricCode3DFaces
        from qectostim.codes.surface.toric_4d import ToricCode4D
        from qectostim.codes.surface.four_d_surface_code import FourDSurfaceCode
        from qectostim.codes.surface.hyperbolic import (
            Hyperbolic45Code, Hyperbolic57Code, Hyperbolic38Code,
            FreedmanMeyerLuoCode, GuthLubotzkyCode, GoldenCode)
        from qectostim.codes.surface.exotic_surface import (
            FractalSurfaceCode, TwistedToricCode, ProjectivePlaneSurfaceCode,
            LCSCode, LoopToricCode4D)
        from qectostim.codes.topological.fracton_codes import XCubeCode, HaahCode, CheckerboardCode
        from qectostim.codes.subsystem.bacon_shor import BaconShorCode
        from qectostim.codes.generic.css_generic import GenericCSSCode
        from qectostim.gadgets.transversal_cnot import TransversalCNOTGadget
        from qectostim.gadgets.teleportation_h_gadgets import CNOTHTeleportGadget
        from qectostim.gadgets.knill_ec import KnillECGadget
        from qectostim.gadgets.css_surgery_cnot import CSSSurgeryCNOTGadget
        from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
        from qectostim.noise.models import CircuitDepolarizingNoise
        from qectostim.decoders.concat_mle_decoder import ConcatMLEDecoder

        code = eval(code_eval)
        n_q, k_q = code.n, code.k
        _d = _code_distance(code)
        gadget = eval(gadget_eval)

        noise = CircuitDepolarizingNoise(p1=P, p2=P)
        rb = _d + 1 if isinstance(gadget, (CNOTHTeleportGadget, KnillECGadget)) else _d

        exp = FaultTolerantGadgetExperiment(
            codes=[code], gadget=gadget, noise_model=noise,
            num_rounds_before=rb, num_rounds_after=_d)
        circuit = exp.to_stim()
        nd = circuit.num_detectors

        if nd > MAX_DET:
            q.put(dict(Code=code_name, Family=family, n=n_q, k=k_q,
                       Gadget=gadget_name, LER=None,
                       Status="SKIP_DET", Detail=f"det={nd}", Time=-1))
            return

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

        q.put(dict(Code=code_name, Family=family, n=n_q, k=k_q,
                   Gadget=gadget_name, LER=ler,
                   Status="OK", Detail=f"{num_errors}/{SHOTS} det={nd}", Time=-1))
    except Exception as e:
        q.put(dict(Code=code_name, Family=family, n=-1, k=-1,
                   Gadget=gadget_name, LER=None,
                   Status="ERROR", Detail=str(e)[:80], Time=-1))

# ── Registries (eval-able strings) ───────────────────────────────────
CODE_SPECS = [
    ("Steane [[7,1,3]]",       "SteaneCode713()",                   "Small"),
    ("Hamming CSS",            "HammingCSSCode()",                   "Small"),
    ("[[4,2,2]]",              "FourQubit422Code()",                 "Small"),
    ("[[6,2,2]]",              "SixQubit622Code()",                  "Small"),
    ("[[8,3,2]]",              "EightThreeTwoCode()",                "Small"),
    ("Shor [[9,1,3]]",         "ShorCode91()",                       "Small"),
    ("Reed-Muller [[15,1,5]]", "ReedMullerCode151()",                "Small"),
    ("Repetition(3)",          "RepetitionCode(3)",                   "Small"),
    ("Tri-Colour d=3",         "TriangularColourCode(distance=3)",   "Color"),
    ("Colour-488 d=3",         "ColourCode488(distance=3)",          "Color"),
    ("Hex-Colour d=2",         "HexagonalColourCode(distance=2)",    "Color"),
    ("Trunc-Trihex",           "TruncatedTrihexColorCode()",         "Color"),
    ("Hyperbolic-Color",       "HyperbolicColorCode()",              "Color"),
    ("Ball-Color",             "BallColorCode()",                    "Color"),
    ("Cubic-Honeycomb",        "CubicHoneycombColorCode()",          "Color"),
    ("Tetrahedral-Color",      "TetrahedralColorCode()",             "Color"),
    ("Color-3D",               "ColorCode3D()",                      "Color"),
    ("Color-3D-Prism",         "ColorCode3DPrism()",                 "Color"),
    ("RotSurf d=3",            "RotatedSurfaceCode(3)",              "Surface"),
    ("XZZX d=3",               "XZZXSurfaceCode(distance=3)",       "Surface"),
    ("Toric [[18,2,3]]",       "ToricCode33()",                      "Surface"),
    ("Toric 3×3",              "ToricCode(3, 3)",                    "Surface"),
    ("Toric-3D",               "ToricCode3D()",                      "Surface"),
    ("Toric-3D-Faces",         "ToricCode3DFaces()",                 "Surface"),
    ("Toric-4D",               "ToricCode4D()",                      "Surface"),
    ("4D-Surface",             "FourDSurfaceCode()",                 "Surface"),
    ("Hyperbolic {4,5}",       "Hyperbolic45Code()",                 "Surface"),
    ("Hyperbolic {5,7}",       "Hyperbolic57Code()",                 "Surface"),
    ("Hyperbolic {3,8}",       "Hyperbolic38Code()",                 "Surface"),
    ("Freedman-Meyer-Luo",     "FreedmanMeyerLuoCode()",             "Surface"),
    ("Guth-Lubotzky",          "GuthLubotzkyCode()",                 "Surface"),
    ("Golden",                 "GoldenCode()",                       "Surface"),
    ("Fractal-Surface",        "FractalSurfaceCode()",               "Surface"),
    ("Twisted-Toric",          "TwistedToricCode()",                 "Surface"),
    ("Projective-Plane",       "ProjectivePlaneSurfaceCode()",       "Surface"),
    ("LCS",                    "LCSCode()",                          "Surface"),
    ("Loop-Toric-4D",          "LoopToricCode4D()",                  "Surface"),
    ("X-Cube L=2",             "XCubeCode(L=2)",                    "Fracton"),
    ("Haah L=2",               "HaahCode(L=2)",                     "Fracton"),
    ("Checkerboard L=2",       "CheckerboardCode(L=2)",             "Fracton"),
    ("Bacon-Shor(3)",          "BaconShorCode(3)",                   "Other"),
    ("GenericCSS (Steane)",    "GenericCSSCode(hx=SteaneCode713().hx, hz=SteaneCode713().hz)", "Other"),
]

GADGET_SPECS = [
    ("TransversalCNOT",  "TransversalCNOTGadget()"),
    ("CNOT H-Teleport",  "CNOTHTeleportGadget(num_ec_rounds=max(_d, 2))"),
    ("Knill EC",         "KnillECGadget(num_ec_rounds=max(_d, 2))"),
    ("CSS Surgery CNOT", "CSSSurgeryCNOTGadget(control_state='0', target_state='0', merge_rounds=1)"),
]

# ── Main ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    multiprocessing.set_start_method('fork', force=True)

    # Pre-check code sizes in main process (very fast)
    code_n = {}
    for cname, ceval, cfam in CODE_SPECS:
        try:
            from qectostim.codes.small.steane_713 import SteaneCode713
            from qectostim.codes.small.hamming_css import HammingCSSCode
            from qectostim.codes.small.four_two_two import FourQubit422Code
            from qectostim.codes.small.six_two_two import SixQubit622Code
            from qectostim.codes.small.eight_three_two import EightThreeTwoCode
            from qectostim.codes.small.shor_code import ShorCode91
            from qectostim.codes.small.reed_muller_code import ReedMullerCode151
            from qectostim.codes.small.repetition_codes import RepetitionCode
            from qectostim.codes.color.triangular_colour import TriangularColourCode
            from qectostim.codes.color.colour_code import ColourCode488
            from qectostim.codes.color.hexagonal_colour import HexagonalColourCode
            from qectostim.codes.color.extended_color import (
                TruncatedTrihexColorCode, HyperbolicColorCode,
                BallColorCode, CubicHoneycombColorCode, TetrahedralColorCode)
            from qectostim.codes.color.color_3d import ColorCode3D, ColorCode3DPrism
            from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
            from qectostim.codes.surface.xzzx_surface import XZZXSurfaceCode
            from qectostim.codes.surface.toric_code import ToricCode33
            from qectostim.codes.surface.toric_code_general import ToricCode
            from qectostim.codes.surface.toric_3d import ToricCode3D, ToricCode3DFaces
            from qectostim.codes.surface.toric_4d import ToricCode4D
            from qectostim.codes.surface.four_d_surface_code import FourDSurfaceCode
            from qectostim.codes.surface.hyperbolic import (
                Hyperbolic45Code, Hyperbolic57Code, Hyperbolic38Code,
                FreedmanMeyerLuoCode, GuthLubotzkyCode, GoldenCode)
            from qectostim.codes.surface.exotic_surface import (
                FractalSurfaceCode, TwistedToricCode, ProjectivePlaneSurfaceCode,
                LCSCode, LoopToricCode4D)
            from qectostim.codes.topological.fracton_codes import XCubeCode, HaahCode, CheckerboardCode
            from qectostim.codes.subsystem.bacon_shor import BaconShorCode
            from qectostim.codes.generic.css_generic import GenericCSSCode
            c = eval(ceval)
            code_n[cname] = c.n
        except Exception as e:
            code_n[cname] = -1
            print(f"  {cname:25s}  CODE_ERR: {e}")

    rows = []
    total = len(CODE_SPECS) * len(GADGET_SPECS)
    done = 0
    t_global = time.time()

    print(f"Sweep: {len(CODE_SPECS)} codes × {len(GADGET_SPECS)} gadgets = {total} combos")
    print(f"p={P:.0e} shots={SHOTS} max_n={MAX_N} timeout={TIMEOUT}s max_det={MAX_DET}")
    print("=" * 90)

    for code_spec in CODE_SPECS:
        cname, ceval, cfam = code_spec
        nq = code_n.get(cname, -1)

        if nq < 0:
            for gname, _ in GADGET_SPECS:
                done += 1
                rows.append(dict(Code=cname, Family=cfam, n=-1, k=-1,
                                 Gadget=gname, LER=None,
                                 Status="CODE_ERR", Detail="instantiation", Time=-1))
            print(f"  {cname:25s}  ⏭ CODE_ERR")
            continue

        if nq > MAX_N:
            for gname, _ in GADGET_SPECS:
                done += 1
                rows.append(dict(Code=cname, Family=cfam, n=nq, k=-1,
                                 Gadget=gname, LER=None,
                                 Status="SKIP_LARGE", Detail=f"n={nq}", Time=-1))
            print(f"  {cname:25s}  ⏭ SKIP (n={nq})")
            continue

        for gname, geval in GADGET_SPECS:
            done += 1
            t0 = time.time()
            q = multiprocessing.Queue()
            proc = multiprocessing.Process(
                target=_worker,
                args=(ceval, geval, cname, gname, cfam, q))
            proc.start()
            proc.join(timeout=TIMEOUT)

            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=3)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
                dt = time.time() - t0
                print(f"  [{done:3d}/{total}] {cname:25s} × {gname:18s}  ⏭ TIMEOUT ({dt:.0f}s)")
                rows.append(dict(Code=cname, Family=cfam, n=nq, k=-1,
                                 Gadget=gname, LER=None,
                                 Status="TIMEOUT", Detail=f">{TIMEOUT}s", Time=round(dt,1)))
            elif not q.empty():
                result = q.get_nowait()
                dt = time.time() - t0
                result['Time'] = round(dt, 1)
                rows.append(result)
                st = result['Status']
                if st == 'OK':
                    print(f"  [{done:3d}/{total}] {cname:25s} × {gname:18s}  "
                          f"LER={result['LER']:.4f}  {result['Detail']}  {dt:.1f}s")
                elif st == 'SKIP_DET':
                    print(f"  [{done:3d}/{total}] {cname:25s} × {gname:18s}  "
                          f"⏭ {result['Detail']}  {dt:.1f}s")
                else:
                    print(f"  [{done:3d}/{total}] {cname:25s} × {gname:18s}  "
                          f"✗ {st}: {result['Detail'][:50]}  {dt:.1f}s")
            else:
                dt = time.time() - t0
                print(f"  [{done:3d}/{total}] {cname:25s} × {gname:18s}  ✗ NO_RESULT  {dt:.1f}s")
                rows.append(dict(Code=cname, Family=cfam, n=nq, k=-1,
                                 Gadget=gname, LER=None,
                                 Status="ERROR", Detail="no result", Time=round(dt,1)))

    elapsed = time.time() - t_global
    n_ok   = sum(1 for r in rows if r['Status'] == 'OK')
    n_skip = sum(1 for r in rows if r['Status'] in ('SKIP_LARGE','TIMEOUT','SKIP_DET'))
    n_err  = sum(1 for r in rows if r['Status'] in ('ERROR','CODE_ERR'))
    print(f"\n{'='*90}")
    print(f"Done in {elapsed:.0f}s  OK={n_ok}  skip={n_skip}  err={n_err}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_sweep_results.json')
    with open(out_path, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"Saved → {out_path}")
