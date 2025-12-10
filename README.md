# Comprehensive Quantum Error Correction Library (QEC-Lib)

A modular and extensible Python framework for building, simulating, and decoding quantum error-correcting codes.

---

# QEC-Lib — Comprehensive Quantum Error Correction Library

> This project is actively evolving — documentation and APIs may change rapidly. If you find inconsistencies or have questions, please open an issue: https://github.com/scottjones03/qec-lib/issues/new/choose

![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue)
[![Stars](https://img.shields.io/github/stars/scottjones03/qec-lib.svg)](https://github.com/scottjones03/qec-lib/stargazers)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

QEC-Lib is a modular and extensible Python framework for building, simulating, and decoding quantum error-correcting codes. It focuses on CSS-style stabilizer codes, fault-tolerant memory experiments, Stim-based circuit generation, and multiple decoder backends for benchmarking and research.

## Highlights

- Extensible base classes for many stabilizer code families (surface, Steane, Shor, Reed-Muller, etc.)
- Stim integration for fast circuit + detector-error-model (DEM) generation
- Multiple decoder backends (PyMatching / MWPM, Union-Find, Fusion Blossom, BP+OSD)
- Simulation utilities for memory experiments, logical-gate gadgets, and benchmarking (LER, NDR)
- Designed for research: modular, instrumentable, and testable

## Quick Start

Install from source (recommended for latest features):

```bash
python -m pip install git+https://github.com/scottjones03/qec-lib.git
```

Or for local development:

```bash
git clone https://github.com.scottjones03/qec-lib.git
cd qec-lib
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

See `INSTALLATION.md` for more platform-specific tips and troubleshooting.

## Basic Usage

Generate a memory experiment for a concatenated surface-steane code and run a quick simulation:

```python
from qectostim.codes.small.steane_713 import SteanCode713
from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
from qectostim.codes.composite.concatenated import ConcatenatedTopologicalCSSCode
from qectostim.experiments.memory import CSSMemoryExperiment
from qectostim.noise.models import CircuitDepolarizingNoise

# Build a memory experiment for a concatenated code.
outer_exp = RotatedSurfaceCode(distance=3)
inner_exp = SteanCode713()
concat_exp = ConcatenatedTopologicalCSSCode(outer_exp, inner_exp)

noise = CircuitDepolarizingNoise(p1=0.001, p2=0.001)
exp = CSSMemoryExperiment(code=concat_exp, noise_model=noise, rounds=1)
results = exp.run(shots=2000)
print(f"Logical error rate: {results.ler:.4f}")
```

For Stim circuit generation and low-level access, see the `qectostim.experiments` modules.

## Feature Status (snapshot)

- Base CSS code classes (surface, Steane, Shor, Reed-Muller): `Done`
- Stim circuit generation for memory experiments: `Done`
- MWPM decoding via PyMatching: `Done`
- Benchmarks for LER / NDR: `Done`
- Composite code constructs (concatenated, dual, subcode): `Done`
- Fault-tolerant gate gadgets (teleportation, surgery): `WIP`

## Documentation

User docs and tutorials are available at: https://github.com/scottjones03/qec-lib (Docs coming via GitHub Pages)

## Ethical Notice

This project is intended for academic, educational, and civilian research in fault-tolerant quantum computing. Please see `ETHICAL_NOTICE.md` for details and responsible-use guidance.

## Contributing

Contributions and issues are welcome. Please read `CONTRIBUTING.md` for contribution guidelines, coding standards, and testing instructions.

## Community

Join the discussion by opening issues or pull requests on GitHub. If you maintain a mailing list, chat, or meeting schedule, add links here to help collaborators get involved.

## Roadmap

- Add more decoder backends 
- Expand supported code families 
- Implement a full logical-gate gadget library (Clifford + T)
- Performance optimisations and parallel simulation support
- Publish documentation site and step-by-step tutorials

## Running examples and benchmarks

```
============================================================================================================================================
LER COMPARISON TABLE (p=0.01) - ALL CODE TYPES
============================================================================================================================================

Total codes in all_codes: 60
Total codes in full_results: 60

Lower is better. Best decoder for each code highlighted.
SKIP = decoder incompatible (e.g., Chromobius requires color-code DEMs)

Code                                | Type       |  d |  No-decode | PyMatching | FusionBlos | BeliefMatc |      BPOSD |  Tesseract |  UnionFind |        MLE | Hypergraph | Chromobius | Best
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
BaconShor_3x3                       | Subsystem  |  3 |     0.0900 |     0.0160 |     0.0270 |     0.0190 |     0.0210 |     0.0180 |     0.0200 |     0.0140 |     0.1010 |       SKIP | MLE
BalancedProduct_5x5_G1              | QLDPC      |  4 |     0.0180 |     0.0040 |     0.0060 |     0.0060 |     0.0070 |     0.0060 |     0.0040 |       FAIL |     0.0560 |       SKIP | PyMatching
BalancedProduct_7x7_G1              | QLDPC      |  5 |     0.0400 |     0.0410 |     0.0600 |     0.0270 |     0.0230 |     0.0130 |     0.0600 |       FAIL |     0.0470 |       SKIP | Tesseract
BareAncilla_713                     | Non-CSS    |  3 |     0.0940 |     0.0940 |     0.1410 |     0.1120 |     0.1380 |     0.0880 |     0.0700 |     0.0660 |     0.0960 |       SKIP | MLE
C6                                  | CSS        |  2 |     0.0710 |     0.0710 |     0.0990 |     0.0930 |     0.1010 |     0.0730 |     0.0810 |     0.0580 |     0.0810 |       SKIP | MLE
CampbellDoubleHGP_3_[[13,1,None]]   | QLDPC      |  ? |     0.3270 |     0.0260 |     0.0200 |     0.0130 |     0.0180 |     0.0220 |     0.0180 |       FAIL |     0.1730 |       SKIP | BeliefMatc
CampbellDoubleHGP_5_[[41,1,None]]   | QLDPC      |  ? |     0.5220 |     0.0430 |     0.0350 |     0.0390 |     0.0320 |     0.0310 |     0.0230 |       FAIL |     0.3470 |       SKIP | UnionFind
Code_832                            | CSS        |  2 |     0.0750 |     0.0090 |     0.0340 |     0.0070 |     0.0140 |     0.0110 |     0.0050 |     0.0030 |     0.0050 |       SKIP | MLE
Colour488_[[9,1,3]]                 | CSS        |  3 |     0.1920 |     0.0360 |     0.0780 |     0.0610 |     0.0790 |     0.0190 |     0.0230 |     0.0370 |     0.0130 |       SKIP | Hypergraph
DLV_8                               | QLDPC      |  3 |     0.0450 |     0.0460 |     0.0600 |     0.0310 |     0.0130 |     0.0200 |     0.0430 |       FAIL |     0.0440 |       SKIP | BPOSD
DoublePinCode_d3                    | CSS        |  3 |     0.3280 |     0.0160 |     0.0180 |     0.0200 |     0.0140 |     0.0100 |     0.0240 |       FAIL |     0.1670 |       SKIP | Tesseract
DoublePinCode_d5                    | CSS        |  5 |     0.4870 |     0.0240 |     0.0280 |     0.0350 |     0.0280 |     0.0270 |     0.0290 |       FAIL |     0.3480 |       SKIP | PyMatching
FourQubit422_[[4,2,2]]              | CSS        |  2 |     0.0610 |     0.0610 |     0.0520 |     0.0450 |     0.0420 |     0.0610 |     0.0430 |     0.0510 |     0.0510 |       SKIP | BPOSD
GaugeColor_3                        | Subsystem  |  3 |     0.0160 |     0.0160 |     0.0410 |     0.0170 |     0.0250 |     0.0170 |     0.0310 |     0.0210 |     0.0280 |       SKIP | PyMatching
GoldenCode_5                        | CSS        |  ? |     0.4880 |     0.0270 |     0.0240 |     0.0290 |     0.0350 |     0.0290 |     0.0250 |       FAIL |     0.3880 |       SKIP | FusionBlos
HDX_4                               | QLDPC      |  4 |     0.0560 |     0.0320 |     0.0190 |     0.0120 |     0.0140 |     0.0230 |     0.0250 |       FAIL |     0.0230 |       SKIP | BeliefMatc
HGPHamming7_[[58,16,None]]          | QLDPC      |  ? |     0.0850 |     0.1200 |     0.1190 |     0.0510 |     0.0360 |     0.0260 |     0.1260 |       FAIL |     0.2650 |       SKIP | Tesseract
HGPRep5_[[41,1,None]]               | QLDPC      |  ? |     0.1480 |     0.0060 |     0.0060 |     0.0090 |     0.0040 |     0.0040 |     0.0060 |       FAIL |     0.2280 |       SKIP | BPOSD
Hamming_CSS_15                      | CSS        |  3 |     0.0930 |     0.2880 |     0.1940 |     0.1920 |     0.0570 |     0.0600 |     0.2980 |     0.0670 |     0.2050 |       SKIP | BPOSD
Hamming_CSS_31                      | CSS        |  3 |     0.0970 |     0.4310 |     0.2240 |     0.3610 |     0.1100 |     0.1280 |     0.4590 |       FAIL |     0.3990 |       SKIP | BPOSD
Hamming_CSS_7                       | CSS        |  3 |     0.1010 |     0.0830 |     0.0710 |     0.0550 |     0.0640 |     0.0140 |     0.0740 |     0.0200 |     0.0970 |       SKIP | Tesseract
HexagonalColour_d2                  | Color      |  2 |     0.1310 |     0.0000 |     0.0000 |     0.0000 |     0.0000 |     0.0000 |     0.0000 |     0.0110 |     0.0690 |     0.0000 | PyMatching
HexagonalColour_d3                  | Color      |  3 |     0.1460 |     0.1460 |     0.1360 |     0.1490 |     0.1440 |     0.1470 |     0.1330 |       FAIL |     0.1300 |     0.1410 | Hypergraph
Honeycomb_2x3                       | Floquet    |  2 |     0.0940 |     0.0800 |     0.1290 |     0.1400 |     0.1100 |     0.0880 |     0.0990 |     0.0690 |     0.1070 |       SKIP | MLE
Hyperbolic38Code                    | CSS        |  3 |     0.4660 |     0.0200 |     0.0170 |     0.0230 |     0.0270 |     0.0290 |     0.0330 |       FAIL |     0.2590 |       SKIP | FusionBlos
Hyperbolic45Code                    | CSS        |  4 |     0.4470 |     0.0110 |     0.0320 |     0.0270 |     0.0270 |     0.0240 |     0.0190 |       FAIL |     0.2680 |       SKIP | PyMatching
Hyperbolic57Code                    | CSS        |  5 |     0.3230 |     0.0240 |     0.0170 |     0.0200 |     0.0140 |     0.0170 |     0.0250 |       FAIL |     0.1880 |       SKIP | BPOSD
HyperbolicColorCode                 | CSS        |  4 |     0.1730 |     0.0150 |     0.0190 |     0.0110 |     0.0160 |     0.0160 |     0.0200 |       FAIL |     0.0100 |       SKIP | Hypergraph
HyperbolicSurfaceCode               | CSS        |  4 |     0.4470 |     0.0290 |     0.0300 |     0.0210 |     0.0260 |     0.0220 |     0.0160 |       FAIL |     0.2560 |       SKIP | UnionFind
LCSCode                             | CSS        |  3 |     0.4980 |     0.2390 |     0.4570 |     0.3870 |     0.4140 |     0.2180 |     0.2250 |       FAIL |     0.2100 |       SKIP | Hypergraph
Mixed_512                           | Non-CSS    |  2 |     0.2120 |     0.2120 |     0.2020 |     0.1980 |     0.1840 |     0.2160 |     0.1960 |     0.1860 |     0.2410 |       SKIP | BPOSD
NonCSS_1023                         | Non-CSS    |  3 |     0.0140 |     0.0140 |     0.0120 |     0.0090 |     0.0090 |     0.0040 |     0.0080 |     0.0030 |     0.0080 |       SKIP | MLE
NonCSS_642                          | Non-CSS    |  2 |     0.0890 |     0.0890 |     0.2020 |     0.1720 |     0.2250 |     0.1060 |     0.0990 |     0.1080 |     0.1010 |       SKIP | PyMatching
Perfect_513                         | Non-CSS    |  3 |     0.3520 |     0.2850 |     0.2360 |     0.2540 |     0.2640 |     0.2050 |     0.3020 |     0.2060 |     0.2820 |       SKIP | Tesseract
ProjectivePlaneSurface_[[13,1,None]] | CSS        |  ? |     0.3600 |     0.1050 |     0.1160 |     0.0400 |     0.0230 |     0.0170 |     0.1090 |       FAIL |     0.2030 |       SKIP | Tesseract
ProjectivePlane_4_[[25,1,None]]     | CSS        |  ? |     0.4600 |     0.1530 |     0.1640 |       FAIL |     0.0260 |     0.0240 |     0.1550 |       FAIL |     0.3650 |       SKIP | Tesseract
QuantumPinCode_d3_m2                | CSS        |  3 |     0.3310 |     0.0210 |     0.0230 |     0.0150 |     0.0140 |     0.0190 |     0.0100 |       FAIL |     0.2010 |       SKIP | UnionFind
QuantumPinCode_d5_m3                | CSS        |  5 |     0.4780 |     0.0440 |     0.0390 |     0.0300 |     0.0310 |     0.0280 |     0.0300 |       FAIL |     0.3430 |       SKIP | Tesseract
QuantumTanner_4                     | QLDPC      |  2 |     0.0440 |     0.0150 |     0.0110 |     0.0080 |     0.0130 |     0.0080 |     0.0120 |       FAIL |     0.0110 |       SKIP | BeliefMatc
RainbowCode_L3_r3                   | CSS        |  3 |     0.3260 |     0.0220 |     0.0160 |     0.0160 |     0.0130 |     0.0160 |     0.0120 |       FAIL |     0.1890 |       SKIP | UnionFind
RainbowCode_L5_r4                   | CSS        |  5 |     0.4640 |     0.0300 |     0.0360 |     0.0300 |     0.0360 |     0.0310 |     0.0390 |       FAIL |     0.3230 |       SKIP | PyMatching
ReedMuller_15_1_3                   | CSS        |  3 |     0.1410 |     0.0770 |     0.0620 |     0.0660 |     0.0450 |     0.0180 |     0.0650 |       FAIL |     0.0930 |       SKIP | Tesseract
Repetition_3                        | CSS        |  3 |     0.0140 |     0.0020 |     0.0020 |     0.0010 |     0.0010 |     0.0000 |     0.0030 |     0.0010 |     0.0130 |       SKIP | Tesseract
Repetition_5                        | CSS        |  5 |     0.0050 |     0.0000 |     0.0000 |     0.0000 |     0.0000 |     0.0000 |     0.0000 |     0.0000 |     0.0140 |       SKIP | PyMatching
Repetition_7                        | CSS        |  7 |     0.0130 |     0.0000 |     0.0000 |     0.0000 |     0.0000 |     0.0000 |     0.0000 |     0.0000 |     0.0140 |       SKIP | PyMatching
RotatedSurface_[[25,1,5]]           | CSS        |  5 |     0.1630 |     0.0090 |     0.0060 |     0.0130 |     0.0080 |     0.0080 |     0.0030 |       FAIL |     0.2050 |       SKIP | UnionFind
RotatedSurface_[[9,1,3]]            | CSS        |  3 |     0.0800 |     0.0200 |     0.0120 |     0.0200 |     0.0220 |     0.0130 |     0.0120 |     0.0150 |     0.1180 |       SKIP | FusionBlos
Shor_91                             | CSS        |  3 |     0.1900 |     0.0080 |     0.0050 |     0.0030 |     0.0050 |     0.0050 |     0.0010 |     0.0140 |     0.1100 |       SKIP | UnionFind
Steane_713                          | CSS        |  3 |     0.0940 |     0.0630 |     0.0740 |     0.0710 |     0.0570 |     0.0220 |     0.0870 |     0.0250 |     0.0800 |       SKIP | Tesseract
SubsystemSurface_3                  | Subsystem  |  3 |     0.0520 |     0.0510 |     0.1060 |     0.1030 |     0.0770 |     0.0480 |     0.0380 |     0.0460 |     0.0450 |       SKIP | UnionFind
SubsystemSurface_5                  | Subsystem  |  5 |     0.0870 |     0.0640 |     0.0940 |     0.0820 |     0.1030 |     0.0500 |     0.0560 |       FAIL |     0.1260 |       SKIP | Tesseract
ToricCode3D_3x3x3                   | CSS        |  3 |        N/A |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL | N/A
ToricCode_3x3                       | CSS        |  3 |     0.1170 |     0.0120 |     0.0040 |     0.0050 |     0.0100 |     0.0050 |     0.0080 |       FAIL |     0.0070 |       SKIP | FusionBlos
ToricCode_5x5                       | CSS        |  5 |     0.1620 |     0.0190 |     0.0150 |     0.0150 |     0.0220 |     0.0150 |     0.0180 |       FAIL |     0.0080 |       SKIP | Hypergraph
TriangularColour_d3                 | Color      |  3 |     0.0900 |     0.0950 |     0.0740 |     0.0630 |     0.0580 |     0.0250 |     0.0760 |     0.0200 |     0.0940 |     0.0150 | Chromobius
TriangularColour_d5                 | Color      |  5 |     0.2280 |     0.1690 |     0.2040 |       FAIL |     0.0790 |     0.0570 |     0.1880 |       FAIL |     0.1920 |       SKIP | Tesseract
TruncatedTrihexColorCode            | CSS        |  5 |     0.1670 |     0.0030 |     0.0070 |     0.0080 |     0.0040 |     0.0080 |     0.0070 |       FAIL |     0.2680 |       SKIP | PyMatching
TwistedToric_4x4_[[32,2,None]]      | CSS        |  ? |     0.5030 |     0.0130 |     0.0080 |     0.0040 |     0.0050 |     0.0070 |     0.0070 |       FAIL |     0.0130 |       SKIP | BeliefMatc
XZZX_Surface_3                      | CSS        |  3 |     0.1020 |     0.0250 |     0.0440 |     0.0310 |     0.0460 |     0.0380 |     0.0280 |     0.0280 |     0.0720 |       SKIP | PyMatching
XZZX_Surface_5                      | CSS        |  5 |     0.1510 |     0.0240 |     0.0520 |     0.0390 |     0.0330 |     0.0260 |     0.0250 |       FAIL |     0.0650 |       SKIP | PyMatching
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Total rows: 60
```
Examples live in `src/examples/`. To run the comprehensive diagnostic benchmark:

## Citation

If you use this library in your research, please cite the repository and include attribution until a formal publication is available.


