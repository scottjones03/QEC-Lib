# Comprehensive Quantum Error Correction Library (QEC-Lib)

A modular and extensible Python framework for building, simulating, and decoding quantum error-correcting codes.

---

## Overview

This library provides:

- Extensibility to many stabilizer code families  
- Integration with **Stim** for fast circuit + detector-error-model generation  
- Automatic decoder selection (e.g. PyMatching, Union-Find, Fusion Blossom)  
- Simulation of fault-tolerant operations and memory experiments  
- Benchmarking via **logical error rate (LER)** and **non-detection rate (NDR)** diagnostics  

Example of how you might use it:

```python
from qec.codes import RotatedSurfaceCode
from qec.sim import CSSMemoryExperiment

code = RotatedSurfaceCode(distance=3)
exp = CSSMemoryExperiment(code)
results = exp.run(noise_level=0.01)
print(results.ler)
```

⸻

## Key Capabilities (Current & Evolving)


```
Base CSS code classes (surface, Steane, Shor, Reed-Muller, etc.)				Done

Composite code constructs (concatenated, dual, subcode, etc.)					WIP

Stim circuit generation for memory experiments									Done

MWPM decoding via PyMatching													Done

Fault-tolerant gate gadgets														Roadmap (teleportation + CSS code surgery)

Benchmarks for LER, LER-no-decode & NDR											Done
```


```
============================================================================================================================================
LER COMPARISON TABLE (p=0.01) - ALL CODE TYPES
============================================================================================================================================

Total codes in all_codes: 65
Total codes in full_results: 65

Lower is better. Best decoder for each code highlighted.
SKIP = decoder incompatible (e.g., Chromobius requires color-code DEMs)

Code                                | Type       |  d |  No-decode | PyMatching | FusionBlos | BeliefMatc |      BPOSD |  Tesseract |  UnionFind |        MLE | Hypergraph | Chromobius | Best
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
BaconShor_3x3                       | Subsystem  |  3 |     0.0770 |     0.0220 |     0.0150 |     0.0170 |     0.0180 |     0.0170 |     0.0220 |     0.0200 |     0.0910 |       SKIP | FusionBlos
BalancedProduct_5x5_G1              | CSS        |  4 |     0.0240 |     0.0050 |     0.0060 |     0.0030 |     0.0040 |     0.0070 |     0.0030 |       FAIL |     0.0570 |       SKIP | BeliefMatc
BalancedProduct_7x7_G1              | CSS        |  5 |     0.0440 |     0.0390 |     0.0730 |     0.0290 |     0.0150 |     0.0140 |     0.0420 |       FAIL |     0.0440 |       SKIP | Tesseract
BareAncilla_713                     | Non-CSS    |  3 |     0.0820 |     0.0820 |     0.1250 |     0.1220 |     0.1210 |     0.0800 |     0.0710 |     0.0980 |     0.0630 |       SKIP | Hypergraph
C6                                  | CSS        |  2 |     0.0690 |     0.0690 |     0.1000 |     0.0740 |     0.0980 |     0.0730 |     0.0790 |     0.0860 |     0.0890 |       SKIP | PyMatching
CampbellDoubleHGP_3_[[13,1,None]]   | QLDPC      |  ? |     0.3150 |     0.0150 |     0.0180 |     0.0240 |     0.0250 |     0.0200 |     0.0190 |       FAIL |     0.1830 |       SKIP | PyMatching
CampbellDoubleHGP_5_[[41,1,None]]   | QLDPC      |  ? |     0.4840 |     0.0270 |     0.0310 |     0.0340 |     0.0420 |     0.0320 |     0.0310 |       FAIL |     0.3400 |       SKIP | PyMatching
ChamonCode_3                        | CSS        |  3 |     0.4870 |     0.4850 |     0.4910 |     0.4600 |     0.4430 |     0.2830 |     0.4810 |       FAIL |     0.5280 |       SKIP | Tesseract
CheckerboardCode_4                  | CSS        |  2 |     0.3330 |     0.3430 |     0.3390 |     0.2980 |     0.1940 |     0.1720 |     0.3280 |       FAIL |     0.3610 |       SKIP | Tesseract
Code_832                            | CSS        |  2 |     0.0710 |     0.0060 |     0.0420 |     0.0090 |     0.0100 |     0.0070 |     0.0080 |     0.0070 |     0.0110 |       SKIP | PyMatching
Colour488_[[9,1,3]]                 | CSS        |  3 |     0.2040 |     0.0290 |     0.0960 |     0.0800 |     0.0760 |     0.0220 |     0.0330 |     0.0460 |     0.0280 |       SKIP | Tesseract
DLV_8                               | CSS        |  3 |     0.0540 |     0.0480 |     0.0650 |     0.0200 |     0.0180 |     0.0150 |     0.0440 |       FAIL |     0.0530 |       SKIP | Tesseract
FibonacciFractalCode_4              | CSS        |  4 |     0.4300 |     0.0440 |     0.0390 |     0.0370 |     0.0330 |     0.0350 |     0.0420 |       FAIL |     0.2880 |       SKIP | BPOSD
FibonacciFractalCode_5              | CSS        |  5 |     0.4750 |     0.0580 |     0.0620 |     0.0620 |     0.0580 |     0.0410 |     0.0600 |       FAIL |     0.4600 |       SKIP | Tesseract
FourQubit422_[[4,2,2]]              | CSS        |  2 |     0.0460 |     0.0460 |     0.0590 |     0.0450 |     0.0480 |     0.0360 |     0.0600 |     0.0470 |     0.0450 |       SKIP | Tesseract
GKPSurface_[[25,1,?]]               | CSS        |  ? |        N/A |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL | N/A
GKPSurface_[[61,1,?]]               | CSS        |  ? |        N/A |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL | N/A
GaloisQuditColor_[[25,1,?]]         | CSS        |  ? |        N/A |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL | N/A
GaloisQuditSurface_[[13,1,?]]       | CSS        |  ? |        N/A |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL | N/A
GaloisQuditSurface_[[25,1,?]]       | CSS        |  ? |        N/A |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL | N/A
GaugeColor_3                        | Subsystem  |  3 |     0.0230 |     0.0230 |     0.0240 |     0.0210 |     0.0190 |     0.0280 |     0.0170 |     0.0210 |     0.0220 |       SKIP | UnionFind
HDX_4                               | CSS        |  4 |     0.0510 |     0.0160 |     0.0270 |     0.0230 |     0.0220 |     0.0220 |     0.0210 |       FAIL |     0.0200 |       SKIP | PyMatching
HGPHamming7_[[58,16,None]]          | QLDPC      |  ? |     0.0830 |     0.1300 |     0.1240 |     0.0650 |     0.0220 |     0.0240 |     0.1260 |       FAIL |     0.2780 |       SKIP | BPOSD
HaahCode_3                          | CSS        |  3 |     0.4770 |       FAIL |       FAIL |       FAIL |     0.2360 |     0.1910 |       FAIL |       FAIL |       FAIL |       SKIP | Tesseract
Hamming_CSS_7                       | CSS        |  3 |     0.0970 |     0.0770 |     0.0690 |     0.0740 |     0.0720 |     0.0210 |     0.0840 |     0.0180 |     0.0880 |       SKIP | MLE
HexagonalColour_d2                  | Color      |  2 |     0.1340 |     0.0000 |     0.0000 |     0.0000 |     0.0000 |     0.0000 |     0.0000 |     0.0080 |     0.0550 |     0.0000 | PyMatching
HexagonalColour_d3                  | Color      |  3 |     0.1310 |     0.1310 |     0.1400 |     0.1400 |     0.1570 |     0.1370 |     0.1350 |       FAIL |     0.1400 |     0.1550 | PyMatching
HomologicalRotor_[[18,1,?]]         | CSS        |  ? |        N/A |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL | N/A
HomologicalRotor_[[50,1,?]]         | CSS        |  ? |        N/A |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL | N/A
Honeycomb_2x3                       | Floquet    |  2 |     0.0820 |     0.0750 |     0.1100 |     0.1150 |     0.1320 |     0.0610 |     0.0830 |     0.0730 |     0.0990 |       SKIP | Tesseract
HyperbolicColorCode                 | CSS        |  4 |     0.1920 |     0.0220 |     0.0150 |     0.0090 |     0.0110 |     0.0110 |     0.0180 |       FAIL |     0.0140 |       SKIP | BeliefMatc
HyperbolicSurfaceCode               | CSS        |  4 |     0.4350 |     0.0220 |     0.0160 |     0.0260 |     0.0220 |     0.0150 |     0.0220 |       FAIL |     0.2520 |       SKIP | Tesseract
IntegerHomologyBosonic_[[25,1,?]]   | CSS        |  ? |        N/A |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL | N/A
IntegerHomologyBosonic_[[41,1,?]]   | CSS        |  ? |        N/A |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL | N/A
LCSCode                             | CSS        |  3 |     0.4970 |     0.2200 |     0.4490 |     0.3680 |     0.3970 |     0.2130 |     0.2300 |       FAIL |     0.2360 |       SKIP | Tesseract
Mixed_512                           | Non-CSS    |  2 |     0.1980 |     0.1980 |     0.2160 |     0.1930 |     0.2080 |     0.2260 |     0.2050 |     0.2050 |     0.2400 |       SKIP | BeliefMatc
ModularQuditColorCode_L3_d3         | CSS        |  ? |     0.4430 |     0.0330 |     0.0230 |     0.0170 |     0.0120 |     0.0180 |     0.0230 |       FAIL |     0.2770 |       SKIP | BPOSD
ModularQuditSurfaceCode_3x3_d3      | CSS        |  3 |     0.3320 |     0.0150 |     0.0160 |     0.0240 |     0.0230 |     0.0080 |     0.0200 |       FAIL |     0.1780 |       SKIP | Tesseract
ModularQuditSurfaceCode_4x4_d5      | CSS        |  4 |     0.4520 |     0.0180 |     0.0220 |     0.0160 |     0.0180 |     0.0170 |     0.0170 |       FAIL |     0.2630 |       SKIP | BeliefMatc
NonCSS_1023                         | Non-CSS    |  3 |     0.0070 |     0.0070 |     0.0070 |     0.0030 |     0.0090 |     0.0070 |     0.0080 |     0.0040 |     0.0060 |       SKIP | BeliefMatc
NonCSS_642                          | Non-CSS    |  2 |     0.0970 |     0.0970 |     0.1930 |     0.2040 |     0.2200 |     0.1210 |     0.0900 |     0.1090 |     0.1100 |       SKIP | UnionFind
Perfect_513                         | Non-CSS    |  3 |     0.3510 |     0.3060 |     0.2600 |     0.2740 |     0.2520 |     0.2120 |     0.2860 |     0.2250 |     0.3010 |       SKIP | Tesseract
ProjectivePlaneSurface_[[13,1,None]] | CSS        |  ? |     0.3700 |     0.1200 |     0.1170 |     0.0360 |     0.0180 |     0.0260 |     0.1190 |       FAIL |     0.2080 |       SKIP | BPOSD
QuantumTanner_4                     | CSS        |  2 |     0.0550 |     0.0100 |     0.0130 |     0.0140 |     0.0080 |     0.0060 |     0.0050 |       FAIL |     0.0070 |       SKIP | UnionFind
RainbowCode_L3_r3                   | CSS        |  3 |     0.3110 |     0.0170 |     0.0160 |     0.0140 |     0.0230 |     0.0200 |     0.0140 |       FAIL |     0.1820 |       SKIP | BeliefMatc
RainbowCode_L5_r4                   | CSS        |  5 |     0.4870 |     0.0320 |     0.0290 |     0.0270 |     0.0310 |     0.0310 |     0.0200 |       FAIL |     0.3340 |       SKIP | UnionFind
ReedMuller_15_1_3                   | CSS        |  3 |     0.1540 |     0.0930 |     0.0910 |     0.0480 |     0.0390 |     0.0130 |     0.0690 |       FAIL |     0.0720 |       SKIP | Tesseract
Repetition_3                        | CSS        |  3 |     0.0100 |     0.0000 |     0.0000 |     0.0010 |     0.0000 |     0.0020 |     0.0020 |     0.0010 |     0.0120 |       SKIP | PyMatching
Repetition_5                        | CSS        |  5 |     0.0100 |     0.0000 |     0.0000 |     0.0000 |     0.0000 |     0.0000 |     0.0000 |     0.0000 |     0.0110 |       SKIP | PyMatching
Repetition_7                        | CSS        |  7 |     0.0110 |     0.0000 |     0.0000 |     0.0000 |     0.0000 |     0.0000 |     0.0000 |     0.0000 |     0.0080 |       SKIP | PyMatching
RotatedSurface_[[25,1,5]]           | CSS        |  5 |     0.1840 |     0.0110 |     0.0080 |     0.0090 |     0.0090 |     0.0060 |     0.0070 |       FAIL |     0.1810 |       SKIP | Tesseract
RotatedSurface_[[9,1,3]]            | CSS        |  3 |     0.1060 |     0.0180 |     0.0240 |     0.0120 |     0.0100 |     0.0130 |     0.0160 |     0.0150 |     0.0980 |       SKIP | BPOSD
Shor_91                             | CSS        |  3 |     0.2140 |     0.0060 |     0.0020 |     0.0080 |     0.0030 |     0.0030 |     0.0060 |     0.0100 |     0.1100 |       SKIP | FusionBlos
SierpinskiPrismCode_3_2             | CSS        |  4 |     0.4880 |     0.2020 |     0.1920 |     0.1710 |     0.1790 |     0.2060 |     0.1950 |       FAIL |     0.4960 |       SKIP | BeliefMatc
Steane_713                          | CSS        |  3 |     0.0960 |     0.0740 |     0.0960 |     0.0700 |     0.0660 |     0.0210 |     0.0770 |     0.0180 |     0.1000 |       SKIP | MLE
SubsystemSurface_3                  | Subsystem  |  3 |     0.0440 |     0.0530 |     0.0960 |     0.0830 |     0.0780 |     0.0350 |     0.0400 |     0.0300 |     0.0500 |       SKIP | MLE
SubsystemSurface_5                  | Subsystem  |  5 |     0.0880 |     0.0580 |     0.1170 |     0.0870 |     0.1020 |     0.0560 |     0.0540 |       FAIL |     0.1170 |       SKIP | UnionFind
ToricCode_3x3                       | CSS        |  3 |     0.1210 |     0.0140 |     0.0100 |     0.0090 |     0.0050 |     0.0120 |     0.0130 |       FAIL |     0.0140 |       SKIP | BPOSD
ToricCode_5x5                       | CSS        |  5 |     0.1860 |     0.0180 |     0.0270 |     0.0160 |     0.0160 |     0.0240 |     0.0150 |       FAIL |     0.0220 |       SKIP | UnionFind
TriangularColour_d3                 | Color      |  3 |     0.0840 |     0.0900 |     0.0750 |     0.0780 |     0.0650 |     0.0140 |     0.0740 |     0.0200 |     0.0810 |     0.0190 | Tesseract
TriangularColour_d5                 | Color      |  5 |     0.2570 |     0.1890 |     0.1980 |       FAIL |     0.0680 |     0.0730 |     0.1760 |       FAIL |     0.1840 |       SKIP | BPOSD
TruncatedTrihexColorCode            | CSS        |  5 |     0.1770 |     0.0110 |     0.0050 |     0.0020 |     0.0030 |     0.0040 |     0.0060 |       FAIL |     0.3040 |       SKIP | BeliefMatc
XCubeCode_3                         | CSS        |  3 |        N/A |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL |       FAIL | N/A
XZZX_Surface_3                      | CSS        |  3 |     0.0910 |     0.0290 |     0.0400 |     0.0360 |     0.0420 |     0.0270 |     0.0290 |     0.0380 |     0.0600 |       SKIP | Tesseract
XZZX_Surface_5                      | CSS        |  5 |     0.1760 |     0.0370 |     0.0340 |     0.0300 |     0.0310 |     0.0350 |     0.0340 |       FAIL |     0.0720 |       SKIP | BeliefMatc
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Total rows: 65
```
⸻

## Architecture & Design

Base CSS Code Classes

Currently supported (or planned):

	•	RotatedSurfaceCode(d)
	
	•	FourQubit422Code ([[4,2,2]])
	
	•	SteanCode713 ([[7,1,3]])
	
	•	ShorCode91 ([[9,1,3]])
	
	•	ReedMuller151 ([[15,1,3]])
	
	•	GenericCSSCode(Hx, Hz) — allows specifying custom CSS codes from parity-check matrices

Each code object exposes:
```
code.n        # number of physical qubits  
code.k        # number of logical qubits  
code.d        # distance (if known)  
code.Hx, code.Hz  
code.logical_ops  
```
Composite & Transform Classes (Roadmap)

	•	ConcatenatedCode — multi-level encoding to increase distance
	•	DualCode — swap X/Z structure of a CSS code (useful for transversal logic)
	•	Subcode, GaugeFixedCode, etc., to construct subcodes or gauge-fixed versions
	•	HomologicalProductCode — for building QLDPC or hypergraph-product codes

⸻

Fault-Tolerance Goals

We plan to support:

	•	Transversal gates, where available (e.g. Steane or 4-qubit code)
	•	Teleportation-based logical Clifford gates — for codes where transversal gates aren’t available
	•	General CSS-code surgery for universal CNOT between arbitrary CSS codes
	•	Mixed-code workflows (e.g. color code → surface code teleportation)

This aims to support a flexible and universal fault-tolerant computing framework.

⸻

Getting Started
```
git clone https://github.com/scottjones03/qec-lib.git
cd qec-lib
pip install -r requirements.txt
```
Example usage:
```
from qec.codes import RotatedSurfaceCode
from qec.sim import CSSMemoryExperiment, DepolarizingNoise

code = RotatedSurfaceCode(distance=3)
exp = CSSMemoryExperiment(code, rounds=3, noise_model=DepolarizingNoise(p=0.01))
results = exp.run(shots=5000)
print(results)
```
To run the diagnostic benchmark:

python examples/comprehensive_diagnostic.py


⸻

Roadmap

	•	Expand supported code families (LDPC, color codes, Bacon-Shor, 3D gauge codes)
	•	Add more decoder backends (Fusion Blossom, BP+OSD, etc.)
	•	Implement full logical-gate gadget library (Clifford + T)
	•	Performance optimisations & parallel simulation support
	•	Documentation website / tutorials / Jupyter notebooks

⸻

Contributing

Contributions are welcome!
Please open issues or pull requests for new codes, decoders, benchmarks or documentation improvements.

⸻

Citation

If you use this library in your work, please cite this repository (or include attribution) until a formal publication is available.

