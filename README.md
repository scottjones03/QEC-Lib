# Comprehensive Quantum Error Correction Library (QEC-Lib)

A modular and extensible Python framework for building, simulating, and decoding quantum error-correcting codes.

---

# QEC-Lib — Comprehensive Quantum Error Correction Library

> This project is actively evolving — documentation and APIs may change rapidly. If you find inconsistencies or have questions, please open an issue: https://github.com/scottjones03/qec-lib/issues/new/choose

![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue)
[![build](https://github.com/scottjones03/qec-lib/actions/workflows/ci.yml/badge.svg)](https://github.com/scottjones03/qec-lib/actions/workflows/ci.yml)
[![docs](https://github.com/scottjones03/qec-lib/actions/workflows/gh-pages.yml/badge.svg)](https://github.com/scottjones03/qec-lib/actions/workflows/gh-pages.yml)
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

Generate a memory experiment for a rotated surface code and run a quick simulation:

```python
from qec.codes import RotatedSurfaceCode
from qec.sim import CSSMemoryExperiment, DepolarizingNoise

code = RotatedSurfaceCode(distance=3)
noise = DepolarizingNoise(p=0.01)
exp = CSSMemoryExperiment(code, rounds=3, noise_model=noise)
results = exp.run(shots=2000)
print(f"Logical error rate: {results.ler:.4f}")
```

For Stim circuit generation and low-level access, see the `qec.io` and `qec.sim` modules.

## Feature Status (snapshot)

- Base CSS code classes (surface, Steane, Shor, Reed-Muller): Done
- Stim circuit generation for memory experiments: Done
- MWPM decoding via PyMatching: Done
- Benchmarks for LER / NDR: Done
- Composite code constructs (concatenated, dual, subcode): WIP
- Fault-tolerant gate gadgets (teleportation, surgery): Roadmap

## Documentation

User docs and tutorials are available at: https://github.com/scottjones03/qec-lib (Docs coming via GitHub Pages)

## Ethical Notice

This project is intended for academic, educational, and civilian research in fault-tolerant quantum computing. Please see `ETHICAL_NOTICE.md` for details and responsible-use guidance.

## Contributing

Contributions and issues are welcome. Please read `CONTRIBUTING.md` for contribution guidelines, coding standards, and testing instructions.

## Community

Join the discussion by opening issues or pull requests on GitHub. If you maintain a mailing list, chat, or meeting schedule, add links here to help collaborators get involved.

## Roadmap

- Add more decoder backends (Fusion Blossom, BP+OSD)
- Expand supported code families (color codes, LDPC / QLDPC, Bacon-Shor)
- Implement a full logical-gate gadget library (Clifford + T)
- Performance optimisations and parallel simulation support
- Publish documentation site and step-by-step tutorials

## Running examples and benchmarks

Examples live in `examples/`. To run the comprehensive diagnostic benchmark:

```bash
python examples/comprehensive_diagnostic.py
```

## Citation

If you use this library in your research, please cite the repository and include attribution until a formal publication is available.

---
Files added: `ETHICAL_NOTICE.md`, `CONTRIBUTING.md`, `INSTALLATION.md`.
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

