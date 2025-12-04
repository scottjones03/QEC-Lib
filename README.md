📦 Comprehensive Quantum Error Correction Library (QEC-Lib)

A modular and extensible Python framework for building, simulating, and decoding quantum error-correcting codes.

⸻

🔍 Overview

This repository provides a rapidly growing quantum error correction library focused on:

✔ Extensibility to many stabilizer code families
✔ Integration with Stim for fast circuit + detector error model generation
✔ Automatic decoder selection (PyMatching, Union-Find, Fusion Blossom, …)
✔ Simulation of fault-tolerant operations and memory experiments
✔ Benchmarking via logical error rate (LER) and non-detection rate (NDR) diagnostics

The goal is to unify cutting-edge QEC research tools under a common Python API:

from qec.codes import RotatedSurfaceCode
from qec.sim import CSSMemoryExperiment

code = RotatedSurfaceCode(distance=3)
exp = CSSMemoryExperiment(code)
results = exp.run(noise_level=0.01)
print(results.ler)


⸻

✨ Key Capabilities (Current & Evolving)

Capability	Status
Base CSS code classes (Surface, Steane, Shor, Reed-Muller, etc.)	✔
Composite code constructs (Concatenated, Dual, Subcode…)	🔧 WIP
Stim circuit generation for memory experiments	✔
MWPM decoding via PyMatching	✔
Fault-tolerant gate gadgets	🚧 Roadmap (teleportation + CSS code surgery)
Benchmarks for LER, LER-no-decode & NDR	✔


⸻

🧱 Library Architecture

Base Code Classes (CSS)

Already implemented or planned:
	•	RotatedSurfaceCode(d)
	•	FourQubit422Code ([[4,2,2]])
	•	SteanCode713 ([[7,1,3]])
	•	ShorCode91 ([[9,1,3]])
	•	ReedMuller151 ([[15,1,3]])
	•	GenericCSSCode(Hx, Hz) — load any new CSS code from literature 🎯

These expose:

code.n    # physical qubits
code.k    # logical qubits
code.d    # distance (if known)
code.Hx, code.Hz
code.logical_ops

Composite Code / Transformations (Roadmap)

Feature	Purpose
ConcatenatedCode	Boost distance by multi-level encoding
DualCode	Swap X/Z structure — useful for transversal logic
Subcode	Freeze logical DOF, align surgery geometry
GaugeFixedCode	Convert subsystem → stabilizer form
HomologicalProductCode	Explore QLDPC & hypergraph product codes


⸻

🎯 Fault-Tolerance Ambitions

Planned support for:

Technique	Scope
Transversal gates	Where available (e.g. Steane, 4-qubit)
Teleportation-based logical Clifford gates	Universal across same-code blocks
General CSS-code surgery for universal CNOT	Arbitrary CSS ⟶ CSS entangling operations
Mixed-code workflows	Example: color code → surface code teleportation

These align with emerging universal FT architectures (e.g., Poirson et al., 2025).

⸻

🚀 Getting Started

git clone https://github.com/<yourname>/qec-lib.git
cd qec-lib
pip install -r requirements.txt

Basic usage:

from qec.codes import RotatedSurfaceCode
from qec.sim import CSSMemoryExperiment, DepolarizingNoise

code = RotatedSurfaceCode(distance=3)
exp = CSSMemoryExperiment(code, rounds=3, noise_model=DepolarizingNoise(p=0.01))
results = exp.run(shots=5000)
print(results)

To reproduce diagnostics:

python examples/comprehensive_diagnostic.py


⸻

📅 Roadmap
	•	⬆ Expand code families: LDPC, color codes, Bacon-Shor, 3D gauge color
	•	⬆ More decoder backends: Fusion Blossom, BP+OSD
	•	🧩 Full logical Clifford + T gadget library
	•	🧠 Performance optimizations + parallel simulation
	•	📚 Jupyter tutorials + documentation website

⸻

📎 References & Prior Art
	•	Stim — high-performance stabilizer simulator
	•	PyMatching / Fusion Blossom — MWPM decoders
	•	Cowtan & Burton, and Poirson et al. (2024–25) — universal CSS code surgery
	•	Standard stabilizer / CSS code theory literature

⸻

🤝 Contributing

Contributions welcome!
Please open PRs for new codes, decoders, benchmarks, or tutorials.

⸻

⭐ Citation

If this library supports your research, please cite this repository until a paper is available.
