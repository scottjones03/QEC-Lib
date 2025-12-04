# 📦 Comprehensive Quantum Error Correction Library (QEC-Lib)

A modular and extensible Python framework for building, simulating, and decoding quantum error-correcting codes.

---

## 🔍 Overview

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

✨ Key Capabilities (Current & Evolving)


Base CSS code classes (surface, Steane, Shor, Reed-Muller, etc.)					✅

Composite code constructs (concatenated, dual, subcode, etc.)						🔧 WIP

Stim circuit generation for memory experiments										✅

MWPM decoding via PyMatching														✅

Fault-tolerant gate gadgets															🚧 Roadmap (teleportation + CSS code surgery)

Benchmarks for LER, LER-no-decode & NDR												✅


⸻

🧱 Architecture & Design

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

🎯 Fault-Tolerance Goals

We plan to support:

	•	Transversal gates, where available (e.g. Steane or 4-qubit code)
	•	Teleportation-based logical Clifford gates — for codes where transversal gates aren’t available
	•	General CSS-code surgery for universal CNOT between arbitrary CSS codes
	•	Mixed-code workflows (e.g. color code → surface code teleportation)

This aims to support a flexible and universal fault-tolerant computing framework.

⸻

🚀 Getting Started
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

📅 Roadmap

	•	Expand supported code families (LDPC, color codes, Bacon-Shor, 3D gauge codes)
	•	Add more decoder backends (Fusion Blossom, BP+OSD, etc.)
	•	Implement full logical-gate gadget library (Clifford + T)
	•	Performance optimisations & parallel simulation support
	•	Documentation website / tutorials / Jupyter notebooks

⸻

🤝 Contributing

Contributions are welcome!
Please open issues or pull requests for new codes, decoders, benchmarks or documentation improvements.

⸻

⭐ Citation

If you use this library in your work, please cite this repository (or include attribution) until a formal publication is available.

---

If you like, I can also add a **Table of Contents** at the top to improve navigation (helpful for long READMEs).
