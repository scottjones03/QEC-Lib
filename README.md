# QECToStim
Python-based quantum error-correction library that is extensible to many code families and integrates with simulation and decoding tools. This library interfaces with Stim (a fast stabilizer simulator) for circuit generation and automatically select appropriate decoders (e.g. PyMatching, Union-Find) based on the code’s syndrome structure. 

# QECToStim

**QECToStim** is an extensible Python library for constructing, composing, simulating, and decoding a wide range of quantum error‑correcting (QEC) codes. It provides a unified framework for:

- Building **base codes** (surface codes, color codes, [[4,2,2]] code, generic CSS codes).
- Constructing **composite codes** (concatenated codes, subcodes, dual codes, gauge‑fixed codes, homological product codes).
- Generating **Stim circuits** for memory experiments and fault‑tolerant logical gate gadgets.
- Automatically selecting the best available **decoder** for a given code and detector error model (PyMatching, Fusion Blossom, Union‑Find, etc.).
- Synthesizing **fault‑tolerant operations** including transversal gates, teleportation‑based logical Cliffords, and universal logical CNOTs via *CSS surgery*.

QECToStim aims to be a research‑grade toolkit that bridges abstract code theory, stabilizer algebra, and real‑world circuit‑level simulation.

---

## 🚀 Features

### 1. Base Code Library
QECToStim ships with implementations of widely‑used stabilizer and CSS codes.

- **Rotated Surface Code**  
  Arbitrary distance, automatic stabilizer layout, X/Z plaquette structure.

- **[[4,2,2]] “Little Shor” Code**  
  Distance‑2 CSS code useful as an inner detection code or concatenation component.

- **2D Color Code**  
  Tricolorable lattice with transversal Clifford gates.

- **Generic CSS Code from Parity‑Check Matrices**  
  Constructor that accepts `Hx` and `Hz` and validates commutation.  
  Useful for arbitrary LDPC codes, hypergraph product codes, published codes, etc.

All base codes expose:
- `n`, `k` (physical/logical qubits)
- `Hx`, `Hz`
- Full stabilizer group
- Logical X/Z operators

---

### 2. Composite Code Framework
QECToStim supports algebraic operations over codes, producing new code objects.

**Composite code types include:**

- **ConcatenatedCode(outer, inner)**  
  Constructs a full stabilizer description of the concatenated code, mapping outer logicals into inner encodings.

- **DualCode(code)**  
  Swaps X/Z checks to produce the dual CSS code.

- **Subcode(code, freeze_logical=...)**  
  Creates a subcode by turning a logical operator into a stabilizer (e.g. [[4,2,2]] → [[4,1,2]]).

- **GaugeFixedCode(subsystem_code, gauge_ops)**  
  Converts subsystem codes into stabilizer codes via gauge fixing.

- **HomologicalProductCode(codeA, codeB)**  
  Builds quantum LDPC codes from two input CSS codes (hypergraph product / homological tensor product).

Each composite code inherits the standard stabilizer interface, making them fully compatible with circuit generation and decoding.

---

### 3. Experiment → Stim Circuit Tooling
Central to the library is the `Experiment` class, which converts any `Code` object into a **Stim circuit**.

Initial support includes:

- **Memory experiments:**  
  - Logical |0⟩ or |+⟩ preparation  
  - Repeated syndrome extraction  
  - Configurable noise models  
  - Automatic DETECTORS and OBSERVABLES

Planned extensions:

- **Logical gate experiments**  
  (logical CNOT, encoded Clifford synthesis, teleportation, CSS surgery gadgets).

This makes it easy to benchmark logical error rates or generate training data for decoders.

---

### 4. Automatic Decoder Selection
QECToStim examines the **Stim Detector Error Model (DEM)** and chooses an appropriate decoder:

- **PyMatching (MWPM)**  
  For graph‑like DEMs (surface codes, color codes, concatenated codes).

- **Fusion Blossom**  
  High‑performance MWPM implementation for large codes.

- **Union‑Find Decoder**  
  Extremely fast approximate decoder for surface‑like or LDPC‑like codes.

- **Custom decoders**  
  Support for plugging in belief‑propagation, neural decoders, or exact ML decoding for small codes.

Users may override the decoder choice manually.

---

## 🧰 Fault‑Tolerant Gadget Library

QECToStim includes a growing library of fault‑tolerant logical operations.

### ✔ Teleportation Gadgets (Logical Cliffords)
Supports Clifford operations between two blocks of the **same code type** via encoded Bell‑pair preparation and Bell‑basis measurements.  
Useful when transversal gates are unavailable or undesirable.

### ✔ Universal Logical CNOT for Any CSS Code
Implements the **general CSS surgery protocol**:  
A powerful 2024–2025 framework that constructs a fault‑tolerant logical CNOT between _any_ two CSS codes using subcode measurements and stabilizer merging.

This is code‑agnostic and geometry‑independent.

### ✔ Transversal Logical Gates
For codes that admit transversal operations (e.g., color codes, Steane, [[4,2,2]] H/S), QECToStim generates the corresponding physical gate patterns automatically.

---

## 📦 Installation

(Coming soon – PyPI package and documentation)

For development:
```
git clone https://github.com/<your‑repo>/QECToStim
cd QECToStim
pip install -e .
```

---

## 📖 Example Usage

```python
from qec_to_stim import RotatedSurfaceCode, Experiment

code = RotatedSurfaceCode(distance=3)
exp  = Experiment(code, rounds=20, noise_model="circuit_depolarizing")

circuit = exp.to_stim()
result  = exp.run_decode(circuit)

print(result.logical_error_rate)
```

---

## 🛣️ Roadmap

### Milestone 1 — Core Infrastructure
- Base code classes  
- Composite code classes  
- Memory experiments → Stim  
- PyMatching integration

### Milestone 2 — Fault‑Tolerant Gates
- Transversal gate interface  
- Teleportation gadgets  
- CSS surgery CNOT

### Milestone 3 — Advanced Codes & Decoders
- Homological product codes  
- Additional decoders (Fusion Blossom, UF, BP)  
- Benchmarking + profiling tools

---

## 🤝 Contributing
Contributions are welcome! The library aims to become a community standard for QEC code simulation and circuit synthesis.

---

## 📜 License
MIT License (or your chosen license).

---

## ✨ Acknowledgements
QECToStim draws inspiration from:
- Stim and PyMatching  
- Fusion Blossom  
- Recent research on CSS code surgery and homological product codes  
- The broader QEC research community
