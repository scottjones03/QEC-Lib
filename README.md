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

# Repository Structure

QECToStim/
├─ pyproject.toml          # or setup.cfg/setup.py
├─ README.md
├─ LICENSE
├─ .gitignore
├─ src/
│  └─ qec_to_stim/
│     ├─ __init__.py
│     ├─ codes/
│     │  ├─ __init__.py
│     │  ├─ base/
│     │  │  ├─ __init__.py
│     │  │  ├─ rotated_surface.py
│     │  │  ├─ four_qubit_422.py
│     │  │  ├─ color_code.py
│     │  │  ├─ generic_css.py
│     │  │  └─ steane_713.py          # (optional later)
│     │  ├─ composite/
│     │  │  ├─ __init__.py
│     │  │  ├─ concatenated.py
│     │  │  ├─ dual.py
│     │  │  ├─ subcode.py
│     │  │  ├─ gauge_fixed.py
│     │  │  └─ homological_product.py
│     │  ├─ complexes/
│     │  │  ├─ __init__.py
│     │  │  ├─ chain_complex.py       # abstract chain complex
│     │  │  ├─ css_complex.py         # 3-chain CSS complexes (Hx, Hz)
│     │  │  └─ importers.py           # load complexes from files/formats
│     │  └─ abstract_code.py          # Code + HomologicalCode + CSSCode base classes
│     │
│     ├─ experiments/
│     │  ├─ __init__.py
│     │  ├─ base.py                   # Experiment base class
│     │  ├─ memory.py                 # MemoryExperiment
│     │  ├─ logical_gates.py          # LogicalGateExperiment
│     │  └─ gadget_experiment.py      # GadgetExperiment
│     │
│     ├─ gadgets/
│     │  ├─ __init__.py
│     │  ├─ teleportation.py          # teleportation-based Cliffords
│     │  ├─ css_surgery.py            # general CSS CNOT gadget
│     │  └─ transversal.py            # transversal gates
│     │
│     ├─ decoders/
│     │  ├─ __init__.py
│     │  ├─ base.py                   # Decoder interface
│     │  ├─ decoder_selector.py       # DEM → choose backend
│     │  ├─ pymatching_decoder.py
│     │  ├─ fusion_blossom_decoder.py
│     │  ├─ union_find_decoder.py
│     │  └─ custom_decoders.py        # hooks for BP / NN / exact, etc.
│     │
│     ├─ noise/
│     │  ├─ __init__.py
│     │  ├─ models.py                 # logical/physical noise model classes
│     │  └─ stim_noise.py             # translate models → Stim noise instructions
│     │
│     ├─ io/
│     │  ├─ __init__.py
│     │  ├─ stim_io.py                # helpers to read/write Stim circuits/DEM
│     │  └─ code_formats.py           # import/export codes (JSON, YAML, etc.)
│     │
│     └─ utils/
│        ├─ __init__.py
│        ├─ pauli.py
│        ├─ symplectic.py
│        ├─ logging.py
│        └─ profiling.py
│
└─ tests/
   ├─ __init__.py
   ├─ test_codes/
   ├─ test_experiments/
   ├─ test_gadgets/
   ├─ test_decoders/
   └─ data/                  # small reference Hx/Hz matrices, etc.

docs/
├─ index.md
├─ design.md                 # your long proposal, API docs
└─ examples/
   ├─ rotated_surface_memory.md
   └─ concatenated_422_surface.md


# Plan for a Comprehensive Quantum Error Correction Code Library

## Overview and Goals

We propose a Python-based quantum error-correction (QEC) library that is extensible to many code families and integrates with simulation and decoding tools. The near-term goal is to implement a broad set of stabilizer codes and fault-tolerant operations, ensuring the design can scale to “all QEC” needs. The library will provide a clear Python API for constructing code objects (from built-in templates or custom parity-check matrices) and executing fault-tolerant protocols. Crucially, it will interface with Stim (a fast stabilizer simulator) for circuit generation and automatically select appropriate decoders (e.g. PyMatching, Union-Find) based on the code’s syndrome structure. The following plan outlines key components.

---

## Base Code Classes (CSS Codes)

We will include a variety of base codes (fundamental QEC codes) as classes, each representing a specific stabilizer code with known parameters and properties. These base classes simplify creation of common codes and serve as building blocks for composite codes. Planned base code classes include:

- **`RotatedSurfaceCode`** – Represents the rotated surface (planar) code defined on a 2D lattice. For example, a distance-\(d\) rotated surface code has qubits on the vertices of a rotated square grid (45° orientation) and alternating \(X/Z\) plaquette stabilizers. The class will allow specifying distance and boundary types, and it will internally generate the appropriate stabilizer parity-check matrices. (The rotated surface code is a commonly used high-threshold CSS code; in fact, the smallest instance of a rotated toric code is the \([[4,2,2]]\) code.)

- **`FourQubitCode` (aka `[[4,2,2]]`)** – Implements the 4-qubit code (also known as the \(C_4\) “Little Shor” code). This is the smallest CSS code that encodes 2 logical qubits and detects a single error. Its stabilizers are `XXXX` and `ZZZZ`, with parity-check matrices \(H_X = H_Z = (1,1,1,1)\). Despite its limited distance \(d = 2\), this code is important as a building block and example; it is essentially a \([[4,2,2]]\) detection code that can be viewed as a tiny planar code or toric code patch. The class will provide this code’s structure and logical operators.

- **`ColorCode`** – Provides the family of color codes (starting with the 2D color code on a hexagonal or square-octagon lattice). Color codes are CSS codes defined on \(D\)-dimensional, \((D+1)\)-colorable lattices. For example, the 2D color code has qubits on vertices of a tricolorable tiling (e.g. 4.8.8 lattice) and faces of two colors defining \(X\)- and \(Z\)-checks. The class might allow specifying lattice size/geometry or use a standard instance (such as the 17-qubit \([[17,1,5]]\) 2D color code). Color codes are notable for high transversal gate possibilities (see **Transversal Gates** below). Initially, we will implement a 2D color code class (with distance corresponding to lattice size), with potential to generalize to 3D and higher dimensions later.

- **`GenericCSSCode`** – A general class that constructs a code from user-specified CSS parity-check matrices. The user can provide \(H_X\) and \(H_Z\) (as binary matrices or in a file format), and the library will form a `CSSCode` object. Internally, it will verify the commutation condition \(H_X H_Z^\mathsf{T} = 0 \bmod 2\) (ensuring each \(X\)-stabilizer commutes with each \(Z\)-stabilizer). This allows arbitrary new codes (e.g. \([[n,k,d]]\) parameters from literature) to be instantiated. The class will expose properties like `.n`, `.k`, `.d` (length, dimension, distance if known) and the full stabilizer matrix. In line with other tools, the stabilizer matrix will be stored in a binary symplectic form, and the class will provide accessors for the separate \(X\) and \(Z\) check matrices (`code.hx` and `code.hz`). This means from a computational perspective, *“a code is entirely described by its parity-check matrix”*. Having a generic CSS code class ensures extensibility – any Calderbank–Shor–Steane code can be integrated into our framework by inputting the two parity-check matrices.

> Additional base codes like the 5-qubit code or Steane \([[7,1,3]]\) code could be included as needed, using the `GenericCSSCode` under the hood. However, priority is given to the above codes that align with later fault-tolerant gadget plans.

---

## Composite and Derived Code Classes

Beyond basic codes, the library will support composite codes – codes constructed from or related to other codes. We will design classes for common code composition techniques and transformations:

- **`ConcatenatedCode`** – Represents a concatenated coding scheme, where an “outer” code’s logical qubits are each encoded into an “inner” code. The class will take two code objects (outer and inner) and produce a new code object. It will compute parameters like
  \[
  n_{\text{new}} = n_{\text{outer}} \times n_{\text{inner}}
  \]
  and aim for distance
  \[
  d_{\text{new}} \ge d_{\text{outer}} \times d_{\text{inner}}.
  \]
  Concatenation is a fundamental technique for boosting distance by multi-level encoding. The class will handle constructing the combined stabilizer list (essentially the tensor product of the two codes’ check matrices in an appropriate way). We’ll also support recursive concatenation (e.g. encoding multiple levels deep) possibly via an argument that specifies number of levels. This enables experiments with concatenated fault-tolerance schemes (e.g. Shor’s code concatenated with itself, etc.).

- **`DualCode`** – A class to derive the dual of a given CSS code. In classical coding, the dual code has the original parity-check matrix as its generator matrix. For CSS stabilizer codes, we define the `DualCode` such that the input code’s \(X\) and \(Z\) check matrices are swapped to create a new code. Essentially, if the original code has checks \(H_X\) and \(H_Z\), the dual code uses \(H_X' = H_Z\) and \(H_Z' = H_X\). Provided the original was a valid CSS code (orthogonal checks), the dual will also satisfy \(H_X' {H_Z'}^\mathsf{T} = 0\) due to the same orthogonality. This operation can change the interpretation of logical operators (what was an \(X\) logical in the original becomes a \(Z\) logical in the dual, and vice versa). The `DualCode` class is useful for exploring code properties; for example, a code might not have a certain transversal gate in its native form, but its dual might have a transversal implementation of that gate (swapping \(X \leftrightarrow Z\) often swaps which logical operators are transversal). The library will make it easy to obtain the dual of any CSS code object.

- **`Subcode`** – Enables constructing a subcode of a given code by adding new stabilizer constraints (or equivalently, fixing some logical degrees of freedom). For instance, from the \([[4,2,2]]\) code (which has 2 logical qubits), one can pick one logical qubit and constrain it (say to logical \(|0\rangle\) state), yielding a \([[4,1,2]]\) subcode. In general, if a code has \(k > 0\) logical qubits, we can introduce an extra stabilizer (often one of the code’s logical operators or a combination) to reduce the number of logical qubits by 1. The `Subcode` class will take an existing code and either (a) an index of a logical qubit to freeze, or (b) an explicit operator (Pauli string) to promote to a stabilizer. It will output a new code with one fewer logical qubit. This operation can sometimes increase distance or enable certain fault-tolerant operations on the subcode. (Subcodes also play a role in code surgery protocols as described later.) By supporting subcodes, our library aligns with recent methods that treat *“subcodes [as] encapsulating all necessary data for performing surgery”* in generalized lattice surgery protocols.

- **`GaugeFixedCode`** – A class to convert a subsystem code (with gauge qubits) into a regular stabilizer code by **gauge fixing**. Gauge fixing means choosing a set of gauge operators to measure and treat as new stabilizers, thereby “freezing” those degrees of freedom. For example, the Bacon–Shor subsystem code can be gauge-fixed into either a surface code or a compass code depending on which gauge checks are enforced. Similarly, a 3D gauge color code can be gauge-fixed to a 2D color code or surface code by measuring appropriate gauge operators (this is how one can achieve a transversal \(T\) gate and then convert the state to a surface code state for error correction in some protocols). The `GaugeFixedCode` class will accept a subsystem code object plus a list of gauge operators to fix; it will output a new stabilizer `Code` object where those operators are now part of the stabilizer group. This reduces the code’s dimension \(k\) (logical qubits), but can simplify certain operations. The class will keep track of which formerly gauge qubit becomes an actual (measurable) syndrome in the new code.

- **`HomologicalProductCode`** – Constructs homological product codes (such as hypergraph product codes) from two input CSS codes. This leverages the idea of treating CSS codes as chain complexes and taking their tensor product in homology. The resulting code is a QLDPC code (quantum low-density parity-check code) with checks derived from combinations of the two input codes’ checks. In practice, if Code A has \((n_A, k_A)\) and Code B has \((n_B, k_B)\), their hypergraph product yields a CSS code with parameters roughly \((n_A n_B + \dots,\; k_A k_B + \dots)\) (specific formulae can be provided). The class will take two code objects and output a new code object with parity-check matrices constructed via the Kronecker-style products of the inputs’ \(H_X, H_Z\). For example, the rotated surface code can be obtained as a hypergraph product of two simple classical codes; more generally, many new quantum LDPC codes arise this way. By including `HomologicalProductCode`, the library will enable exploration of next-generation quantum codes suitable for sparse decoding algorithms. (Note: If inputs are not homological (e.g. not CSS), we will restrict to CSS inputs to ensure output is CSS.)

Each composite code class will implement the standard code interface (with attributes `.n`, `.k`, `.hx`, `.hz`, `.stabilizers`, etc.), either by analytical construction (for known formulae like concatenation) or by performing operations on the parity-check matrices of input code(s).

---

## Integration with Stim and Decoders

A major feature of the library is the integration with Stim for simulating circuits and experiments, as well as hooking into various decoder backends. This will allow users to easily test the error-correcting performance of any code in the library.

### Stim Circuit Generation

Each code object will support generating Stim circuits for common benchmark experiments. Initially, we focus on **memory experiments** for a single logical qubit: i.e. prepare a logical state (\(|0\rangle\) or \(|1\rangle\), or \(|+\rangle\) for phase-flip testing), run rounds of error syndrome extraction (with configurable noise), then measure to see if the logical state is preserved.

The library can provide a method like:

```python
circuit = code.generate_memory_experiment(duration, error_model)
```

that returns a Stim Circuit or DetectorErrorModel. Under the hood, this uses Stim’s capabilities to place noise on gates and add DETECTOR instructions for stabilizer checks. For example, a distance-(d) surface code memory experiment will involve (d) rounds of syndrome measurements and corresponding Stim detectors for any pattern of check results that indicates a logical error. By leveraging Stim’s detector error model output, we can obtain a description of the error syndrome graph for the experiment.

In later iterations, we’ll generalize circuit generation to logical gate experiments as well – e.g. preparing two code blocks and performing a logical CNOT, or other logical operations (teleportation circuits, etc.). The goal is that given any code (base or composite), the library can intelligently compile a fault-tolerant circuit (using that code’s stabilizers and logical ops) for any supported operation.

### Decoder Integration

Once a Stim detector error model (DEM) is obtained for a given experiment, the library will select an appropriate decoder to interpret the syndromes and attempt to correct errors. We plan to integrate all well-known open-source decoders that can consume Stim’s data. The choice of decoder will depend on the code and error model:
	•	For matching-compatible codes (those whose error syndrome graph is mostly composed of pairwise detection events), we will use minimum-weight perfect matching (MWPM) decoders. The primary tool here is PyMatching, a fast Python/C++ implementation of MWPM. PyMatching works on Stim’s detector graph directly, finding the most likely correction by pairing detection nodes. We will incorporate PyMatching to handle planar/surface codes, color codes, and many CSS codes which yield a graph-like syndrome (each error causes one or two detection events). In fact, Stim and PyMatching are designed to work together – e.g. the sinter package combines Stim simulation with PyMatching decoding for efficient threshold estimation. Our library will follow this approach, using Stim+PyMatching to simulate and decode many codes’ error-correction circuits in a fast, parallelized manner.
	•	For cases requiring even faster or alternative MWPM decoders, we aim to support Fusion Blossom (a Rust-based MWPM decoder, with Python bindings). Fusion Blossom is an independent implementation that shares similarities with PyMatching (sparse blossom algorithm) and introduces union-find-like growth of clusters. Integrating Fusion Blossom provides another option, especially beneficial for very large codes where performance is critical. The library can default to PyMatching and allow switching to Fusion Blossom via a flag.
	•	For union-find decoders, we will include support when appropriate. The Union-Find decoder by Delfosse and coworkers is an approximate but very fast decoder for surface and homological product codes. It works by growing clusters of syndromes using a union-find data structure. While not optimal for all errors, it has linear complexity and performs well for low-weight errors. If the code’s detection graph is tree-like or amenable to union-find (e.g. large LDPC codes where a quick heuristic is desired), the library can invoke a union-find decoding routine. We could either use an existing implementation or include a custom one (there are open references and code for union-find decoders).
	•	If needed for more complex codes (e.g. those not having a simple matching structure), we will consider belief-propagation or neural decoders. For instance, small quantum LDPC codes might use a BP+OSD (order statistics decoder) approach, and some newer decoders employ machine learning. Our framework will be designed to allow plugging in any decoder that can input a syndrome or error model and output a correction. In the near term, our focus is on decoders compatible with Stim’s detection event data, which primarily means graph-like decoders (MWPM, UF, etc.), as these are well-developed and open source.

The library’s decoder module will automatically choose a default decoder based on the code. For example:
	•	surface code → MWPM (PyMatching),
	•	color code → MWPM or Union-Find,
	•	concatenated codes → either decode per level or use a brute-force decoder for small inner codes combined with an outer decoder.

Users can override the choice or provide custom decoders. After decoding, the library will report logical error rates or identify logical error occurrences (with tools to interpret stim.DetectorErrorModel outcomes). This tight integration will make it straightforward to evaluate the error-correcting performance of any new code class we add.

⸻

## Fault-Tolerant Logical Gate Gadgets

In addition to memory experiments, our library will support constructing fault-tolerant gadgets for logical operations on encoded qubits. We outline several critical gadgets and how they will be implemented.

### Teleportation-Based Logical Cliffords

We will provide a protocol to perform arbitrary Clifford gates between two code blocks of the same code type via state teleportation. The idea is to use a Bell pair (an entangled logical state) as a resource to teleport operations from one block to another. For example, to enact a logical two-qubit Clifford (such as a CX or CZ) between two ([[4,2,2]]) code blocks, one can prepare an entangled logical Bell state using one qubit from each block, then perform joint measurements that project onto the desired gate operation (with Pauli frame corrections applied based on outcomes).

In practice, this is analogous to lattice surgery on surface code patches: e.g. measuring joint (X) stabilizers or (Z) stabilizers of two blocks can entangle or swap logical information. Our library will include a function (or class) to generate a teleportation circuit given two code objects and a target Clifford gate. Under the hood, it will attach ancilla code blocks if needed, prepare encoded Bell pairs, and measure appropriate stabilizer combinations to achieve the gate. For instance, performing a logical CZ could involve merging two code patches along a boundary (measuring a multi-qubit (Z) stabilizer across them) and then splitting them – a procedure demonstrated as “entangling logical qubits with lattice surgery” in experiments.

All intermediate steps (ancilla preparation, syndrome measurements for the joint operators, etc.) will be done using Stim circuits to maintain error-tracking. This teleportation approach provides a uniform way to enact gates without requiring the codes themselves to support transversal interactions. Initially, we focus on Clifford operations (which only need Pauli measurements and possibly magic state injections for certain Cliffords like CCZ if ever needed). The output gadget circuit will come with its own detection events so that errors during the gate can be decoded by the same decoders chosen for the codes.

Universal CNOT via CSS Code Surgery

A highlight feature will be a gadget for a fault-tolerant CNOT gate between any two logical qubits, in any CSS code from our library. We will implement the recently proposed “CSS code surgery” technique by Cowtan & Burton (2024) and extended by Poirson et al. (2025). In their framework, one uses an intermediate subcode and a sequence of multi-qubit measurements to enact a CNOT between arbitrarily chosen CSS code blocks.

Our plan is to incorporate this by automatically generating the required surgery measurements given two CSSCode objects and specifying which logical qubits (of each) to couple. For example, the protocol as described by Poirson et al. implements a logical CNOT between any two CSS logical qubits with fault-tolerance guarantees. The library will create a combined stabilizer structure (the subcode that overlaps the two codes) and insert Stim commands to measure the necessary joint stabilizers (like summing an (X) from code A and (X) from code B to couple them, etc.).

Because this method does not assume any particular geometry or transversal gate, it is extremely general – it “implements a logical CNOT gate between any two logical qubits of any CSS code”. We will follow the paper’s explicit circuit constructions (which likely involve two rounds of measurements for (X) and (Z) basis coupling) to ensure the operation is fault-tolerant.

This gadget will greatly enhance the library’s capability: users can take, say, a color code and a toric code (both CSS) and perform a logical CNOT between them via this unified surgery approach. It effectively abstracts lattice surgery to the codeword level using algebraic manipulation of stabilizers. Our implementation will manage the intermediate ancilla states (if any) and feed the resulting detection events to appropriate decoders (likely the same decoders for each code, combined appropriately). This feature positions the library at the cutting edge of QEC research.

### Transversal Logical Gates

For code families that admit transversal gates (applying single-qubit unitaries in parallel across the block), we will include straightforward gadgets to perform those operations. The library will have metadata on each code class listing which logical operations are transversal (e.g. the color code has transversal Clifford gates in 2D, and transversal (T) gates in 3D; the 4-qubit code has transversal Hadamards that apply logical Hadamard on both qubits).

If a user requests a logical gate that is known to be transversal for that code, the library can simply apply the corresponding physical gates to all qubits of each code block. For example, for the Steane ([[7,1,3]]) code, a logical Hadamard is just 7 physical Hadamards, and a logical (S) gate is 7 physical (S) gates (Steane is CSS and transversal for all Clifford gates). For the ([[4,2,2]]) code, a tensor product of four (H) gates implements logical Hadamard on both encoded qubits, and a tensor of (S) gates has the effect of a logical CZ between them (plus logical (Z) corrections). Our library will incorporate these known results – e.g. “a tensor product of (S) gates applies a CZ between logical qubits” in the 4-qubit code – and provide a one-line method to apply a transversal gate.

We will verify the operation by checking that it preserves the stabilizer group and transforms logical operators correctly (possibly using the code’s automorphism if available). Transversal gates are fault-tolerant by construction (no two qubits of the same code block interact during the gate), so no special syndrome extraction is needed during their application (only afterwards, as usual). However, if desired, the library could output a Stim circuit snippet for the transversal gate (which would just be parallel single-qubit operations, and possibly flagging any resulting Pauli frame change).

⸻

By supporting these three categories of gate implementations (teleportation, code-agnostic surgery, and transversal), the library will enable universal fault-tolerant computing across the codes. A possible use-case is to demonstrate a logical circuit composed of different codes: e.g. use a color code for T-state injection (leveraging its transversal (T) gate), then teleport that state into a surface code for storage, then perform a CNOT between the surface code qubit and a Steane code qubit using the general CSS surgery gadget, etc. The framework will make such heterogeneous schemes possible through a unified interface.

⸻

## Extensibility and Implementation Notes

### Extensibility

A core design principle is to make the library easily extensible to new codes and decoders. By having the GenericCSSCode class and composition classes, researchers can plug in novel codes (e.g. a newly discovered ([[n,k,d]]) CSS code from a paper) simply by providing its stabilizer checks. The decoder interface will allow adding new decoding algorithms without altering the code classes – for example, if a neural network decoder for surface codes becomes available, one could integrate it as a decoder option for SurfaceCode classes.

We also plan to include an error model interface so that different noise models (depolarizing, biased noise, etc.) can be specified consistently when generating Stim circuits, as this may influence decoder choice (e.g. use a biased decoder for XZZX code with biased noise).

---

# Python API

The library will be written in Python for accessibility and rapid development. Users will interact with high-level classes and functions, for example:

code = RotatedSurfaceCode(distance=5)
circuit = code.generate_memory_experiment(T=100, error_model=my_noise_model)
result = code.run_decoder(circuit)

Under the hood, performance-critical parts (like decoder calls or large matrix operations) may use optimized libraries (NumPy/SciPy or C++ extensions). Python’s flexibility is advantageous for integrating with Stim’s Python API and other tools like PyMatching (also Python/C++). The API will be designed to feel intuitive: constructing a code, running an experiment, and getting a decoding result should only take a few lines in a Jupyter notebook.


---

# 📦 Installation

(Coming soon – PyPI package and documentation)

For development:
```
git clone https://github.com/<your‑repo>/QECToStim
cd QECToStim
pip install -e .
```

---

# 📖 Example Usage

```python
from qec_to_stim import RotatedSurfaceCode, Experiment

code = RotatedSurfaceCode(distance=3)
exp  = Experiment(code, rounds=20, noise_model="circuit_depolarizing")

circuit = exp.to_stim()
result  = exp.run_decode(circuit)

print(result.logical_error_rate)
```

---

# 🤝 Contributing
Contributions are welcome! The library aims to become a community standard for QEC code simulation and circuit synthesis.

---

# 📜 License
MIT License (or your chosen license).

---
