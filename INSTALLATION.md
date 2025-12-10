Installation
============

This page contains additional installation tips and troubleshooting advice for QEC-Lib.

Install from Git
----------------

To install the latest version directly from GitHub:

```bash
python -m pip install git+https://github.com/scottjones03/qec-lib.git
```

Local development
-----------------

1. Clone the repository and create a virtual environment:

```bash
git clone https://github.com/scottjones03/qec-lib.git
cd qec-lib
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies and editable package:

```bash
pip install -r requirements.txt
pip install -e .
```

Platform notes
--------------

- macOS: Xcode command-line tools may be required for compiling extension dependencies.
- Linux: ensure `build-essential` (or distro equivalent) and Python dev headers are
  available for building wheels.

Common issues
-------------

- If a dependency fails to build: check the error message for missing system libraries
  (e.g. `libffi`, `libssl`) and install the corresponding development packages.
- If `stim` or other performance packages are missing, install them into the same
  environment used to run the library.

Verifying installation
----------------------

Start an interactive Python session in the activated venv and run:

```python
import qec
print(qec.__version__)
```

If that imports without error you are ready to run examples in `examples/`.
