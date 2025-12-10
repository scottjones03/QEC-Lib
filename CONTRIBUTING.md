Contributing to QEC-Lib
=======================

Thank you for your interest in contributing to QEC-Lib. We welcome bug reports, feature
requests, documentation improvements, and code contributions.

How to get started
------------------

1. Fork the repository and clone your fork.
2. Create a feature branch: `git checkout -b feat/your-feature`.
3. Run tests locally and keep changes small and well-scoped.
4. Open a pull request against `main` and describe the intent and any testing done.

Development Environment
-----------------------

We recommend creating a virtual environment and installing the requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Testing
-------

- Run unit tests with `pytest`.
- Add tests for new features or bug fixes.

Code Style
----------

- Follow PEP8 for Python code. Use `black` for formatting where appropriate.
- Keep public APIs stable when possible; document breaking changes in PRs.

Pull Request Guidelines
-----------------------

- Provide a clear description and link to any related issues.
- Include tests or examples demonstrating the change.
- Maintain backwards compatibility where feasible; if not, document the
  breaking change and rationale.

Security and Responsible Disclosure
----------------------------------

If you discover a security or privacy issue, please open a private issue and mark it
as sensitive so maintainers can coordinate a responsible disclosure.

Questions
---------

If you're unsure how to contribute, open an issue describing what you'd like to work on
and a maintainer can help triage and assign.
