"""
End-to-end demo and test suite for the trapped-ion QCCD framework.

Modules
-------
run
    Standalone runner that compiles, visualises, and (optionally)
    simulates a d=2 rotated surface code on both Augmented Grid and
    WISE architectures.  Produces MP4 animations, static PNG figures,
    and prints compilation / simulation metrics.

test_e2e
    Pytest test suite exercising the full pipeline:
    build_ideal_circuit → compile → display → animate → noise → simulate.
"""
